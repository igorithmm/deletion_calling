#!/usr/bin/env python3
"""Generate a fused-pipeline dataset: pileup images + per-sample metadata.

This script implements **Steps 1–3** of the CADC pipeline:

    Step 1  Feature extraction & noise filtering
            (read depth + clipping signals, 61-bp median filter)
    Step 2  Candidate finding & breakpoint refinement
            (3-cluster K-means on smoothed features → exact breakpoints)
    Step 3  Image generation
            (50-bp sliding window → 256×256 RGB pileup)

For every 50-bp window it writes:
  * a PNG image to ``<out_dir>/<sample>/<deletion|non_deletion>/<filename>.png``
  * a row to ``<out_dir>/<sample>/manifest.csv`` with columns:
      image_path, chrom, position, label, length, neg_type

Negative-sampling strategies (``--neg-strategy``):
  anchor  — paired up/down flanks only (original behaviour).
  mixed   — balanced 1:4 strategy. If a deletion renders N positive images,
            render up to N images from each negative source:
            down anchor, up anchor, regional, and random.

The manifest is the **source of truth** consumed by:
  * ``train_fused_model.py``  — builds ``FusedDataset`` from these rows
  * ``call_fused_deletions.py`` — runs inference on these rows

Example (anchor strategy — original)
--------------------------------------
    python3 scripts/generate_fused_dataset.py \\
        --sample NA12878 \\
        --bam raw/NA12878.bam \\
        --vcf raw/ALL.wgs.integrated_sv_map_v2.20130502.svs.genotypes.vcf.gz \\
        --chroms 20,21,22 --output-dir data/fused

Example (mixed / E3 strategy)
--------------------------------------
    python3 scripts/generate_fused_dataset.py \\
        --sample NA12878 \\
        --bam raw/NA12878.bam \\
        --vcf raw/ALL.wgs.integrated_sv_map_v2.20130502.svs.genotypes.vcf.gz \\
        --chroms 20,21,22 \\
        --neg-strategy mixed \\
        --neg-regional-distance-kb 200 \\
        --output-dir data/fused_mixed
"""
from __future__ import annotations

import argparse
import csv
import logging
import numpy as np
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Make the package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import Variant, VCFHandler
from deepsv.processing.refinement import BoundaryRefiner
from deepsv.visualization.image_generator import ImageGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


WINDOW_BP = 50  # window size — must match the embeddings precompute


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WindowRecord:
    """One row in the manifest CSV."""

    image_path: str
    chrom: str
    position: int  # 0-based start of the 50-bp window
    label: int  # 1 = deletion, 0 = non-deletion
    length: int  # parent variant length (bp)
    neg_type: str = ""  # "anchor_up" | "anchor_down" | "regional" | "random" | ""


# ─────────────────────────────────────────────────────────────────────────────
# Image generation per variant
# ─────────────────────────────────────────────────────────────────────────────


def windows_for_variant(
    variant: Variant,
    bam: BAMHandler,
    image_gen: ImageGenerator,
    out_dir: Path,
    label: int,
    neg_type: str = "",
    max_windows: Optional[int] = None,
) -> List[WindowRecord]:
    """Render a 50-bp sliding window across [variant.start, variant.end).

    Returns a list of WindowRecord rows for the manifest. Skips windows
    that produce empty pileups.
    """
    records: List[WindowRecord] = []
    n_windows = max(1, (variant.end - variant.start) // WINDOW_BP)

    for w in range(n_windows):
        if max_windows is not None and len(records) >= max_windows:
            break
        start = variant.start + w * WINDOW_BP
        end = start + WINDOW_BP
        try:
            pileup = bam.get_pileup_data(variant.chrom, start, end)
            if not pileup:
                continue
            clipping = bam.get_clipping_info(variant.chrom, start, end)
            img = image_gen.generate_image(
                pileup_data=pileup,
                clipping_data=clipping,
                region_start=start,
                region_length=WINDOW_BP,
            )
        except Exception as e:
            logger.debug("skipping window %s:%d-%d (%s)", variant.chrom, start, end, e)
            continue

        tag = neg_type or "positive"
        fname = f"{variant.chrom}_{start}_{end}_{label}_{tag}.png"
        out_path = out_dir / fname
        image_gen.save_image(img, str(out_path))
        records.append(
            WindowRecord(
                image_path=str(out_path),
                chrom=variant.chrom,
                position=start,
                label=label,
                length=variant.length,
                neg_type=neg_type,
            )
        )
    return records


def render_target_windows(
    candidates: Sequence[Variant],
    target_n: int,
    bam: BAMHandler,
    image_gen: ImageGenerator,
    out_dir: Path,
    neg_type: str,
) -> List[WindowRecord]:
    """Render up to ``target_n`` negative windows from candidate regions."""
    records: List[WindowRecord] = []
    for candidate in candidates:
        remaining = target_n - len(records)
        if remaining <= 0:
            break
        records.extend(
            windows_for_variant(
                candidate,
                bam,
                image_gen,
                out_dir,
                label=0,
                neg_type=neg_type,
                max_windows=remaining,
            )
        )
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome lengths from BAM header
# ─────────────────────────────────────────────────────────────────────────────


def get_chrom_lengths(bam_path: str) -> Dict[str, int]:
    """Return {chrom: length} from the BAM header (uses pysam)."""
    import pysam

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        return {sq["SN"]: sq["LN"] for sq in bam.header["SQ"]}


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def generate(
    sample: str,
    bam_path: str,
    vcf_path: str,
    chroms: Sequence[str],
    output_dir: str,
    max_length: int = 10_000,
    min_length: int = 50,
    del_count_per_chrom: Optional[int] = None,
    refine_boundaries: bool = True,
    neg_strategy: str = "anchor",
    neg_regional_distance_kb: int = 200,
    neg_random_pool_factor: float = 1.0,
    seed: int = 42,
) -> Path:
    """Run Steps 1–3 end-to-end and write a manifest CSV.

    Args:
        neg_strategy: ``"anchor"`` (original) or ``"mixed"`` (1:4).
        neg_regional_distance_kb: Distance in kb for Level-2 regional negatives.
        neg_random_pool_factor:   Multiplier for random candidate regions tried
                                  per deletion in mixed mode.

    Returns:
        Path to the manifest CSV.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    out_root = Path(output_dir) / sample
    del_dir = out_root / "deletion"
    nondel_dir = out_root / "non_deletion"
    del_dir.mkdir(parents=True, exist_ok=True)
    nondel_dir.mkdir(parents=True, exist_ok=True)

    image_gen = ImageGenerator()
    refiner = BoundaryRefiner() if refine_boundaries else None

    # ── Load variants ────────────────────────────────────────────────────────
    logger.info("Loading variants from %s …", vcf_path)
    vcf = VCFHandler(vcf_path)
    all_variants = vcf.load_variants(variant_type="deletion", sample_id=sample)
    chroms_set = set(str(c) for c in chroms)
    variants = [
        v
        for v in all_variants
        if str(v.chrom) in chroms_set and min_length <= v.length <= max_length
    ]
    logger.info(
        "Kept %d / %d variants (chroms=%s, %d ≤ length ≤ %d)",
        len(variants),
        len(all_variants),
        sorted(chroms_set),
        min_length,
        max_length,
    )

    # Optional per-chromosome cap.
    if del_count_per_chrom is not None:
        capped: List[Variant] = []
        per_chrom: dict = {}
        rng.shuffle(variants)
        for v in variants:
            n = per_chrom.get(v.chrom, 0)
            if n < del_count_per_chrom:
                capped.append(v)
                per_chrom[v.chrom] = n + 1
        variants = capped
        logger.info(
            "Capped to %d deletions (≤ %d per chrom)",
            len(variants),
            del_count_per_chrom,
        )

    # ── Mixed-strategy setup ─────────────────────────────────────────────────
    chrom_lengths: Dict[str, int] = {}
    if neg_strategy == "mixed":
        logger.info(
            "Mixed negative strategy enabled — reading chromosome lengths from BAM …"
        )
        chrom_lengths = get_chrom_lengths(bam_path)
        # Restrict chrom_lengths to the chroms being processed so random
        # negatives don't land on chromosomes we aren't generating images for.
        chrom_lengths = {c: l for c, l in chrom_lengths.items() if c in chroms_set}
        logger.info("  chromosome lengths available for %d chroms", len(chrom_lengths))


    # ── Main loop ────────────────────────────────────────────────────────────
    rows: List[WindowRecord] = []
    n_rejected = 0  # candidates dropped by K-means confirmation

    with BAMHandler(bam_path) as bam:
        for i, variant in enumerate(variants, 1):
            # Step 2: refine breakpoints via depth + clipping K-means.
            # Like legacy call_del, candidates whose depth profile does not
            # show a confirmed deletion signal are dropped entirely.
            if refiner is not None:
                try:
                    refined = refiner.refine_boundaries(bam, variant)
                except Exception as e:
                    logger.debug("refinement failed for %s: %s", variant, e)
                    refined = None

                if refined is None:
                    n_rejected += 1
                    logger.debug(
                        "candidate rejected by K-means: %s:%d-%d",
                        variant.chrom,
                        variant.start,
                        variant.end,
                    )
                    continue
                variant = refined

            # ── Positive-class images ────────────────────────────────────────
            positive_rows = windows_for_variant(
                variant, bam, image_gen, del_dir, label=1
            )
            rows.extend(positive_rows)
            n_positive_windows = len(positive_rows)
            if n_positive_windows == 0:
                continue

            # ── Negative-class images ────────────────────────────────────────
            if neg_strategy == "anchor":
                # Original behaviour: only anchor flanks.
                anchors = vcf.get_non_deletion_regions([variant], anchor_type="up")
                anchors += vcf.get_non_deletion_regions([variant], anchor_type="down")
                for a in anchors:
                    rows.extend(
                        windows_for_variant(
                            a,
                            bam,
                            image_gen,
                            nondel_dir,
                            label=0,
                            neg_type="anchor",
                        )
                    )

            else:  # "mixed" — up to N windows per each negative source.
                target_bp = n_positive_windows * WINDOW_BP

                up_anchor = vcf.get_non_deletion_regions(
                    [variant], anchor_type="up", region_length=target_bp
                )
                rows.extend(
                    render_target_windows(
                        up_anchor,
                        n_positive_windows,
                        bam,
                        image_gen,
                        nondel_dir,
                        neg_type="anchor_up",
                    )
                )

                down_anchor = vcf.get_non_deletion_regions(
                    [variant], anchor_type="down", region_length=target_bp
                )
                rows.extend(
                    render_target_windows(
                        down_anchor,
                        n_positive_windows,
                        bam,
                        image_gen,
                        nondel_dir,
                        neg_type="anchor_down",
                    )
                )

                regional = vcf.sample_regional_negatives(
                    variant=variant,
                    all_variants=all_variants,
                    chrom_lengths=chrom_lengths,
                    distance_kb=neg_regional_distance_kb,
                    win_len=target_bp,
                    rng=rng,
                )
                rows.extend(
                    render_target_windows(
                        regional,
                        n_positive_windows,
                        bam,
                        image_gen,
                        nondel_dir,
                        neg_type="regional",
                    )
                )

                random_candidate_count = max(1, int(4 * neg_random_pool_factor))
                random_candidates = vcf.sample_random_negatives(
                    all_variants=all_variants,
                    chrom_lengths=chrom_lengths,
                    win_len=target_bp,
                    n_windows=random_candidate_count,
                    rng=rng,
                )
                rows.extend(
                    render_target_windows(
                        random_candidates,
                        n_positive_windows,
                        bam,
                        image_gen,
                        nondel_dir,
                        neg_type="random",
                    )
                )

            if i % 25 == 0 or i == len(variants):
                logger.info(
                    "  processed %d/%d variants — %d windows so far",
                    i,
                    len(variants),
                    len(rows),
                )

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "chrom", "position", "label", "length", "neg_type"])
        for r in rows:
            w.writerow(
                [r.image_path, r.chrom, r.position, r.label, r.length, r.neg_type]
            )

    n_pos = sum(1 for r in rows if r.label == 1)
    n_neg = len(rows) - n_pos
    n_anchor = sum(1 for r in rows if r.neg_type in ("anchor", "anchor_up", "anchor_down"))
    n_anchor_up = sum(1 for r in rows if r.neg_type == "anchor_up")
    n_anchor_down = sum(1 for r in rows if r.neg_type == "anchor_down")
    n_regional = sum(1 for r in rows if r.neg_type == "regional")
    n_random = sum(1 for r in rows if r.neg_type == "random")

    if n_rejected:
        logger.info(
            "K-means filtering rejected %d / %d candidates (%.1f%%)",
            n_rejected,
            len(variants),
            100 * n_rejected / max(len(variants), 1),
        )

    logger.info(
        "Wrote %d rows to %s  (pos=%d  neg=%d  "
        "[anchor=%d up=%d down=%d  regional=%d  random=%d])",
        len(rows),
        manifest_path,
        n_pos,
        n_neg,
        n_anchor,
        n_anchor_up,
        n_anchor_down,
        n_regional,
        n_random,
    )
    return manifest_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sample", required=True, help="Sample ID (e.g. NA12878)")
    p.add_argument("--bam", required=True, help="Path to indexed BAM file")
    p.add_argument("--vcf", required=True, help="Path to indexed VCF.gz with SVs")
    p.add_argument(
        "--chroms",
        required=True,
        help="Comma-separated chromosome list (e.g. '20,21,22' or 'chr20,chr21')",
    )
    p.add_argument(
        "--output-dir",
        default="data/fused",
        help="Root directory for images + manifest (default: data/fused)",
    )
    p.add_argument("--max-length", type=int, default=10_000)
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument(
        "--del-count",
        type=int,
        default=None,
        help="Optional cap on deletions per chromosome",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip K-means breakpoint refinement (Step 2)",
    )
    p.add_argument("--seed", type=int, default=42)

    # ── Negative-strategy flags ──────────────────────────────────────────────
    p.add_argument(
        "--neg-strategy",
        choices=["anchor", "mixed"],
        default="anchor",
        help=(
            "Negative-sampling strategy.\n"
            "  anchor — original: only paired flanks of each deletion.\n"
            "  mixed  — render N positives and up to N negatives from each "
            "of: up anchor, down anchor, regional, random."
        ),
    )
    p.add_argument(
        "--neg-regional-distance-kb",
        type=int,
        default=200,
        help="Distance (kb) from deletion for Level-2 regional negatives (default: 200).",
    )
    p.add_argument(
        "--neg-random-pool-factor",
        type=float,
        default=1.0,
        help=(
            "Scaling factor for random candidate regions tried per deletion "
            "in mixed mode (default: 1.0)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    if not chroms:
        raise SystemExit("--chroms produced an empty list")
    generate(
        sample=args.sample,
        bam_path=args.bam,
        vcf_path=args.vcf,
        chroms=chroms,
        output_dir=args.output_dir,
        max_length=args.max_length,
        min_length=args.min_length,
        del_count_per_chrom=args.del_count,
        refine_boundaries=not args.no_refine,
        neg_strategy=args.neg_strategy,
        neg_regional_distance_kb=args.neg_regional_distance_kb,
        neg_random_pool_factor=args.neg_random_pool_factor,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
