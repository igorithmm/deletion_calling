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
  mixed   — E3 strategy: 40 % anchor + 30 % regional + 30 % random.
            Regional = same chrom, ~200 kb away.
            Random   = arbitrary genomic position, filtered against known SVs.

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
import random
import sys
from dataclasses import dataclass, field
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
# Mixed-strategy target ratios (E3)
# ─────────────────────────────────────────────────────────────────────────────
ANCHOR_RATIO   = 0.40
REGIONAL_RATIO = 0.30
RANDOM_RATIO   = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WindowRecord:
    """One row in the manifest CSV."""
    image_path: str
    chrom: str
    position: int       # 0-based start of the 50-bp window
    label: int          # 1 = deletion, 0 = non-deletion
    length: int         # parent variant length (bp)
    neg_type: str = ""  # "anchor" | "regional" | "random" | "" (positives)


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
) -> List[WindowRecord]:
    """Render a 50-bp sliding window across [variant.start, variant.end).

    Returns a list of WindowRecord rows for the manifest. Skips windows
    that produce empty pileups.
    """
    records: List[WindowRecord] = []
    n_windows = max(1, (variant.end - variant.start) // WINDOW_BP)

    for w in range(n_windows):
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
            logger.debug(
                "skipping window %s:%d-%d (%s)", variant.chrom, start, end, e
            )
            continue

        fname = f"{variant.chrom}_{start}_{end}_{label}.png"
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
        neg_strategy: ``"anchor"`` (original) or ``"mixed"`` (E3).
        neg_regional_distance_kb: Distance in kb for Level-2 regional negatives.
        neg_random_pool_factor:   Multiplier for number of random negatives
                                  relative to anchor negatives (default 1.0
                                  means equal total count per variant).

    Returns:
        Path to the manifest CSV.
    """
    rng = random.Random(seed)

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
        v for v in all_variants
        if str(v.chrom) in chroms_set
        and min_length <= v.length <= max_length
    ]
    logger.info(
        "Kept %d / %d variants (chroms=%s, %d ≤ length ≤ %d)",
        len(variants), len(all_variants), sorted(chroms_set), min_length, max_length,
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
            len(variants), del_count_per_chrom,
        )

    # ── Mixed-strategy setup ─────────────────────────────────────────────────
    chrom_lengths: Dict[str, int] = {}
    if neg_strategy == "mixed":
        logger.info("Mixed negative strategy enabled — reading chromosome lengths from BAM …")
        chrom_lengths = get_chrom_lengths(bam_path)
        # Restrict chrom_lengths to the chroms being processed so random
        # negatives don't land on chromosomes we aren't generating images for.
        chrom_lengths = {c: l for c, l in chrom_lengths.items() if c in chroms_set}
        logger.info("  chromosome lengths available for %d chroms", len(chrom_lengths))

        # Pre-build the random-negative pool for the whole run.
        # We target RANDOM_RATIO of total negatives.  Since each positive
        # variant produces 2 anchors (up + down), we aim for a proportionally
        # sized random pool distributed across all variants.
        avg_win_len = (
            int(sum(v.length for v in variants) / len(variants))
            if variants else 500
        )
        avg_win_len = min(avg_win_len, 700)  # same 80%-cap as anchor logic
        n_random_total = max(
            1,
            int(len(variants) * 2 * (RANDOM_RATIO / ANCHOR_RATIO) * neg_random_pool_factor),
        )
        logger.info("  pre-sampling %d random genome-wide negatives …", n_random_total)
        random_neg_pool: List[Variant] = vcf.sample_random_negatives(
            all_variants=all_variants,
            chrom_lengths=chrom_lengths,
            win_len=avg_win_len,
            n_windows=n_random_total,
            rng=rng,
        )
        rng.shuffle(random_neg_pool)
        random_pool_iter = iter(random_neg_pool)
        logger.info("  random pool ready: %d windows", len(random_neg_pool))

    # ── Main loop ────────────────────────────────────────────────────────────
    rows: List[WindowRecord] = []

    with BAMHandler(bam_path) as bam:
        for i, variant in enumerate(variants, 1):
            # Step 2: refine breakpoints via depth + clipping K-means.
            if refiner is not None:
                try:
                    variant = refiner.refine_boundaries(bam, variant)
                except Exception as e:
                    logger.debug("refinement failed for %s: %s", variant, e)

            # ── Positive-class images ────────────────────────────────────────
            rows.extend(
                windows_for_variant(variant, bam, image_gen, del_dir, label=1)
            )

            # ── Negative-class images ────────────────────────────────────────
            if neg_strategy == "anchor":
                # Original behaviour: only anchor flanks.
                anchors = vcf.get_non_deletion_regions([variant], anchor_type="up")
                anchors += vcf.get_non_deletion_regions([variant], anchor_type="down")
                for a in anchors:
                    rows.extend(
                        windows_for_variant(
                            a, bam, image_gen, nondel_dir,
                            label=0, neg_type="anchor",
                        )
                    )

            else:  # "mixed" — E3 strategy
                # Level 1: Anchor negatives (40 %)
                anchors = vcf.get_non_deletion_regions([variant], anchor_type="up")
                anchors += vcf.get_non_deletion_regions([variant], anchor_type="down")
                for a in anchors:
                    rows.extend(
                        windows_for_variant(
                            a, bam, image_gen, nondel_dir,
                            label=0, neg_type="anchor",
                        )
                    )

                # Level 2: Regional negatives (30 %) — same chrom, ~distance_kb away
                regional = vcf.sample_regional_negatives(
                    variant=variant,
                    all_variants=all_variants,
                    chrom_lengths=chrom_lengths,
                    distance_kb=neg_regional_distance_kb,
                    rng=rng,
                )
                for r in regional:
                    rows.extend(
                        windows_for_variant(
                            r, bam, image_gen, nondel_dir,
                            label=0, neg_type="regional",
                        )
                    )

                # Level 3: Random genome-wide negatives (30 %)
                # Draw one window from the pre-sampled pool per variant.
                rand_neg = next(random_pool_iter, None)
                if rand_neg is not None:
                    rows.extend(
                        windows_for_variant(
                            rand_neg, bam, image_gen, nondel_dir,
                            label=0, neg_type="random",
                        )
                    )

            if i % 25 == 0 or i == len(variants):
                logger.info(
                    "  processed %d/%d variants — %d windows so far",
                    i, len(variants), len(rows),
                )

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "chrom", "position", "label", "length", "neg_type"])
        for r in rows:
            w.writerow([r.image_path, r.chrom, r.position, r.label, r.length, r.neg_type])

    n_pos = sum(1 for r in rows if r.label == 1)
    n_neg = len(rows) - n_pos
    n_anchor   = sum(1 for r in rows if r.neg_type == "anchor")
    n_regional = sum(1 for r in rows if r.neg_type == "regional")
    n_random   = sum(1 for r in rows if r.neg_type == "random")

    logger.info(
        "Wrote %d rows to %s  (pos=%d  neg=%d  [anchor=%d  regional=%d  random=%d])",
        len(rows), manifest_path, n_pos, n_neg, n_anchor, n_regional, n_random,
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
        "--chroms", required=True,
        help="Comma-separated chromosome list (e.g. '20,21,22' or 'chr20,chr21')",
    )
    p.add_argument(
        "--output-dir", default="data/fused",
        help="Root directory for images + manifest (default: data/fused)",
    )
    p.add_argument("--max-length", type=int, default=10_000)
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument(
        "--del-count", type=int, default=None,
        help="Optional cap on deletions per chromosome",
    )
    p.add_argument(
        "--no-refine", action="store_true",
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
            "  mixed  — E3: 40%% anchor + 30%% regional + 30%% random."
        ),
    )
    p.add_argument(
        "--neg-regional-distance-kb", type=int, default=200,
        help="Distance (kb) from deletion for Level-2 regional negatives (default: 200).",
    )
    p.add_argument(
        "--neg-random-pool-factor", type=float, default=1.0,
        help=(
            "Scaling factor for the genome-wide random pool size "
            "(default: 1.0 = match anchor count)."
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
