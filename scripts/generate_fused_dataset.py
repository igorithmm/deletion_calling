#!/usr/bin/env python3
"""Generate a fused-pipeline dataset: pileup images + per-sample metadata.

This script implements **Steps 1–3** of the DeepSV 3.0 algorithm:

    Step 1  Feature extraction & noise filtering
            (read depth + clipping signals, 61-bp median filter)
    Step 2  Candidate finding & breakpoint refinement
            (3-cluster K-means on smoothed features → exact breakpoints)
    Step 3  Image generation
            (50-bp sliding window → 256×256 RGB pileup)

For every 50-bp window it writes:
  * a PNG image to ``<out_dir>/<sample>/<deletion|non_deletion>/<filename>.png``
  * a row to ``<out_dir>/<sample>/manifest.csv`` with columns:
      image_path, chrom, position, label, length, repeat_class

The manifest is the **source of truth** consumed by:
  * ``train_fused_model.py``  — builds ``FusedDataset`` from these rows
  * ``call_fused_deletions.py`` — runs inference on these rows

The HyenaDNA HDF5 file is **not** read here; it is consumed downstream by
the training / inference scripts. Generation can therefore proceed in
parallel with (or before) the precompute.

Example
-------
    python3 scripts/generate_fused_dataset.py \\
        --sample NA12878 \\
        --bam raw/NA12878.bam \\
        --vcf raw/ALL.wgs.integrated_sv_map_v2.20130502.svs.genotypes.vcf.gz \\
        --chroms 20,21,22 \\
        --max-length 10000 \\
        --del-count 1500 \\
        --output-dir data/fused
"""
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Make the package importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import DeletionSize, Variant, VCFHandler
from deepsv.processing.refinement import BoundaryRefiner
from deepsv.visualization.image_generator import ImageGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


WINDOW_BP = 50  # window size — must match the embeddings precompute


# ─────────────────────────────────────────────────────────────────────────────
# Repeat-class lookup (optional, opt-in via --repeat-bed)
# ─────────────────────────────────────────────────────────────────────────────


def classify_repeat(
    chrom: str,
    start: int,
    end: int,
    repeat_bed: Optional[str],
    simple_repeat_bed: Optional[str],
    segdup_bed: Optional[str],
) -> str:
    """Return one of ``"unique"`` / ``"simple-repeat"`` / ``"segmental-dup"``.

    If ``pybedtools`` is unavailable or no BED files are supplied, returns
    ``"unique"`` for every sample (the safe default — equivalent to "no
    information").
    """
    if not (repeat_bed or simple_repeat_bed or segdup_bed):
        return "unique"
    try:
        import pybedtools  # local import — heavy, optional
    except ImportError:
        return "unique"

    interval = pybedtools.BedTool(f"{chrom}\t{start}\t{end}\n", from_string=True)
    if segdup_bed and Path(segdup_bed).exists():
        if interval.intersect(pybedtools.BedTool(segdup_bed), u=True).count() > 0:
            return "segmental-dup"
    if simple_repeat_bed and Path(simple_repeat_bed).exists():
        if interval.intersect(pybedtools.BedTool(simple_repeat_bed), u=True).count() > 0:
            return "simple-repeat"
    if repeat_bed and Path(repeat_bed).exists():
        if interval.intersect(pybedtools.BedTool(repeat_bed), u=True).count() > 0:
            return "simple-repeat"
    return "unique"


# ─────────────────────────────────────────────────────────────────────────────
# Image generation per variant
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WindowRecord:
    """One row in the manifest CSV."""
    image_path: str
    chrom: str
    position: int       # 0-based start of the 50-bp window
    label: int          # 1 = deletion, 0 = non-deletion
    length: int         # parent variant length (bp)
    repeat_class: str   # unique / simple-repeat / segmental-dup


def windows_for_variant(
    variant: Variant,
    bam: BAMHandler,
    image_gen: ImageGenerator,
    out_dir: Path,
    label: int,
    repeat_class: str,
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
                repeat_class=repeat_class,
            )
        )
    return records


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
    repeat_bed: Optional[str] = None,
    simple_repeat_bed: Optional[str] = None,
    segdup_bed: Optional[str] = None,
    seed: int = 42,
) -> Path:
    """Run Steps 1–3 end-to-end and write a manifest CSV.

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

    # Load deletions from the VCF, filter by length and chromosome.
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

    rows: List[WindowRecord] = []

    with BAMHandler(bam_path) as bam:
        for i, variant in enumerate(variants, 1):
            # Step 2: refine breakpoints via depth + clipping K-means.
            if refiner is not None:
                try:
                    variant = refiner.refine_boundaries(bam, variant)
                except Exception as e:
                    logger.debug("refinement failed for %s: %s", variant, e)

            repeat_cls = classify_repeat(
                variant.chrom, variant.start, variant.end,
                repeat_bed, simple_repeat_bed, segdup_bed,
            )

            # Positive-class images.
            rows.extend(
                windows_for_variant(
                    variant, bam, image_gen, del_dir,
                    label=1, repeat_class=repeat_cls,
                )
            )

            # Negative-class images: paired up/down anchor regions of the
            # same length, on the same chromosome.
            anchors = vcf.get_non_deletion_regions([variant], anchor_type="up")
            anchors += vcf.get_non_deletion_regions([variant], anchor_type="down")
            for a in anchors:
                # The anchor inherits the parent's repeat_class — it's used
                # only for stratified eval grouping.
                rows.extend(
                    windows_for_variant(
                        a, bam, image_gen, nondel_dir,
                        label=0, repeat_class=repeat_cls,
                    )
                )

            if i % 25 == 0 or i == len(variants):
                logger.info(
                    "  processed %d/%d variants — %d windows so far",
                    i, len(variants), len(rows),
                )

    # Write manifest.
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "chrom", "position", "label", "length", "repeat_class"])
        for r in rows:
            w.writerow([r.image_path, r.chrom, r.position, r.label, r.length, r.repeat_class])

    n_pos = sum(1 for r in rows if r.label == 1)
    n_neg = len(rows) - n_pos
    logger.info(
        "Wrote %d rows to %s (positives=%d negatives=%d)",
        len(rows), manifest_path, n_pos, n_neg,
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
    p.add_argument("--repeat-bed", default=None, help="RepeatMasker BED for stratification")
    p.add_argument("--simple-repeat-bed", default=None)
    p.add_argument("--segdup-bed", default=None)
    p.add_argument("--seed", type=int, default=42)
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
        repeat_bed=args.repeat_bed,
        simple_repeat_bed=args.simple_repeat_bed,
        segdup_bed=args.segdup_bed,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
