#!/usr/bin/env python3
"""
Generate a balanced image dataset for any sample, any set of chromosomes,
and a single coloring mode.

The BAM file is auto-discovered by searching for *.bam files whose name
contains the sample ID (e.g. --sample NA12878 will find
NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam).

Usage examples:

  # NA12878, all autosomes, standard coloring, default counts
  python3 scripts/generate_dataset.py --sample NA12878 --mode standard

  # HG00096, chromosomes 20 and 21 only, kmer coloring, custom counts
  python3 scripts/generate_dataset.py \
      --sample HG00096 \
      --mode kmer \
      --chroms 20,21 \
      --del-count 500 --up-count 250 --down-count 250

Output structure:
  data/{sample}/{mode}/deletion/
  data/{sample}/{mode}/non_deletion/
"""
import argparse
import glob
import sys
import logging
import random
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import VCFHandler, Variant, DeletionSize
from deepsv.visualization.image_generator import ImageGenerator
from deepsv.processing.refinement import BoundaryRefiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BAM auto-discovery
# ---------------------------------------------------------------------------

def find_bam_for_sample(sample: str, search_dir: str = "raw") -> str:
    """Find a BAM file whose name contains the sample ID.

    Searches *search_dir* for ``*.bam`` files that have *sample* as a
    substring in the filename.  Returns the path to the matching file.

    Raises:
        FileNotFoundError: No matching BAM found.
        RuntimeError: Multiple BAM files match (ambiguous).
    """
    matches = [
        p for p in Path(search_dir).glob("*.bam")
        if sample in p.name
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No BAM file containing '{sample}' found in {search_dir}. "
            "Provide the path explicitly with --bam."
        )
    if len(matches) > 1:
        names = ", ".join(m.name for m in matches)
        raise RuntimeError(
            f"Multiple BAM files match '{sample}': {names}. "
            "Provide the exact path with --bam."
        )

    found = str(matches[0])
    logger.info("Auto-detected BAM file: %s", found)
    return found


# ---------------------------------------------------------------------------
# Core image generation
# ---------------------------------------------------------------------------

def generate_images_for_regions(
    bam: BAMHandler,
    image_gen: ImageGenerator,
    regions: List[Variant],
    target_count: int,
    output_dir: Path,
    prefix: str,
    region_size: int = 50,
) -> int:
    """Generate up to *target_count* images from *regions*.

    Each variant/anchor region is windowed into ``region_size`` bp chunks.
    Returns the number of images actually generated.
    """
    generated = 0

    for region in regions:
        if generated >= target_count:
            break

        region_start = region.start
        while region_start < region.end and generated < target_count:
            region_end = min(region_start + region_size, region.end)

            try:
                pileup_data = bam.get_pileup_data(
                    region.chrom, region_start, region_end
                )
                clipping_data = bam.get_clipping_info(
                    region.chrom, region_start, region_end
                )

                if pileup_data:
                    image_name = (
                        f"{prefix}_{region.chrom}_{region_start}_{region_end}.png"
                    )

                    img = image_gen.generate_image(
                        pileup_data, clipping_data, region_start,
                        region_end - region_start,
                    )
                    image_gen.save_image(img, str(output_dir / image_name))
                    generated += 1
            except Exception as e:
                logger.error(
                    "Error processing region %s:%d-%d: %s",
                    region.chrom, region_start, region_end, e,
                )

            region_start += region_size

    return generated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a balanced image dataset for a given sample, "
            "chromosome set, and coloring mode."
        ),
    )
    parser.add_argument(
        "--sample",
        required=True,
        help="Sample ID (e.g. NA12878). Used for VCF filtering and output path.",
    )
    parser.add_argument(
        "--bam",
        default=None,
        help="Path to BAM file. If omitted, auto-detected by searching for "
             "a *.bam file containing the sample name.",
    )
    parser.add_argument(
        "--vcf",
        default="raw/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz",
        help="Path to VCF file.",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "kmer"],
        required=True,
        help="Coloring mode for image generation.",
    )
    parser.add_argument(
        "--chroms",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22",
        help="Comma-separated list of chromosomes to process (default: 1-22).",
    )
    parser.add_argument(
        "--del-count",
        type=int,
        default=2000,
        help="Target deletion images per chromosome (default: 2000).",
    )
    parser.add_argument(
        "--up-count",
        type=int,
        default=1000,
        help="Target up-anchor non-deletion images per chromosome (default: 1000).",
    )
    parser.add_argument(
        "--down-count",
        type=int,
        default=1000,
        help="Target down-anchor non-deletion images per chromosome (default: 1000).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Root output directory. Default: data/{sample}",
    )
    parser.add_argument(
        "--region-size",
        type=int,
        default=50,
        help="Window size in bp for each image (default: 50).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=10000,
        help="Maximum variant length to process (default: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible region shuffling.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Resolve defaults ───────────────────────────────────────────────
    if args.bam:
        bam_path = args.bam
    else:
        bam_path = find_bam_for_sample(args.sample)
    root = Path(args.output) if args.output else Path("data") / args.sample

    chroms = [c.strip() for c in args.chroms.split(",")]

    # ── Create output directories ──────────────────────────────────────
    del_dir = root / args.mode / "deletion"
    non_del_dir = root / args.mode / "non_deletion"
    del_dir.mkdir(parents=True, exist_ok=True)
    non_del_dir.mkdir(parents=True, exist_ok=True)

    # ── Load variants ──────────────────────────────────────────────────
    vcf_handler = VCFHandler(args.vcf)
    logger.info("Loading variants for sample %s …", args.sample)
    all_variants = vcf_handler.load_variants(sample_id=args.sample)
    logger.info("Total variants loaded: %d", len(all_variants))

    image_gen = ImageGenerator(coloring_mode=args.mode)
    refiner = BoundaryRefiner()

    # ── Process each chromosome ────────────────────────────────────────
    grand_del, grand_up, grand_down = 0, 0, 0

    with BAMHandler(bam_path) as bam:
        for chrom in chroms:
            logger.info("===== Chromosome %s =====", chrom)

            # Gather & filter deletion variants for this chromosome
            chrom_variants = [v for v in all_variants if v.chrom == chrom]
            if args.max_length:
                chrom_variants = [
                    v for v in chrom_variants if v.length <= args.max_length
                ]
            logger.info(
                "  Deletion variants on chr%s (≤%d bp): %d",
                chrom, args.max_length, len(chrom_variants),
            )

            # Refine boundaries
            refined_variants: List[Variant] = []
            for v in chrom_variants:
                try:
                    refined_variants.append(refiner.refine_boundaries(bam, v))
                except Exception as e:
                    logger.warning(
                        "  Could not refine %s:%d-%d: %s",
                        v.chrom, v.start, v.end, e,
                    )
                    refined_variants.append(v)

            # Shuffle for diversity
            random.shuffle(refined_variants)

            # 1. Deletion images
            del_count = generate_images_for_regions(
                bam, image_gen, refined_variants,
                args.del_count, del_dir, prefix="del",
                region_size=args.region_size,
            )
            logger.info(
                "  Deletion images: %d / %d", del_count, args.del_count
            )

            # 2. Non-deletion up-anchor images
            up_anchors = vcf_handler.get_non_deletion_regions(
                refined_variants, anchor_type="up"
            )
            random.shuffle(up_anchors)

            up_count = generate_images_for_regions(
                bam, image_gen, up_anchors,
                args.up_count, non_del_dir, prefix="non_del_up",
                region_size=args.region_size,
            )
            logger.info(
                "  Up-anchor images: %d / %d", up_count, args.up_count
            )

            # 3. Non-deletion down-anchor images
            down_anchors = vcf_handler.get_non_deletion_regions(
                refined_variants, anchor_type="down"
            )
            random.shuffle(down_anchors)

            down_count = generate_images_for_regions(
                bam, image_gen, down_anchors,
                args.down_count, non_del_dir, prefix="non_del_down",
                region_size=args.region_size,
            )
            logger.info(
                "  Down-anchor images: %d / %d", down_count, args.down_count
            )

            # Chromosome summary
            logger.info(
                "  Chr %s — del: %d, non-del: %d (up: %d, down: %d)",
                chrom, del_count, up_count + down_count, up_count, down_count,
            )

            grand_del += del_count
            grand_up += up_count
            grand_down += down_count

    logger.info("=" * 50)
    logger.info(
        "TOTAL — del: %d, non-del: %d (up: %d, down: %d)",
        grand_del, grand_up + grand_down, grand_up, grand_down,
    )
    logger.info("Output: %s", root / args.mode)


if __name__ == "__main__":
    main()
