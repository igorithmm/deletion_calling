#!/usr/bin/env python3
"""Generate training images from BAM and VCF files"""
import argparse
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import VCFHandler, DeletionSize
from deepsv.visualization.image_generator import ImageGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_deletion_images(bam_path: str,
                            vcf_path: str,
                            output_dir: str,
                            deletion_size: str,
                            region_size: int = 50,
                            target_chrom: str = None,
                            coloring_mode: str = "standard",
                            limit: int = None,
                            sample: str = None,
                            max_length: int = None):
    """
    Generate images for deletion regions
    
    Args:
        bam_path: Path to BAM file
        vcf_path: Path to VCF file
        output_dir: Output directory for images
        deletion_size: Size category ('small', 'medium', 'large', 'very_large')
        region_size: Size of each image region in bp
        target_chrom: Optional chromosome to filter by
        coloring_mode: 'standard' or 'kmer'
    """
    from deepsv.processing.refinement import BoundaryRefiner
    refiner = BoundaryRefiner()

    # Map size string to enum
    size_map = {
        'small': DeletionSize.SMALL,
        'medium': DeletionSize.MEDIUM,
        'large': DeletionSize.LARGE,
        'very_large': DeletionSize.VERY_LARGE
    }
    
    # Load VCF variants
    vcf_handler = VCFHandler(vcf_path)
    variants = vcf_handler.load_variants(sample_id=sample)

    if deletion_size == 'all':
        target_variants = []
        for cat in DeletionSize:
            target_variants.extend(vcf_handler.get_variants_by_size(cat))
        logger.info(f"Using all size categories. Total variants: {len(target_variants)}")
    elif deletion_size in size_map:
        size_category = size_map[deletion_size]
        target_variants = vcf_handler.get_variants_by_size(size_category)
        logger.info(f"Found {len(target_variants)} variants in category {deletion_size}")
    else:
        raise ValueError(f"Invalid deletion size: {deletion_size}")
    
    if target_chrom:
        target_variants = [v for v in target_variants if v.chrom == target_chrom]
        logger.info(f"Filtered to {len(target_variants)} variants on chromosome {target_chrom}")
        
    # Filter by max length
    if max_length:
        original_count = len(target_variants)
        target_variants = [v for v in target_variants if v.length <= max_length]
        logger.info(f"Filtered {original_count - len(target_variants)} variants longer than {max_length}bp. Remaining: {len(target_variants)}")
    
    # Initialize handlers
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_gen = ImageGenerator(coloring_mode=coloring_mode)
    
    with BAMHandler(bam_path) as bam:
        for idx, variant in enumerate(target_variants):
            if limit and idx >= limit:
                logger.info(f"Reached limit of {limit} variants. Stopping.")
                break

            logger.info(f"Processing variant {idx+1}/{len(target_variants)}: "
                       f"{variant.chrom}:{variant.start}-{variant.end}")
            
            # Refine deletion boundaries using K-Means
            variant = refiner.refine_boundaries(bam, variant)
            
            # Generate images for overlapping regions
            start = variant.start
            end = variant.end
            
            region_start = start
            region_idx = 0
            
            while region_start < end:
                region_end = min(region_start + region_size, end)
                
                # Get pileup and clipping data
                try:
                    pileup_data = bam.get_pileup_data(variant.chrom, region_start, region_end)
                    clipping_data = bam.get_clipping_info(variant.chrom, region_start, region_end)
                    
                    if pileup_data:
                        # Generate image
                        image = image_gen.generate_image(
                            pileup_data,
                            clipping_data,
                            region_start,
                            region_end - region_start
                        )
                        
                        # Save image
                        image_name = f"del_{variant.chrom}_{region_start}_{region_end}.png"
                        image_path = output_path / image_name
                        image_gen.save_image(image, str(image_path))
                except Exception as e:
                    logger.error(f"Error processing region {variant.chrom}:{region_start}-{region_end}: {e}")
                
                region_start += region_size
                region_idx += 1


def generate_non_deletion_images(bam_path: str,
                                vcf_path: str,
                                output_dir: str,
                                deletion_size: str,
                                anchor_type: str = "up",
                                region_size: int = 50,
                                target_chrom: str = None,
                                coloring_mode: str = "standard",
                                limit: int = None,
                                sample: str = None,
                                max_length: int = None):
    """
    Generate images for non-deletion anchor regions
    
    Args:
        bam_path: Path to BAM file
        vcf_path: Path to VCF file
        output_dir: Output directory for images
        deletion_size: Size category
        anchor_type: 'up' or 'down' anchor
        target_chrom: Optional chromosome to filter by
        coloring_mode: 'standard' or 'kmer'
        limit: Limit number of variants to process
        sample: Sample ID to filter
        max_length: Maximum variant length (used to filter source variants before generating anchors)
    """
    size_map = {
        'small': DeletionSize.SMALL,
        'medium': DeletionSize.MEDIUM,
        'large': DeletionSize.LARGE,
        'very_large': DeletionSize.VERY_LARGE
    }
    
    # Load VCF variants
    vcf_handler = VCFHandler(vcf_path)
    variants = vcf_handler.load_variants(sample_id=sample)

    if deletion_size == 'all':
        target_variants = []
        for cat in DeletionSize:
            target_variants.extend(vcf_handler.get_variants_by_size(cat))
        logger.info(f"Using all size categories for anchors. Total variants: {len(target_variants)}")
    elif deletion_size in size_map:
        size_category = size_map[deletion_size]
        target_variants = vcf_handler.get_variants_by_size(size_category)
    else:
        raise ValueError(f"Invalid deletion size: {deletion_size}")
    
    if target_chrom:
        target_variants = [v for v in target_variants if v.chrom == target_chrom]
    
    # Filter source variants by max length
    if max_length:
        original_count = len(target_variants)
        target_variants = [v for v in target_variants if v.length <= max_length]
        logger.info(f"Filtered {original_count - len(target_variants)} variants longer than {max_length}bp for anchors.")

    anchor_regions = vcf_handler.get_non_deletion_regions(target_variants, anchor_type)
    
    logger.info(f"Found {len(anchor_regions)} anchor regions")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_gen = ImageGenerator(coloring_mode=coloring_mode)
    
    with BAMHandler(bam_path) as bam:
        for idx, region in enumerate(anchor_regions):
            if limit and idx >= limit:
                logger.info(f"Reached limit of {limit} regions. Stopping.")
                break
                
            logger.info(f"Processing anchor {idx+1}/{len(anchor_regions)}: "
                       f"{region.chrom}:{region.start}-{region.end} (len: {region.length})")
            
            # Window the anchor region into region_size (50bp) chunks, 
            # matching the deletion image generation logic
            region_start = region.start
            while region_start < region.end:
                region_end = min(region_start + region_size, region.end)
                
                try:
                    pileup_data = bam.get_pileup_data(region.chrom, region_start, region_end)
                    clipping_data = bam.get_clipping_info(region.chrom, region_start, region_end)
                    
                    if pileup_data:
                        image = image_gen.generate_image(
                            pileup_data,
                            clipping_data,
                            region_start,
                            region_end - region_start
                        )
                        
                        image_name = f"non_del_{anchor_type}_{region.chrom}_{region_start}_{region_end}.png"
                        image_path = output_path / image_name
                        image_gen.save_image(image, str(image_path))
                except Exception as e:
                    logger.error(f"Error processing region {region.chrom}:{region_start}-{region_end}: {e}")
                
                region_start += region_size


def main():
    parser = argparse.ArgumentParser(description="Generate training images from BAM/VCF")
    parser.add_argument("--bam", required=True, help="Path to BAM file")
    parser.add_argument("--vcf", required=True, help="Path to VCF file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--type", choices=["deletion", "non_deletion"], required=True,
                       help="Type of images to generate")
    parser.add_argument("--size", choices=["small", "medium", "large", "very_large", "all"],
                       required=True, help="Deletion size category or 'all' for all sizes")
    parser.add_argument("--anchor", choices=["up", "down"], default="up",
                       help="Anchor type for non-deletion images")
    parser.add_argument("--chrom", help="Filter by chromosome (e.g. '20')")
    parser.add_argument("--coloring-mode", choices=["standard", "kmer"], default="standard",
                       help="Coloring mode: 'standard' (default) or 'kmer'")
    parser.add_argument("--limit", type=int, help="Limit number of variants/regions to process")
    parser.add_argument("--sample", help="Sample ID to filter variants (required for multi-sample VCF)")
    parser.add_argument("--max_length", type=int, default=10000, 
                       help="Maximum variant length to process (default: 10000)")
    
    args = parser.parse_args()
    
    if args.type == "deletion":
        generate_deletion_images(args.bam, args.vcf, args.output, args.size, target_chrom=args.chrom, 
                               coloring_mode=args.coloring_mode, limit=args.limit, sample=args.sample, 
                               max_length=args.max_length)
    else:
        generate_non_deletion_images(args.bam, args.vcf, args.output, args.size, anchor_type=args.anchor,
                                   target_chrom=args.chrom, coloring_mode=args.coloring_mode, limit=args.limit, 
                                   sample=args.sample, max_length=args.max_length)

if __name__ == "__main__":
    main()

