#!/usr/bin/env python3
"""Call deletions from whole genome images using trained model"""
import argparse
import sys
from pathlib import Path
import logging
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.models.cnn import ModernDeletionCNN
from deepsv.inference.predictor import DeletionPredictor
from deepsv.utils.vcf_writer import VCFWriter, DeletionCall

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_image_filename(filename: Path) -> tuple:
    """Parse chromosome and position from image filename"""
    # Expected format: del_chr1_12345_12450.png
    parts = filename.stem.split('_')
    if len(parts) >= 4:
        chrom = parts[1]
        start = int(parts[2])
        end = int(parts[3])
        return chrom, start, end
    return None, None, None


def main():
    parser = argparse.ArgumentParser(description="Call deletions from images")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--images", required=True, help="Directory with genome images")
    parser.add_argument("--output", required=True, help="Output VCF file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--sample", default="SAMPLE", help="Sample name")
    
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = ModernDeletionCNN(num_classes=2)
    model.load_state_dict(torch.load(args.model, map_location='cpu', weights_only=True))
    
    # Initialize predictor
    predictor = DeletionPredictor(model, threshold=args.threshold)
    
    # Process images
    image_dir = Path(args.images)
    image_files = list(image_dir.glob("*.png"))
    
    logger.info(f"Processing {len(image_files)} images...")
    
    vcf_writer = VCFWriter(sample=args.sample)
    
    for image_file in image_files:
        prob, pred = predictor.predict_image(image_file)
        
        if pred == 1:  # Deletion detected
            chrom, start, end = parse_image_filename(image_file)
            if chrom and start and end:
                call = DeletionCall(
                    chrom=chrom,
                    start=start,
                    end=end,
                    quality=prob,
                    filter_status="PASS" if prob > 0.7 else "LowQual"
                )
                vcf_writer.add_call(call)
                logger.info(f"Deletion detected: {chrom}:{start}-{end} (prob={prob:.3f})")
    
    # Write VCF
    output_path = Path(args.output)
    vcf_writer.write(output_path)
    logger.info(f"VCF file written to {output_path}")


if __name__ == "__main__":
    main()

