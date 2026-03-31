#!/bin/bash
# Master pipeline for DeepSV2.5 Hybrid Mode (RGB + DNABERT-2)
# Usage: ./scripts/run_deepsv2_5.sh <bam> <vcf> <fasta> <sample> <output_dir> [limit]

set -e # Exit on error

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <bam> <vcf> <fasta> <sample> <output_dir> [limit]"
    exit 1
fi

BAM=$1
VCF=$2
FASTA=$3
SAMPLE=$4
OUT_DIR=$5
LIMIT=${6:-5000}

DEVICE="cuda"
if ! command -v nvidia-smi &> /dev/null; then
    DEVICE="cpu"
    echo "nvidia-smi not found, using CPU."
fi

echo "--- STEP 1: GENERATE DATA ---"
python scripts/generate_image_tensor_dataset.py generate \
    --bam "$BAM" --vcf "$VCF" --fasta "$FASTA" --output "$OUT_DIR" \
    --sample "$SAMPLE" --device "$DEVICE" --limit "$LIMIT"

echo "--- STEP 2: FIT PCA (8 components) ---"
python scripts/generate_image_tensor_dataset.py pca \
    --dataset "$OUT_DIR" --n-components 8

echo "--- STEP 3: TRAIN MODEL ---"
python scripts/train_image_tensor_model.py \
    --data-root "$OUT_DIR" \
    --output "$OUT_DIR/models" \
    --epochs 20 \
    --batch-size 32 \
    --device "$DEVICE"

echo "Done! DeepSV2.5 pipeline finished. Check $OUT_DIR/models for results."
