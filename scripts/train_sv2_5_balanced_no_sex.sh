#!/bin/bash
# Master pipeline for DeepSV 2.5 (Hybrid Mode)
# 1. Excludes sex chromosomes (X, Y)
# 2. Balances classes (Deletion == Non-Deletion count)
# 3. Excludes VERY_LARGE deletions.

set -e

BAM="raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
VCF="raw/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"
FASTA="raw/hs37d5.fa"
SAMPLE="NA12878"
OUT_ROOT="data/hybrid_balanced_no_sex"
DEVICE="cuda"

echo "--- STEP 1: Generating Balanced Data (Small, Medium, Large) ---"
# Note: --exclude-sex filters X/Y, --balance ensures class equilibrium
python scripts/generate_image_tensor_dataset.py generate \
    --bam "$BAM" --vcf "$VCF" --fasta "$FASTA" --output "$OUT_ROOT" \
    --sample "$SAMPLE" --size "small,medium,large" --device "$DEVICE" \
    --exclude-sex --balance

echo "--- STEP 2: PCA Fitting ---"
python scripts/generate_image_tensor_dataset.py pca \
    --dataset "$OUT_ROOT" --n-components 8

echo "--- STEP 3: Model Training (Hybrid 11-channel) ---"
python scripts/train_image_tensor_model.py \
    --data-root "$OUT_ROOT" \
    --output "$OUT_ROOT/models" \
    --epochs 3 \
    --batch-size 32 \
    --device "$DEVICE"

echo "--- Pipeline Script Created! ---"
echo "To run this script, use: chmod +x scripts/train_sv2_5_balanced_no_sex.sh && ./scripts/train_sv2_5_balanced_no_sex.sh"
