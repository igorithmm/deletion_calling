#!/bin/bash
# Master pipeline for training DeepSV 2.5 (Hybrid Mode)
# Excludes VERY_LARGE deletions.

set -e

BAM="raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam"
VCF="raw/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"
FASTA="raw/hs37d5.fa"
SAMPLE="NA12878"
OUT_ROOT="data/hybrid_no_very_large"
DEVICE="cuda"

echo "--- STEP 1: Generating Data for SMALL category ---"
python scripts/generate_image_tensor_dataset.py generate \
    --bam "$BAM" --vcf "$VCF" --fasta "$FASTA" --output "$OUT_ROOT" \
    --sample "$SAMPLE" --size small --device "$DEVICE"

echo "--- STEP 2: Generating Data for MEDIUM category ---"
python scripts/generate_image_tensor_dataset.py generate \
    --bam "$BAM" --vcf "$VCF" --fasta "$FASTA" --output "$OUT_ROOT" \
    --sample "$SAMPLE" --size medium --device "$DEVICE"

echo "--- STEP 3: Generating Data for LARGE category ---"
python scripts/generate_image_tensor_dataset.py generate \
    --bam "$BAM" --vcf "$VCF" --fasta "$FASTA" --output "$OUT_ROOT" \
    --sample "$SAMPLE" --size large --device "$DEVICE"

echo "--- STEP 4: PCA Fitting ---"
python scripts/generate_image_tensor_dataset.py pca \
    --dataset "$OUT_ROOT" --n-components 8

echo "--- STEP 5: Model Training ---"
python scripts/train_image_tensor_model.py \
    --data-root "$OUT_ROOT" \
    --output "$OUT_ROOT/models" \
    --epochs 20 \
    --batch-size 32 \
    --device "$DEVICE"

echo "--- Pipeline Complete! Results in $OUT_ROOT/models ---"
