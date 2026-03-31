#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== 0. Cleaning old data ==="
rm -rf data/tensor_test

echo "=== 1. Generating Training Data (chr1) ==="
python scripts/generate_tensor_dataset.py generate \
  --bam raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam \
  --vcf raw/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz \
  --fasta raw/hs37d5.fa \
  --sample NA12878 \
  --chrom 1 \
  --limit 200 \
  --device cuda \
  --output data/tensor_test

echo "=== 2. Generating Validation Data (chr2) ==="
python scripts/generate_tensor_dataset.py generate \
  --bam raw/NA12878.mapped.ILLUMINA.bwa.CEU.low_coverage.20121211.bam \
  --vcf raw/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz \
  --fasta raw/hs37d5.fa \
  --sample NA12878 \
  --chrom 2 \
  --limit 50 \
  --device cuda \
  --output data/tensor_test

echo "=== 3. Fitting PCA ==="
python scripts/generate_tensor_dataset.py pca \
  --dataset data/tensor_test \
  --n-components 8

echo "=== 4. Training Model ==="
python scripts/train_tensor_model.py \
  --data-root data/tensor_test \
  --output models/tensor_test \
  --context-channels 8 \
  --epochs 2 \
  --batch-size 8 \
  --train-chroms 1 \
  --val-chroms 2

echo "=== Pipeline Test Completed ==="
