#!/bin/bash
#=============================================
# A/B comparison for NA12878 (high_coverage):
#   Run A: 3-channel image-only (DeepSV baseline)
#   Run B: 11-channel image + DNABERT-2 PCA-8 context
# Both runs use the same BroadcastContextCNN, same hyperparameters,
# same train/val split, same seed. Only context_channels differs.
#=============================================

#=============================================
# SLURM parameters
#=============================================
#SBATCH --job-name=deepsv_AB_NA12878
#SBATCH --partition=gpu_T4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/igorno-my_work/logs/job_AB_%j.out
#SBATCH --error=/scratch/igorno-my_work/logs/job_AB_%j.err
#SBATCH --mail-user=igor.no.02@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

#=============================================
# Paths
#=============================================
DATA_DIR=/datasets/igorno-genomes_1000
WORK_DIR=/scratch/igorno-my_work
REPO_DIR=~/deletion_calling

BAM_PATH="${DATA_DIR}/bam/high_coverage/NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam"
VCF="${DATA_DIR}/vcf/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"
FASTA="${DATA_DIR}/fasta/hs37d5.fa"
SAMPLE="NA12878"

# Single shared dataset for both runs (generated WITH DNABERT-2, run A
# simply ignores the context channels).
DATASET_DIR="${WORK_DIR}/data/AB_${SAMPLE}_high_cov"
RUN_A_DIR="${WORK_DIR}/models/AB_${SAMPLE}/run_A_image_only_3ch"
RUN_B_DIR="${WORK_DIR}/models/AB_${SAMPLE}/run_B_image_dnabert_11ch"
COMPARE_DIR="${WORK_DIR}/reports/AB_${SAMPLE}"

# Hyperparameters — must be identical between the two runs.
EPOCHS=50
BATCH_SIZE=32
LR=3e-4
SEED=42
N_PCA_COMPONENTS=8

#=============================================
# HuggingFace cache redirect
# $HOME (/beegfs/...) is read-only on this cluster, but transformers with
# trust_remote_code=True needs to write extracted .py modules somewhere.
# Point all HF cache locations into the writable scratch tree.
#=============================================
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_MODULES_CACHE" "$HF_DATASETS_CACHE"

#=============================================
# Conda
#=============================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepsv2_new

#=============================================
# Sanity checks
#=============================================
for f in "$BAM_PATH" "$VCF" "$FASTA"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: file not found: $f"
        exit 1
    fi
done

mkdir -p "$DATASET_DIR" "$RUN_A_DIR" "$RUN_B_DIR" "$COMPARE_DIR" "${WORK_DIR}/logs"
cd "$REPO_DIR"

echo "=== A/B comparison pipeline — NA12878 (high_coverage) ==="
echo "BAM:     $BAM_PATH"
echo "VCF:     $VCF"
echo "FASTA:   $FASTA"
echo "Dataset: $DATASET_DIR"
echo "Run A:   $RUN_A_DIR"
echo "Run B:   $RUN_B_DIR"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

#=============================================
# STAGE 1: dataset generation (image + DNABERT-2 raw embedding)
# Skipped if already complete (idempotent on re-runs).
#=============================================
STAGE1_DONE_MARKER="${DATASET_DIR}/.stage1_done"
if [ -f "$STAGE1_DONE_MARKER" ]; then
    echo "=== Stage 1 skipped (marker exists at $STAGE1_DONE_MARKER) ==="
else
    echo "=== Stage 1: generating image tensors + DNABERT-2 raw embeddings ==="
    python3 scripts/generate_image_tensor_dataset.py generate \
        --bam "$BAM_PATH" \
        --vcf "$VCF" \
        --fasta "$FASTA" \
        --output "$DATASET_DIR" \
        --sample "$SAMPLE" \
        --device cuda \
        --exclude-sex \
        --balance \
        --size all \
        --coloring-mode standard \
        --seed "$SEED" \
        --dnabert-source hf
    touch "$STAGE1_DONE_MARKER"
    echo "=== Stage 1 done ==="
fi
echo

#=============================================
# STAGE 2: PCA + z-score (creates 'context' field used by run B).
#=============================================
STAGE2_DONE_MARKER="${DATASET_DIR}/.stage2_done"
if [ -f "$STAGE2_DONE_MARKER" ]; then
    echo "=== Stage 2 skipped (marker exists at $STAGE2_DONE_MARKER) ==="
else
    echo "=== Stage 2: PCA(${N_PCA_COMPONENTS}) + z-score on training chromosomes ==="
    python3 scripts/generate_image_tensor_dataset.py pca \
        --dataset "$DATASET_DIR" \
        --n-components "$N_PCA_COMPONENTS"
    touch "$STAGE2_DONE_MARKER"
    echo "=== Stage 2 done ==="
fi
echo

#=============================================
# STAGE 3a: Run A — 3-channel image-only baseline.
#=============================================
echo "=== Stage 3a: training Run A (image-only, 3ch baseline) ==="
python3 scripts/train_image_tensor_model.py \
    --data-root "$DATASET_DIR" \
    --output "$RUN_A_DIR" \
    --context-channels 0 \
    --run-name "image_only_3ch" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --device cuda
echo "=== Run A done ==="
echo

#=============================================
# STAGE 3b: Run B — 11-channel image + DNABERT-2.
#=============================================
echo "=== Stage 3b: training Run B (image + DNABERT-2, 11ch) ==="
python3 scripts/train_image_tensor_model.py \
    --data-root "$DATASET_DIR" \
    --output "$RUN_B_DIR" \
    --context-channels "$N_PCA_COMPONENTS" \
    --run-name "image_dnabert_11ch" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --device cuda
echo "=== Run B done ==="
echo

#=============================================
# STAGE 4: side-by-side comparison.
#=============================================
echo "=== Stage 4: comparing runs ==="
python3 scripts/compare_runs.py \
    --run-a "$RUN_A_DIR" \
    --run-b "$RUN_B_DIR" \
    --output "$COMPARE_DIR"

echo
echo "=== A/B pipeline complete ==="
echo "Comparison written to: $COMPARE_DIR"
echo "  - comparison.png  (overlaid validation curves)"
echo "  - comparison.txt  (best-epoch deltas)"
