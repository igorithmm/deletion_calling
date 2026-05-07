#!/bin/bash
#=============================================
# SLURM script for precomputing HyenaDNA embeddings
#=============================================

#=============================================
# SLURM parameters
#=============================================
#SBATCH --job-name=precompute_hyenadna
#SBATCH --partition=gpu_T4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/igorno-my_work/logs/job_hyenadna_%j.out
#SBATCH --error=/scratch/igorno-my_work/logs/job_hyenadna_%j.err
#SBATCH --mail-user=igor.no.02@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

#=============================================
# Paths
#=============================================
DATA_DIR=/datasets/igorno-genomes_1000
WORK_DIR=/scratch/igorno-my_work
REPO_DIR=~/deletion_calling

FASTA="${DATA_DIR}/fasta/hs37d5.fa"
MODEL_PATH="${DATA_DIR}/weights/hyenadna-small-32k-seqlen-hf"
OUTPUT_H5="${WORK_DIR}/data/hyenadna_embeddings_hs37d5.h5"

#=============================================
# HuggingFace cache redirect
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
if [ ! -f "$FASTA" ]; then
    echo "ERROR: FASTA file not found: $FASTA"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model directory not found: $MODEL_PATH"
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT_H5")" "${WORK_DIR}/logs"
cd "$REPO_DIR"

echo "=== Precomputing HyenaDNA embeddings ==="
echo "FASTA:       $FASTA"
echo "Model Path:  $MODEL_PATH"
echo "Output H5:   $OUTPUT_H5"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

#=============================================
# Run precomputation
#=============================================
# Remove --chrom to process all autosomes, or specify them like: --chrom 1 2 3
python3 scripts/precompute_hyenadna_embeddings.py \
    --fasta "$FASTA" \
    --output "$OUTPUT_H5" \
    --model-id "$MODEL_PATH" \
    --device cuda \
    --genome-build hs37d5 \
    --resume

echo
echo "=== Precomputation complete ==="
echo "Embeddings written to: $OUTPUT_H5"
