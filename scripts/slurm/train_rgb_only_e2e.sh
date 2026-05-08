#!/bin/bash
#=============================================
# SLURM script for End-to-End RGB-only Training (Model M0)
#=============================================

#=============================================
# SLURM parameters
#=============================================
#SBATCH --job-name=cadc_rgb_e2e
#SBATCH --partition=gpu_T4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/scratch/igorno-my_work/logs/job_rgb_e2e_%j.out
#SBATCH --error=/scratch/igorno-my_work/logs/job_rgb_e2e_%j.err
#SBATCH --mail-user=igor.no.02@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

#=============================================
# Paths & Settings
#=============================================
DATA_DIR="/datasets/igorno-genomes_1000"
WORK_DIR="/scratch/igorno-my_work"
REPO_DIR=~/deletion_calling

BAM_DIR="${DATA_DIR}/bam/high_coverage"
VCF="${DATA_DIR}/vcf/ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"

TRAIN_CHROMS="1,2,3,4,5,6,7,8,9,10,11"
VAL_CHROMS="12,13,14,15,16,17,18,19,20,21,22"
ALL_CHROMS="${TRAIN_CHROMS},${VAL_CHROMS}"

OUTPUT_DIR="${WORK_DIR}/data/fused_${SLURM_JOB_ID}"
COMBINED_MANIFEST="${WORK_DIR}/data/manifest_${SLURM_JOB_ID}.csv"
OUTPUT_MODEL="${WORK_DIR}/models/model_${SLURM_JOB_ID}.pth"
LOG_FILE="${WORK_DIR}/logs/train_${SLURM_JOB_ID}.log"
DASHBOARD="${WORK_DIR}/logs/dashboard_${SLURM_JOB_ID}.png"

#=============================================
# Cache Setup
#=============================================
export HF_HOME="${WORK_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export MPLCONFIGDIR="${WORK_DIR}/.cache/matplotlib"
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$HF_DATASETS_CACHE" "$MPLCONFIGDIR"

#=============================================
# Conda Setup
#=============================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate deepsv2_new

mkdir -p "$OUTPUT_DIR" "${WORK_DIR}/models" "${WORK_DIR}/logs"
cd "$REPO_DIR"

echo "=== Starting RGB-only End-to-End Pipeline ==="
echo "BAM Dir:       $BAM_DIR"
echo "VCF:           $VCF"
echo "Train Chroms:  $TRAIN_CHROMS"
echo "Val Chroms:    $VAL_CHROMS"
echo "GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo

#=============================================
# Step 1: Generate Data for all BAMs
#=============================================
echo "=== Step 1: Generating datasets for all samples ==="

rm -f "$COMBINED_MANIFEST"

for BAM in "${BAM_DIR}"/*.bam; do
    # Extract sample name (e.g. NA12878 from NA12878.mapped.ILLUMINA.bwa.CEU.high_coverage_pcr_free.20130906.bam)
    SAMPLE=$(basename "$BAM" | cut -d'.' -f1)
    MANIFEST="${OUTPUT_DIR}/${SAMPLE}/manifest.csv"
    
    if [ -f "$MANIFEST" ]; then
        echo "  -> Manifest for $SAMPLE already exists. Skipping data generation."
    else
        echo "  -> Processing sample: $SAMPLE"
        python3 scripts/generate_fused_dataset.py \
            --sample "$SAMPLE" \
            --bam "$BAM" \
            --vcf "$VCF" \
            --chroms "$ALL_CHROMS" \
            --output-dir "${OUTPUT_DIR}" \
            --max-length 10000 \
            --min-length 50 \
            --del-count 1500
    fi
    
    # Merge into the combined manifest
    if [ ! -f "$COMBINED_MANIFEST" ]; then
        head -n 1 "$MANIFEST" > "$COMBINED_MANIFEST"
    fi
    tail -n +2 "$MANIFEST" >> "$COMBINED_MANIFEST"
done

echo "Finished generating datasets. Combined manifest saved to: $COMBINED_MANIFEST"
echo "Total rows in combined manifest: $(wc -l < "$COMBINED_MANIFEST")"
echo

#=============================================
# Step 2: Train Model M0 (RGB-only)
#=============================================
echo "=== Step 2: Training RGB-only Model (M0) ==="
echo "Logging output to: $LOG_FILE"

# Run training and tee output to the log file for parsing later
python3 scripts/train_fused_model.py \
    --manifest "$COMBINED_MANIFEST" \
    --model cnn \
    --output "$OUTPUT_MODEL" \
    --train-chroms "$TRAIN_CHROMS" \
    --val-chroms "$VAL_CHROMS" \
    --epochs 20 \
    --batch-size 128 \
    --lr-cnn 1e-4 | tee "$LOG_FILE"

echo
echo "Training complete. Best model saved to: $OUTPUT_MODEL"

#=============================================
# Step 3: Generate Dashboard
#=============================================
echo "=== Step 3: Generating Metrics Dashboard ==="

python3 - <<EOF
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

log_file = "${LOG_FILE}"
dashboard_out = "${DASHBOARD}"

train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_f1s, val_f1s = [], []
train_aucs, val_aucs = [], []

with open(log_file, "r") as f:
    for line in f:
        m_train = re.search(r"Train Loss: ([\d\.]+), Train Acc: ([\d\.]+)%, P: ([\d\.]+), R: ([\d\.]+), F1: ([\d\.]+), AUC: ([\d\.]+)", line)
        if m_train:
            train_losses.append(float(m_train.group(1)))
            train_accs.append(float(m_train.group(2)))
            train_f1s.append(float(m_train.group(5)))
            train_aucs.append(float(m_train.group(6)))
            
        m_val = re.search(r"Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)%, P: ([\d\.]+), R: ([\d\.]+), F1: ([\d\.]+), AUC: ([\d\.]+)", line)
        if m_val:
            val_losses.append(float(m_val.group(1)))
            val_accs.append(float(m_val.group(2)))
            val_f1s.append(float(m_val.group(5)))
            val_aucs.append(float(m_val.group(6)))

if not train_losses or not val_losses:
    print("Warning: No metrics found in log file to plot.")
else:
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cross-Entropy Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accs)+1), train_accs, label="Train Acc", marker='o')
    plt.plot(range(1, len(val_accs)+1), val_accs, label="Val Acc", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(train_f1s)+1), train_f1s, label="Train F1", marker='o')
    plt.plot(range(1, len(val_f1s)+1), val_f1s, label="Val F1", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score (Best Checkpoint Metric)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(train_aucs)+1), train_aucs, label="Train AUC", marker='o')
    plt.plot(range(1, len(val_aucs)+1), val_aucs, label="Val AUC", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(dashboard_out)
    print(f"Dashboard saved to {dashboard_out}")
EOF

echo "=== Pipeline Finished Successfully ==="
