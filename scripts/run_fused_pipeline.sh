#!/usr/bin/env bash
# End-to-end CADC pipeline driver.
#
# Stages
# ──────
#   0. Precompute HyenaDNA embeddings  (precompute_hyenadna_embeddings.py)
#      — only if the HDF5 doesn't already exist (idempotent)
#   1+2+3. Generate fused dataset       (generate_fused_dataset.py)
#      — Steps 1+2: feature extraction + 3-cluster K-means breakpoint refinement
#      — Step 3: render 50-bp pileup → 256x256 RGB images + manifest CSV
#   4a. Train one of M0 / M1            (train_fused_model.py)
#   4b. Run inference, emit predictions + VCF  (call_fused_deletions.py)
#
# Usage
# ─────
#   ./scripts/run_fused_pipeline.sh \
#       --sample NA12878 \
#       --bam   raw/NA12878.bam \
#       --vcf   raw/sv_truth.vcf.gz \
#       --fasta raw/hs37d5.fa \
#       --model fused \
#       --train-chroms 20,21 --val-chroms 22 \
#       --out-root runs/exp1
#
# Arguments map 1:1 onto the underlying scripts; everything else uses sane
# defaults. Set CADC_PYTHON to override the python interpreter.

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
SAMPLE=""
BAM=""
VCF=""
FASTA=""
MODEL="fused"                      # cnn | fused
FUSION_MODE="film_context"         # film | context | film_context
CHROMS="20,21,22"
TRAIN_CHROMS="20,21"
VAL_CHROMS="22"
MAX_LENGTH=10000
DEL_COUNT=""
EPOCHS=10
STAGE_A_EPOCHS=2
BATCH_SIZE=128
LR_CNN=1e-4
LR_FILM=1e-3
INIT_CNN_CHECKPOINT=""
CONTEXT_HIDDEN_DIM=128
CONTEXT_DROPOUT=0.1
THRESHOLD=0.5
OUT_ROOT="runs/cadc"
DEVICE="cuda"
EMBEDDINGS_H5=""                   # if empty, will be set under OUT_ROOT
SKIP_PRECOMPUTE=0
SKIP_GENERATE=0
SKIP_TRAIN=0
SKIP_INFER=0

PY="${CADC_PYTHON:-python3}"

# ── Arg parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample)         SAMPLE="$2"; shift 2;;
        --bam)            BAM="$2"; shift 2;;
        --vcf)            VCF="$2"; shift 2;;
        --fasta)          FASTA="$2"; shift 2;;
        --model)          MODEL="$2"; shift 2;;
        --fusion-mode)    FUSION_MODE="$2"; shift 2;;
        --chroms)         CHROMS="$2"; shift 2;;
        --train-chroms)   TRAIN_CHROMS="$2"; shift 2;;
        --val-chroms)     VAL_CHROMS="$2"; shift 2;;
        --max-length)     MAX_LENGTH="$2"; shift 2;;
        --del-count)      DEL_COUNT="$2"; shift 2;;
        --epochs)         EPOCHS="$2"; shift 2;;
        --stage-a-epochs) STAGE_A_EPOCHS="$2"; shift 2;;
        --batch-size)     BATCH_SIZE="$2"; shift 2;;
        --lr-cnn)         LR_CNN="$2"; shift 2;;
        --lr-film)        LR_FILM="$2"; shift 2;;
        --init-cnn-checkpoint) INIT_CNN_CHECKPOINT="$2"; shift 2;;
        --context-hidden-dim) CONTEXT_HIDDEN_DIM="$2"; shift 2;;
        --context-dropout) CONTEXT_DROPOUT="$2"; shift 2;;
        --threshold)      THRESHOLD="$2"; shift 2;;
        --out-root)       OUT_ROOT="$2"; shift 2;;
        --device)         DEVICE="$2"; shift 2;;
        --embeddings)     EMBEDDINGS_H5="$2"; shift 2;;
        --skip-precompute) SKIP_PRECOMPUTE=1; shift;;
        --skip-generate)   SKIP_GENERATE=1; shift;;
        --skip-train)      SKIP_TRAIN=1; shift;;
        --skip-infer)      SKIP_INFER=1; shift;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0;;
        *)
            echo "Unknown argument: $1" >&2; exit 1;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────
[[ -z "$SAMPLE" ]] && { echo "ERROR: --sample is required" >&2; exit 1; }
[[ -z "$BAM" ]]    && { echo "ERROR: --bam is required"    >&2; exit 1; }
[[ -z "$VCF" ]]    && { echo "ERROR: --vcf is required"    >&2; exit 1; }
if [[ "$MODEL" != "cnn" && -z "$FASTA" && -z "$EMBEDDINGS_H5" ]]; then
    echo "ERROR: --fasta or --embeddings is required when --model=$MODEL" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"
DATA_DIR="$OUT_ROOT/data"
MODELS_DIR="$OUT_ROOT/models"
PRED_DIR="$OUT_ROOT/predictions"
mkdir -p "$DATA_DIR" "$MODELS_DIR" "$PRED_DIR"

if [[ -z "$EMBEDDINGS_H5" ]]; then
    EMBEDDINGS_H5="$OUT_ROOT/hyenadna_embeddings.h5"
fi

CHECKPOINT="$MODELS_DIR/${SAMPLE}_${MODEL}_best.pth"
PRED_CSV="$PRED_DIR/${SAMPLE}_${MODEL}_predictions.csv"
VCF_OUT="$PRED_DIR/${SAMPLE}_${MODEL}.vcf"
MANIFEST="$DATA_DIR/$SAMPLE/manifest.csv"

echo "════════════════════════════════════════════════════════════════════"
echo " CADC pipeline"
echo "   sample      : $SAMPLE"
echo "   model       : $MODEL"
echo "   fusion_mode : $FUSION_MODE"
echo "   bam         : $BAM"
echo "   vcf         : $VCF"
echo "   fasta       : ${FASTA:-<n/a>}"
echo "   chroms      : $CHROMS"
echo "   train/val   : $TRAIN_CHROMS  /  $VAL_CHROMS"
echo "   embeddings  : $EMBEDDINGS_H5"
echo "   checkpoint  : $CHECKPOINT"
echo "   out_root    : $OUT_ROOT"
echo "════════════════════════════════════════════════════════════════════"

# ── Stage 0: precompute HyenaDNA embeddings ───────────────────────────────
if [[ "$MODEL" != "cnn" && "$SKIP_PRECOMPUTE" -eq 0 ]]; then
    if [[ -f "$EMBEDDINGS_H5" ]]; then
        echo "[0/4] Embeddings exist at $EMBEDDINGS_H5 — skipping precompute"
    else
        [[ -z "$FASTA" ]] && { echo "ERROR: --fasta required to precompute" >&2; exit 1; }
        echo "[0/4] Precomputing HyenaDNA embeddings → $EMBEDDINGS_H5"
        IFS=',' read -r -a chrom_array <<< "$CHROMS"
        "$PY" scripts/precompute_hyenadna_embeddings.py \
            --fasta "$FASTA" \
            --output "$EMBEDDINGS_H5" \
            --device "$DEVICE" \
            --chrom "${chrom_array[@]}"
    fi
fi

# ── Stage 1+2+3: generate fused dataset ───────────────────────────────────
if [[ "$SKIP_GENERATE" -eq 0 ]]; then
    echo "[1+2+3/4] Generating images + manifest → $MANIFEST"
    GENERATE_ARGS=(
        --sample "$SAMPLE"
        --bam "$BAM"
        --vcf "$VCF"
        --chroms "$CHROMS"
        --output-dir "$DATA_DIR"
        --max-length "$MAX_LENGTH"
    )
    [[ -n "$DEL_COUNT" ]] && GENERATE_ARGS+=(--del-count "$DEL_COUNT")
    "$PY" scripts/generate_fused_dataset.py "${GENERATE_ARGS[@]}"
else
    echo "[1+2+3/4] --skip-generate set — using existing $MANIFEST"
fi

# ── Stage 4a: train ────────────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    echo "[4a/4] Training $MODEL → $CHECKPOINT"
    TRAIN_ARGS=(
        --manifest "$MANIFEST"
        --embeddings "$EMBEDDINGS_H5"
        --model "$MODEL"
        --output "$CHECKPOINT"
        --train-chroms "$TRAIN_CHROMS"
        --val-chroms "$VAL_CHROMS"
        --epochs "$EPOCHS"
        --stage-a-epochs "$STAGE_A_EPOCHS"
        --batch-size "$BATCH_SIZE"
        --lr-cnn "$LR_CNN"
        --lr-film "$LR_FILM"
    )
    if [[ "$MODEL" != "cnn" ]]; then
        TRAIN_ARGS+=(
            --fusion-mode "$FUSION_MODE"
            --context-hidden-dim "$CONTEXT_HIDDEN_DIM"
            --context-dropout "$CONTEXT_DROPOUT"
        )
        [[ -n "$INIT_CNN_CHECKPOINT" ]] && TRAIN_ARGS+=(--init-cnn-checkpoint "$INIT_CNN_CHECKPOINT")
    fi
    "$PY" scripts/train_fused_model.py "${TRAIN_ARGS[@]}"
else
    echo "[4a/4] --skip-train set — using existing $CHECKPOINT"
fi

# ── Stage 4b: inference ────────────────────────────────────────────────────
if [[ "$SKIP_INFER" -eq 0 ]]; then
    echo "[4b/4] Inference → $PRED_CSV  +  $VCF_OUT"
    INFER_ARGS=(
        --manifest "$MANIFEST"
        --checkpoint "$CHECKPOINT"
        --model "$MODEL"
        --predictions-out "$PRED_CSV"
        --vcf-out "$VCF_OUT"
        --threshold "$THRESHOLD"
        --sample-id "$SAMPLE"
    )
    if [[ "$MODEL" != "cnn" ]]; then
        INFER_ARGS+=(
            --embeddings "$EMBEDDINGS_H5"
            --fusion-mode "$FUSION_MODE"
            --context-hidden-dim "$CONTEXT_HIDDEN_DIM"
            --context-dropout "$CONTEXT_DROPOUT"
        )
    fi
    "$PY" scripts/call_fused_deletions.py "${INFER_ARGS[@]}"
fi

echo "════════════════════════════════════════════════════════════════════"
echo " Pipeline complete."
echo "   manifest    : $MANIFEST"
echo "   checkpoint  : $CHECKPOINT"
echo "   predictions : $PRED_CSV"
echo "   vcf         : $VCF_OUT"
echo "════════════════════════════════════════════════════════════════════"
