# CADC — Context-Aware Deletion Caller

**CADC** is a deep-learning pipeline for detecting **genomic deletions** from whole-genome sequencing (WGS) data. It converts BAM pileups into RGB images and classifies each 50 bp window as *deletion* or *non-deletion* using a CNN backbone optionally conditioned on HyenaDNA sequence embeddings via **FiLM modulation**.

Two operating modes are supported:

| Mode | Model | Input | Use case |
|------|-------|-------|----------|
| **M0** (RGB-only) | `ModernDeletionCNN` | Pileup image | Fast, no embeddings needed |
| **M1** (Fused) | `FusedDeepSV` | Pileup image + HyenaDNA embedding | Full context-aware calling |

---

## Table of Contents

1. [Algorithm Overview](#algorithm-overview)
2. [FiLM + HyenaDNA (M1 mode)](#film--hyenadna-m1-mode)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Data Requirements](#data-requirements)
6. [Pipeline Stages](#pipeline-stages)
7. [CLI Reference](#cli-reference)
8. [Python API](#python-api)
9. [Configuration](#configuration)
10. [Acknowledgements](#acknowledgements)

---

## Algorithm Overview

```
BAM + VCF
    │
    ▼
Step 1  Feature extraction      read depth + soft-clipping signals
Step 2  Breakpoint refinement   3-cluster K-means → exact deletion boundaries
Step 3  Image generation        50 bp sliding window → 256×256 RGB pileup PNG
    │
    ▼  (M1 only)
HyenaDNA embedding precompute   32 k bp context window → 256-dim vector per 50 bp tile
    │
    ▼
Step 4a Training                M0: ModelTrainer (accuracy)
                                M1: FiLMTrainer  (F1, two-stage)
    │
    ▼
Step 4b Inference               per-window probabilities → merged DEL calls → VCF
```

### Core idea

1. **Load known deletions** from a truth VCF (e.g. 1000 Genomes Phase 3 SV calls).
2. **Slide a 50 bp window** across each deletion, rendering a 256×256 RGB image from the BAM pileup. Each pixel column = one genomic position; each row = one overlapping read. Pixel colour encodes nucleotide, mapping quality, pair status, CIGAR operations, and soft-clipping.
3. **Generate matching non-deletion windows** (upstream / downstream anchor regions of the same length) as the negative class.
4. **Train a CNN** to classify each image as `deletion` (label 1) or `non-deletion` (label 0).
5. **At inference** the window slides across the target genome, classifies each window, and adjacent positive predictions are merged into deletion calls written to VCF.

---

## FiLM + HyenaDNA (M1 mode)

M1 adds a **second modality**: frozen sequence embeddings from **HyenaDNA-small-32k** fused with CNN features through **FiLM conditioning** (Feature-wise Linear Modulation, Perez et al. 2018).

### Why it helps

The baseline CNN (M0) sees only the pileup image. The same genomic position carries its own *sequence context* (motifs, repeats, GC content) that partly explains how likely a deletion signal is. The HyenaDNA embedding provides a compressed 256-dimensional description of this context, and FiLM lets it **modulate CNN feature maps on the fly** without changing the underlying architecture.

### FiLM equation

For a feature map `F` of shape `(B, C, H, W)`:

```
γ, β = FiLMGenerator(embedding)          # each (B, C)
F_out = F × (1 + γ.view(B,C,1,1)) + β.view(B,C,1,1)
```

Each `FiLMGenerator` is a two-layer MLP `Linear(256→128) → ReLU → Linear(128→2·C)`. **The final Linear is zero-initialised** (weights and biases), so at step 0 γ = β = 0 and FiLM is an **exact identity**: `F_out = F`. Training starts from baseline M0 behaviour and deviates only as the FiLM heads learn.

### Injection points (ModernDeletionCNN)

| Point | Channels | Modulated |
|-------|----------|-----------|
| After block 1 (pool1) | 32 | ✗ too low-level |
| After block 2 (pool2) | 64 | ✗ too low-level |
| **After block 3 (pool3)** | **128** | **✓ hook3** |
| **After block 4 (pool4)** | **256** | **✓ hook4** |
| After block 5 (refinement) | 256 | ✗ spatial info already aggregated |

### Two-stage training

Implemented in `deepsv/training/film_trainer.py`:

- **Stage A** (default 2 epochs): backbone frozen, FiLM generators only. Single param group at `lr_film = 1e-3`.
- **Stage B** (remaining epochs): backbone unfrozen, two param groups — CNN @ `lr_cnn = 1e-4`, FiLM @ `lr_film = 1e-3`.

Best checkpoint is selected by **validation F1** (positive class), not accuracy. Loss = `nn.CrossEntropyLoss` on the 2-logit softmax head.

### Embedding precompute

HyenaDNA is **never loaded during training**. Run once upfront:

```bash
python scripts/precompute_hyenadna_embeddings.py \
    --fasta  raw/hs37d5.fa \
    --output data/hyenadna_embeddings.h5 \
    --device cuda \
    --chrom  20 21 22 \
    --genome-build hs37d5
```

HDF5 layout:

```
/chr1   shape=(n_windows, 256)  dtype=float16   # n_windows = ceil(chrom_len / 50)
/chr2   ...
attrs:
  window_bp  = 50
  embed_dim  = 256
  model_id   = LongSafari/hyenadna-small-32k-seqlen-hf
  ...
```

Lookup: `emb = f["chr21"][position // 50]`. For GPU-constrained nodes, specified chromosomes are preloaded into RAM as a dict at `FusedDataset` init to avoid per-batch HDF5 I/O. float16 → float32 cast happens before the model forward pass.

### Stratified validation

`FiLMTrainer.validate(...)` reports per-bucket precision / recall / F1 across:

- **Deletion length**: `50–200`, `200–500`, `500–1k`, `1k–5k`, `5k–10k` bp.

---

## Project Structure

```
cadc/
├── deepsv/                          # Core library (package name kept for compatibility)
│   ├── data/
│   │   ├── bam_handler.py           # BAM: pileup, coverage depth, soft-clipping
│   │   ├── vcf_handler.py           # VCF: variant loading, non-deletion anchor generation
│   │   └── fused_dataset.py         # FusedDataset: (image, HyenaDNA emb, label) triples
│   ├── visualization/
│   │   └── image_generator.py       # Pileup → 256×256 RGB image
│   ├── models/
│   │   ├── cnn.py                   # ModernDeletionCNN (M0 backbone)
│   │   ├── film.py                  # FiLMGenerator + apply_film
│   │   └── fused_cnn.py             # FusedDeepSV (M1): CNN + FiLM injection
│   ├── training/
│   │   ├── trainer.py               # ModelTrainer: M0 accuracy-based training
│   │   └── film_trainer.py          # FiLMTrainer: two-stage, F1-best, stratified eval
│   ├── inference/
│   │   ├── predictor.py             # DeletionPredictor: M0 image-only inference
│   │   └── fused_predictor.py       # FusedPredictor: M1 image + embedding lookup
│   ├── processing/
│   │   └── refinement.py            # BoundaryRefiner: K-means breakpoint refinement
│   └── utils/
│       └── kmeans.py                # Custom K-means implementation
│
├── scripts/
│   ├── generate_fused_dataset.py    # Steps 1–3: BAM+VCF → images + manifest CSV
│   ├── precompute_hyenadna_embeddings.py  # HyenaDNA → HDF5 embedding store
│   ├── train_fused_model.py         # Step 4a: train M0 or M1
│   ├── call_fused_deletions.py      # Step 4b: inference → predictions CSV + VCF
│   ├── run_fused_pipeline.sh        # End-to-end driver (all 4 stages)
│   ├── test_raw_e2e.py              # End-to-end sanity test on real BAM/VCF
│   └── slurm/                       # SLURM job scripts for cluster runs
│
├── raw/                             # Input data: BAM, VCF, FASTA (user-provided)
├── data/                            # Generated: images, manifest CSV, HDF5 embeddings
├── models/                          # Saved model checkpoints
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
# Clone and install dependencies
git clone <repo-url> cadc
cd cadc
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | CNN training and inference |
| `pysam` | BAM/VCF reading (pileup, genotype filtering) |
| `pyfaidx` | FASTA chunking for embedding precompute |
| `numpy`, `scipy` | Numerical computing, rolling median |
| `scikit-learn` | Metrics (F1, AUC), class-weight computation |
| `Pillow` | PNG image generation |
| `tqdm` | Progress bars |
| `transformers`, `huggingface-hub`, `einops` | HyenaDNA model loading |
| `h5py` | Precomputed embedding store |

---

## Data Requirements

| File | Description |
|------|-------------|
| `raw/*.bam` + `*.bam.bai` | Aligned WGS reads (e.g. NA12878 low-coverage) |
| `raw/*.vcf.gz` + `*.vcf.gz.tbi` | Truth SV VCF (e.g. 1000 Genomes Phase 3) |
| `raw/*.fa` + `*.fa.fai` | Reference FASTA — only needed for M1 precompute |

The VCF must contain records with `SVTYPE=DEL` or `ALT=<DEL>`. The pipeline filters for deletions only and ignores other SV types.

---

## Pipeline Stages

### Stage 0 — Precompute HyenaDNA embeddings *(M1 only)*

```bash
python scripts/precompute_hyenadna_embeddings.py \
    --fasta  raw/hs37d5.fa \
    --output data/hyenadna_embeddings.h5 \
    --device cuda \
    --chrom  20 21 22 \
    --genome-build hs37d5 \
    --resume          # skip chromosomes already present in the HDF5
```

Run once per reference genome. Output is reusable across samples. Uses 30 kbp core chunks with 1 kbp flanks (32 kbp total input) to fit HyenaDNA's 32 k context window.

---

### Stage 1–3 — Generate dataset

```bash
python scripts/generate_fused_dataset.py \
    --sample NA12878 \
    --bam    raw/NA12878.bam \
    --vcf    raw/truth.vcf.gz \
    --chroms 20,21,22 \
    --max-length 10000 \
    --del-count  500 \           # optional: cap deletions per chromosome
    --output-dir data/fused
```

**What it does:**

1. Loads deletions from VCF, filters by chromosome and length.
2. Refines each deletion's breakpoints using 3-cluster K-means on coverage depth + soft-clipping signals (61 bp rolling median pre-smoothed).
3. Slides a 50 bp window across each deletion → 256×256 RGB pileup PNG → manifest row.
4. Generates upstream and downstream non-deletion anchor windows (same length) as the negative class.

**Output:** `data/fused/NA12878/manifest.csv` with columns `image_path, chrom, position, label, length`.

---

### Stage 4a — Train

```bash
# M0: RGB-only baseline
python scripts/train_fused_model.py \
    --manifest data/fused/NA12878/manifest.csv \
    --model cnn \
    --train-chroms 20,21 --val-chroms 22 \
    --output models/m0_best.pth \
    --epochs 10

# M1: Fused (image + HyenaDNA FiLM)
python scripts/train_fused_model.py \
    --manifest     data/fused/NA12878/manifest.csv \
    --model        fused \
    --embeddings   data/hyenadna_embeddings.h5 \
    --train-chroms 20,21 --val-chroms 22 \
    --output       models/m1_best.pth \
    --epochs       10 \
    --stage-a-epochs 2 \
    --lr-cnn  1e-4 \
    --lr-film 1e-3
```

The chromosome split ensures the model learns general biological patterns rather than region-specific features.

---

### Stage 4b — Inference

```bash
# M0 inference
python scripts/call_fused_deletions.py \
    --manifest        data/fused/NA12878/manifest.csv \
    --model           cnn \
    --checkpoint      models/m0_best.pth \
    --predictions-out runs/m0_predictions.csv \
    --vcf-out         runs/m0_calls.vcf

# M1 inference
python scripts/call_fused_deletions.py \
    --manifest        data/fused/NA12878/manifest.csv \
    --model           fused \
    --checkpoint      models/m1_best.pth \
    --embeddings      data/hyenadna_embeddings.h5 \
    --predictions-out runs/m1_predictions.csv \
    --vcf-out         runs/m1_calls.vcf \
    --threshold       0.5
```

Adjacent positive windows on the same chromosome are merged into single deletion calls. The output VCF contains `SVTYPE=DEL`, `END`, `SVLEN`, `NWIN` (number of merged windows), and `MAXPROB` (max deletion probability).

---

### Full pipeline (one command)

```bash
./scripts/run_fused_pipeline.sh \
    --sample       NA12878 \
    --bam          raw/NA12878.bam \
    --vcf          raw/truth.vcf.gz \
    --fasta        raw/hs37d5.fa \
    --model        fused \
    --chroms       20,21,22 \
    --train-chroms 20,21 \
    --val-chroms   22 \
    --out-root     runs/experiment1
```

Use `--skip-precompute`, `--skip-generate`, `--skip-train`, `--skip-infer` to resume from an intermediate stage.

---

## CLI Reference

### `generate_fused_dataset.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--sample` | *required* | Sample ID (used for output directory naming) |
| `--bam` | *required* | Indexed BAM file |
| `--vcf` | *required* | Indexed VCF.gz with SV calls |
| `--chroms` | *required* | Comma-separated chromosomes (e.g. `20,21,22`) |
| `--output-dir` | `data/fused` | Root directory for images + manifest |
| `--max-length` | `10000` | Skip deletions longer than this (bp) |
| `--min-length` | `50` | Skip deletions shorter than this (bp) |
| `--del-count` | *none* | Optional cap on deletions per chromosome |
| `--no-refine` | *off* | Skip K-means breakpoint refinement |
| `--seed` | `42` | Random seed for per-chromosome capping |

### `train_fused_model.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | *required* | CSV from `generate_fused_dataset.py` |
| `--model` | `fused` | `cnn` (M0) or `fused` (M1) |
| `--embeddings` | *none* | HDF5 embedding file (required for `--model fused`) |
| `--output` | *required* | Path to save best checkpoint |
| `--train-chroms` | *required* | Comma-separated training chromosomes |
| `--val-chroms` | *required* | Comma-separated validation chromosomes |
| `--epochs` | `10` | Total training epochs |
| `--stage-a-epochs` | `2` | FiLM-only warm-up epochs (M1 only) |
| `--batch-size` | `128` | Batch size |
| `--lr-cnn` | `1e-4` | CNN backbone learning rate (Stage B) |
| `--lr-film` | `1e-3` | FiLM generator learning rate (both stages) |
| `--weight-decay` | `1e-6` | L2 regularisation |
| `--embed-dim` | `256` | HyenaDNA embedding dimension |

### `call_fused_deletions.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--manifest` | *required* | CSV from `generate_fused_dataset.py` |
| `--checkpoint` | *required* | Trained model state dict |
| `--model` | `fused` | `cnn` or `fused` |
| `--embeddings` | *none* | HDF5 file (required for `--model fused`) |
| `--predictions-out` | *required* | Output CSV with per-window probabilities |
| `--vcf-out` | *none* | Optional VCF with merged DEL calls |
| `--threshold` | `0.5` | P(deletion) threshold for positive classification |
| `--batch-size` | `64` | Inference batch size |
| `--embed-dim` | `256` | Must match training value |

---

## Python API

```python
from deepsv.models import ModernDeletionCNN, FusedDeepSV
from deepsv.data import FusedDataset
from deepsv.training import FiLMTrainer
from deepsv.training.trainer import ImageDataset, ModelTrainer
from deepsv.inference import FusedPredictor
from deepsv.inference.predictor import DeletionPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── M0: RGB-only ──────────────────────────────────────────────────────────

train_ds = ImageDataset(image_paths=train_paths, labels=train_labels, transform=transform)
model_m0 = ModernDeletionCNN(num_classes=2)
trainer_m0 = ModelTrainer(model_m0)
trainer_m0.setup_optimizer(learning_rate=1e-4)
trainer_m0.train(
    train_loader=DataLoader(train_ds, batch_size=128, shuffle=True),
    num_epochs=10,
    save_path="models/m0_best.pth",
)

# ── M1: Fused (image + HyenaDNA) ─────────────────────────────────────────

train_ds = FusedDataset(
    image_paths=train_paths,
    labels=train_labels,
    chroms=train_chroms,
    positions=train_positions,
    embeddings_h5="data/hyenadna_embeddings.h5",
    preload_chroms=["20", "21"],
    transform=transform,
)
model_m1 = FusedDeepSV(embed_dim=256, num_classes=2)
trainer_m1 = FiLMTrainer(model_m1)
trainer_m1.train(
    train_loader=DataLoader(train_ds, batch_size=128, shuffle=True),
    val_loader=val_loader,
    num_epochs=10,
    stage_a_epochs=2,
    lr_cnn=1e-4,
    lr_film=1e-3,
    save_path="models/m1_best.pth",
)

# ── Inference ─────────────────────────────────────────────────────────────

with FusedPredictor(
    model=model_m1,
    embeddings_h5="data/hyenadna_embeddings.h5",
    threshold=0.5,
    preload_chroms=["22"],
) as predictor:
    results = predictor.predict_batch(
        image_paths=paths, chroms=chroms, positions=positions
    )
```

---

## Acknowledgements

CADC is a modernized and extended evolution of the original **DeepSV** project. We gratefully acknowledge the authors for the foundational ideas and code:

*   **Repository**: [CSuperlei/DeepSV](https://github.com/CSuperlei/DeepSV)
*   **Paper**: Cai, L., Wu, Y., & Gao, J. (2019). *DeepSV: accurate calling of genomic deletions from high-throughput sequencing data using deep convolutional neural network.* BMC Bioinformatics, 20, 665. [https://doi.org/10.1186/s12859-019-3299-y](https://doi.org/10.1186/s12859-019-3299-y)
