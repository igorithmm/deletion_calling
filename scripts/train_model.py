#!/usr/bin/env python3
"""
Train the CNN model with:
  - Selectable coloring mode  (--mode standard | kmer)
  - Chromosome-based split    (train: chr 1-11, test: chr 12-22)
  - Class-imbalance handling   (weighted loss + optional weighted sampler)
  - Comprehensive per-epoch metrics & live-updating plots

Metrics tracked every epoch:
  ▸ Train / Val loss
  ▸ Balanced accuracy, per-class & weighted F1
  ▸ Per-class Precision / Recall
  ▸ ROC-AUC, PR-AUC
  ▸ Confusion matrix snapshot
  ▸ ROC & Precision-Recall curves
"""
import argparse
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.models.cnn import ModernDeletionCNN
from deepsv.training.trainer import ImageDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Non-Deletion", "Deletion"]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_chrom_from_filename(filename: str) -> str:
    """Extract chromosome number from image filename.

    Supported formats:
      del_{chrom}_{start}_{end}.png
      non_del_up_{chrom}_{start}_{end}.png
      non_del_down_{chrom}_{start}_{end}.png
    """
    parts = filename.split("_")
    if filename.startswith("del_"):
        return parts[1]
    elif filename.startswith("non_del_"):
        # non_del_{up|down}_{chrom}_...
        return parts[3]
    return None


def collect_split(
    del_dir: Path,
    non_del_dir: Path,
    chrom_set: set,
) -> Tuple[List[str], List[int]]:
    """Collect image paths + labels restricted to *chrom_set*."""
    paths, labels = [], []

    for img in del_dir.glob("*.png"):
        if get_chrom_from_filename(img.name) in chrom_set:
            paths.append(str(img))
            labels.append(1)

    for img in non_del_dir.glob("*.png"):
        if get_chrom_from_filename(img.name) in chrom_set:
            paths.append(str(img))
            labels.append(0)

    return paths, labels


def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Per-sample weight sampler to oversample the minority class."""
    counts = np.bincount(labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weight = 1.0 / counts
    sample_weights = np.array([class_weight[l] for l in labels])
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Metrics collection
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_preds: np.ndarray,
) -> Dict:
    """Return a comprehensive metrics dictionary."""
    # Per-class precision / recall / F1
    prec, rec, f1, sup = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], zero_division=0,
    )
    # Weighted & macro F1
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0,
    )
    _, _, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0,
    )

    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    # ROC-AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # PR-AUC (positive class = Deletion = 1)
    pr_prec, pr_rec, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        "balanced_acc": bal_acc,
        "precision": prec,    # array len 2
        "recall": rec,
        "f1": f1,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "tpr": tpr,
        "pr_prec": pr_prec,
        "pr_rec": pr_rec,
        "confusion_matrix": cm,
        "support": sup,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Train for one epoch.  Returns (avg_loss, labels, probs, preds)."""
    model.train()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    pbar = tqdm(loader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs), np.array(all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate.  Returns (avg_loss, labels, probs, preds)."""
    model.eval()
    running_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        running_loss += loss.item() * images.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    return avg_loss, np.array(all_labels), np.array(all_probs), np.array(all_preds)


# ═══════════════════════════════════════════════════════════════════════════
# Live-updating plots
# ═══════════════════════════════════════════════════════════════════════════

def update_dashboard(
    history: Dict[str, list],
    val_metrics: Dict,
    epoch: int,
    output_dir: Path,
):
    """Render and save a multi-panel dashboard PNG that is updated every epoch."""

    fig = plt.figure(figsize=(22, 18), constrained_layout=True)
    fig.suptitle(f"Training Dashboard — Epoch {epoch}", fontsize=16, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig)

    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Panel 1: Loss ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, history["train_loss"], "o-", label="Train")
    ax.plot(epochs, history["val_loss"], "s-", label="Val")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Balanced Accuracy ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, history["train_bal_acc"], "o-", label="Train")
    ax.plot(epochs, history["val_bal_acc"], "s-", label="Val")
    ax.set_title("Balanced Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Weighted & Macro F1 ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, history["val_f1_weighted"], "s-", label="Val Weighted F1")
    ax.plot(epochs, history["val_f1_macro"], "^-", label="Val Macro F1")
    ax.plot(epochs, history["train_f1_weighted"], "o--", alpha=0.6, label="Train Weighted F1")
    ax.set_title("F1 Scores")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Per-class Precision ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(epochs, history["val_prec_0"], "s-", label=f"Val {CLASS_NAMES[0]}")
    ax.plot(epochs, history["val_prec_1"], "^-", label=f"Val {CLASS_NAMES[1]}")
    ax.set_title("Precision (per class)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 5: Per-class Recall ───────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(epochs, history["val_rec_0"], "s-", label=f"Val {CLASS_NAMES[0]}")
    ax.plot(epochs, history["val_rec_1"], "^-", label=f"Val {CLASS_NAMES[1]}")
    ax.set_title("Recall (per class)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 6: Per-class F1 ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(epochs, history["val_f1_0"], "s-", label=f"Val {CLASS_NAMES[0]}")
    ax.plot(epochs, history["val_f1_1"], "^-", label=f"Val {CLASS_NAMES[1]}")
    ax.set_title("F1 (per class)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 7: Confusion matrix (latest epoch) ───────────────────────
    ax = fig.add_subplot(gs[2, 0])
    cm = val_metrics["confusion_matrix"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Val)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # ── Panel 8: ROC curve (latest epoch) ──────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(val_metrics["fpr"], val_metrics["tpr"], lw=2,
            label=f'ROC (AUC = {val_metrics["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_title("ROC Curve (Val)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Panel 9: Precision-Recall curve (latest epoch) ─────────────────
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(val_metrics["pr_rec"], val_metrics["pr_prec"], lw=2,
            label=f'PR (AP = {val_metrics["pr_auc"]:.3f})')
    ax.set_title("Precision-Recall Curve (Val)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "training_dashboard.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train CNN on standard or kmer dataset with imbalance handling "
                    "and live metrics dashboard.",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "kmer"],
        required=True,
        help="Coloring mode of the dataset to train on.",
    )
    parser.add_argument(
        "--data-root",
        default="data/NA12878_dataset",
        help="Root directory containing standard/ and kmer/ sub-trees.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for model & plots (default: models/<mode>).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--use-sampler",
        action="store_true",
        help="Use weighted random sampler in addition to weighted loss.",
    )
    parser.add_argument(
        "--train-chroms",
        default="1,2,3,4,5,6,7,8,9,10,11",
        help="Comma-separated chromosomes for training.",
    )
    parser.add_argument(
        "--val-chroms",
        default="12,13,14,15,16,17,18,19,20,21,22",
        help="Comma-separated chromosomes for validation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Seed everything ────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Resolve directories ────────────────────────────────────────────
    data_root = Path(args.data_root) / args.mode
    del_dir = data_root / "deletion"
    non_del_dir = data_root / "non_deletion"

    if not del_dir.exists() or not non_del_dir.exists():
        logger.error("Dataset directories not found under %s", data_root)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("models") / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Chromosome splits ──────────────────────────────────────────────
    train_chroms = set(args.train_chroms.split(","))
    test_chroms = set(args.val_chroms.split(","))

    train_paths, train_labels = collect_split(del_dir, non_del_dir, train_chroms)
    test_paths, test_labels = collect_split(del_dir, non_del_dir, test_chroms)

    logger.info("Mode:  %s", args.mode)
    logger.info("Train: %d images  (Del %d / Non-Del %d) — Chr 1-11",
                len(train_paths), sum(train_labels),
                len(train_labels) - sum(train_labels))
    logger.info("Test:  %d images  (Del %d / Non-Del %d) — Chr 12-22",
                len(test_paths), sum(test_labels),
                len(test_labels) - sum(test_labels))

    if len(train_paths) == 0 or len(test_paths) == 0:
        logger.error("Empty split detected — aborting.")
        sys.exit(1)

    # ── Class weights (for loss) ───────────────────────────────────────
    class_weights = compute_class_weights(train_labels)
    logger.info("Class weights: %s", class_weights.tolist())

    # ── Transforms ─────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(10),
        #transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageDataset(train_paths, train_labels, transform=train_transform)
    test_dataset = ImageDataset(test_paths, test_labels, transform=val_transform)

    # ── Sampler (optional, for additional imbalance handling) ──────────
    sampler = build_weighted_sampler(train_labels) if args.use_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Model, loss, optimizer, scheduler ──────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = ModernDeletionCNN(num_classes=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{param_count:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── History dict (accumulates across epochs for live plots) ────────
    history_keys = [
        "train_loss", "val_loss",
        "train_bal_acc", "val_bal_acc",
        "train_f1_weighted", "val_f1_weighted",
        "val_f1_macro",
        "val_prec_0", "val_prec_1",
        "val_rec_0", "val_rec_1",
        "val_f1_0", "val_f1_1",
        "val_roc_auc", "val_pr_auc",
    ]
    history: Dict[str, list] = {k: [] for k in history_keys}

    best_val_f1 = 0.0
    model_path = output_dir / "best_model.pth"

    # ── Training loop ──────────────────────────────────────────────────
    logger.info("Starting training for %d epochs …", args.epochs)
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        logger.info("─── Epoch %d / %d ───", epoch, args.epochs)

        # Train
        train_loss, tr_labels, tr_probs, tr_preds = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        train_m = compute_metrics(tr_labels, tr_probs, tr_preds)

        # Validate
        val_loss, va_labels, va_probs, va_preds = evaluate(
            model, test_loader, criterion, device,
        )
        val_m = compute_metrics(va_labels, va_probs, va_preds)

        # LR step
        scheduler.step()

        # Accumulate history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bal_acc"].append(train_m["balanced_acc"])
        history["val_bal_acc"].append(val_m["balanced_acc"])
        history["train_f1_weighted"].append(train_m["f1_weighted"])
        history["val_f1_weighted"].append(val_m["f1_weighted"])
        history["val_f1_macro"].append(val_m["f1_macro"])
        history["val_prec_0"].append(val_m["precision"][0])
        history["val_prec_1"].append(val_m["precision"][1])
        history["val_rec_0"].append(val_m["recall"][0])
        history["val_rec_1"].append(val_m["recall"][1])
        history["val_f1_0"].append(val_m["f1"][0])
        history["val_f1_1"].append(val_m["f1"][1])
        history["val_roc_auc"].append(val_m["roc_auc"])
        history["val_pr_auc"].append(val_m["pr_auc"])

        elapsed = time.time() - epoch_start

        # Console log
        logger.info(
            "  Loss  train=%.4f  val=%.4f  |  BalAcc  train=%.3f  val=%.3f",
            train_loss, val_loss, train_m["balanced_acc"], val_m["balanced_acc"],
        )
        logger.info(
            "  Val → Weighted-F1=%.3f  Macro-F1=%.3f  ROC-AUC=%.3f  PR-AUC=%.3f",
            val_m["f1_weighted"], val_m["f1_macro"],
            val_m["roc_auc"], val_m["pr_auc"],
        )
        logger.info(
            "  Val → Prec [%s=%.3f, %s=%.3f]  Rec [%s=%.3f, %s=%.3f]",
            CLASS_NAMES[0], val_m["precision"][0],
            CLASS_NAMES[1], val_m["precision"][1],
            CLASS_NAMES[0], val_m["recall"][0],
            CLASS_NAMES[1], val_m["recall"][1],
        )
        logger.info("  Epoch time: %.1fs  |  LR: %.2e", elapsed,
                     optimizer.param_groups[0]["lr"])

        # Save best model by weighted F1
        if val_m["f1_weighted"] > best_val_f1:
            best_val_f1 = val_m["f1_weighted"]
            torch.save(model.state_dict(), model_path)
            logger.info("  ★ New best model (Weighted-F1 = %.4f) saved.", best_val_f1)

        # Live dashboard update
        update_dashboard(history, val_m, epoch, output_dir)

    total_time = time.time() - total_start
    logger.info("Training complete in %.1f min.", total_time / 60)

    # ── Save final metrics file ────────────────────────────────────────
    metrics_path = output_dir / "final_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Mode:             {args.mode}\n")
        f.write(f"Epochs:           {args.epochs}\n")
        f.write(f"Best Weighted F1: {best_val_f1:.4f}\n")
        f.write(f"Final ROC-AUC:    {history['val_roc_auc'][-1]:.4f}\n")
        f.write(f"Final PR-AUC:     {history['val_pr_auc'][-1]:.4f}\n")
        f.write(f"Final Bal. Acc:   {history['val_bal_acc'][-1]:.4f}\n")
        f.write(f"\nPer-class (final epoch):\n")
        f.write(f"  {CLASS_NAMES[0]}: Prec={history['val_prec_0'][-1]:.4f}  "
                f"Rec={history['val_rec_0'][-1]:.4f}  F1={history['val_f1_0'][-1]:.4f}\n")
        f.write(f"  {CLASS_NAMES[1]}: Prec={history['val_prec_1'][-1]:.4f}  "
                f"Rec={history['val_rec_1'][-1]:.4f}  F1={history['val_f1_1'][-1]:.4f}\n")
    logger.info("Metrics written to %s", metrics_path)
    logger.info("Dashboard saved to %s", output_dir / "training_dashboard.png")


if __name__ == "__main__":
    main()
