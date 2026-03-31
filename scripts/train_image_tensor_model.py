#!/usr/bin/env python3
"""Train the BroadcastContextCNN on DeepSV2.5 image+tensor data.

Usage:
  python train_image_tensor_model.py \
      --data-root data/image_tensor_dataset \
      --context-channels 8 \
      --epochs 20 --batch-size 32
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, balanced_accuracy_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.models.multichannel_cnn import BroadcastContextCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

CLASS_NAMES = ["Non-Deletion", "Deletion"]

# Dataset
class ImageTensorDataset(Dataset):
    def __init__(self, file_paths, labels, context_channels=8, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.context_channels = context_channels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx], weights_only=False)
        img_tensor = data["image"]  # (3, H, W) float32 in [0, 1]

        if self.transform:
            img_tensor = self.transform(img_tensor)

        if "context" in data and self.context_channels > 0:
            ctx = data["context"]  # (K,)
            _, h, w = img_tensor.shape
            ctx_broadcast = ctx.unsqueeze(-1).unsqueeze(-1).expand(self.context_channels, h, w)
            tensor = torch.cat([img_tensor, ctx_broadcast], dim=0) # (3+K, H, W)
        else:
            tensor = img_tensor

        label = data["label"]
        return tensor, label


# Helpers
def get_chrom_from_pt_filename(filename: str) -> str:
    parts = filename.replace(".pt", "").split("_")
    if filename.startswith("del_"): return parts[1]
    elif filename.startswith("non_del_"): return parts[2]
    return None

def collect_split(del_dir: Path, non_del_dir: Path, chrom_set: set) -> Tuple[List[str], List[int]]:
    paths, labels = [], []
    if del_dir.exists():
        for f in sorted(del_dir.glob("*.pt")):
            if get_chrom_from_pt_filename(f.name) in chrom_set:
                paths.append(str(f)); labels.append(1)
    if non_del_dir.exists():
        for f in sorted(non_del_dir.glob("*.pt")):
            if get_chrom_from_pt_filename(f.name) in chrom_set:
                paths.append(str(f)); labels.append(0)
    return paths, labels

def compute_class_weights(labels: List[int], num_classes: int = 2) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(labels) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)

def build_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = np.bincount(labels).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weight = 1.0 / counts
    sample_weights = np.array([class_weight[l] for l in labels])
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels), replacement=True)


# Metrics
def compute_metrics(all_labels: np.ndarray, all_probs: np.ndarray, all_preds: np.ndarray) -> Dict:
    prec, rec, f1, sup = precision_recall_fscore_support(all_labels, all_preds, labels=[0, 1], zero_division=0)
    _, _, f1_weighted, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    pr_prec, pr_rec, _ = precision_recall_curve(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    return {
        "balanced_acc": bal_acc, "precision": prec, "recall": rec,
        "f1": f1, "f1_weighted": f1_weighted, "f1_macro": f1_macro,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "fpr": fpr, "tpr": tpr, "pr_prec": pr_prec, "pr_rec": pr_rec,
        "confusion_matrix": cm, "support": sup,
    }


# Training / evaluation
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, all_labels, all_probs, all_preds = 0.0, [], [], []
    pbar = tqdm(loader, desc="  Train", leave=False)
    for tensors, labels in pbar:
        tensors, labels = tensors.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(tensors)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * tensors.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset), np.array(all_labels), np.array(all_probs), np.array(all_preds)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, all_labels, all_probs, all_preds = 0.0, [], [], []
    pbar = tqdm(loader, desc="  Val  ", leave=False)
    for tensors, labels in pbar:
        tensors, labels = tensors.to(device), labels.to(device)
        outputs = model(tensors)
        loss = criterion(outputs, labels)

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * tensors.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return running_loss / len(loader.dataset), np.array(all_labels), np.array(all_probs), np.array(all_preds)


# Dashboard
def update_dashboard(history, val_metrics, epoch, output_dir):
    fig = plt.figure(figsize=(22, 18), constrained_layout=True)
    fig.suptitle(f"Tensor Training Dashboard — Epoch {epoch}", fontsize=16, fontweight="bold")
    gs = GridSpec(3, 3, figure=fig)
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(epochs, history["train_loss"], "o-", label="Train")
    ax.plot(epochs, history["val_loss"], "s-", label="Val")
    ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(epochs, history["train_bal_acc"], "o-", label="Train")
    ax.plot(epochs, history["val_bal_acc"], "s-", label="Val")
    ax.set_title("Balanced Accuracy"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, history["val_f1_weighted"], "s-", label="Val Weighted F1")
    ax.plot(epochs, history["val_f1_macro"], "^-", label="Val Macro F1")
    ax.set_title("F1 Scores"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    for panel_idx, (metric, title) in enumerate([("prec", "Precision"), ("rec", "Recall"), ("f1", "F1")]):
        ax = fig.add_subplot(gs[1, panel_idx])
        ax.plot(epochs, history[f"val_{metric}_0"], "s-", label=CLASS_NAMES[0])
        ax.plot(epochs, history[f"val_{metric}_1"], "^-", label=CLASS_NAMES[1])
        ax.set_title(f"{title} (per class)"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    cm = val_metrics["confusion_matrix"]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix (Val)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(val_metrics["fpr"], val_metrics["tpr"], lw=2, label=f'ROC (AUC={val_metrics["roc_auc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5); ax.set_title("ROC Curve (Val)"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(val_metrics["pr_rec"], val_metrics["pr_prec"], lw=2, label=f'PR (AP={val_metrics["pr_auc"]:.3f})')
    ax.set_title("Precision-Recall (Val)"); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "training_dashboard.png", dpi=150)
    plt.close(fig)


# Main
def main():
    parser = argparse.ArgumentParser(description="Train model on image+tensor dataset.")
    parser.add_argument("--data-root", default="data/image_tensor_dataset")
    parser.add_argument("--output", default=None)
    parser.add_argument("--context-channels", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--use-sampler", action="store_true")
    parser.add_argument("--train-chroms", default="1,2,3,4,5,6,7,8,9,10,11")
    parser.add_argument("--val-chroms", default="12,13,14,15,16,17,18,19,20,21,22")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Device (cuda/cpu). Auto-detect if None.")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    data_root = Path(args.data_root)
    del_dir = data_root / "deletion"
    non_del_dir = data_root / "non_deletion"

    if not del_dir.exists() and not non_del_dir.exists():
        logger.error("No dataset found at %s", data_root)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else Path("models") / "image_tensor"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_chroms = set(args.train_chroms.split(","))
    val_chroms = set(args.val_chroms.split(","))
    train_paths, train_labels = collect_split(del_dir, non_del_dir, train_chroms)
    val_paths, val_labels = collect_split(del_dir, non_del_dir, val_chroms)

    logger.info("Train: %d samples", len(train_paths))
    logger.info("Val:   %d samples", len(val_paths))
    if not train_paths or not val_paths:
        sys.exit(1)

    class_weights = compute_class_weights(train_labels)
    
    # Identify actual context components
    first_sample = torch.load(train_paths[0], weights_only=False)
    detected_k = first_sample.get("context", torch.zeros(0)).shape[0]
    if detected_k > 0 and detected_k != args.context_channels:
        args.context_channels = detected_k

    # Set up transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageTensorDataset(train_paths, train_labels, args.context_channels, transform=train_transform)
    val_dataset = ImageTensorDataset(val_paths, val_labels, args.context_channels, transform=val_transform)

    sampler = build_weighted_sampler(train_labels) if args.use_sampler else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Using device: %s", device)
    
    # 3 image channels + K context channels
    model = BroadcastContextCNN(num_classes=2, context_channels=args.context_channels, alignment_channels=3).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history_keys = ["train_loss", "val_loss", "train_bal_acc", "val_bal_acc", "train_f1_weighted", "val_f1_weighted", "val_f1_macro", "val_prec_0", "val_prec_1", "val_rec_0", "val_rec_1", "val_f1_0", "val_f1_1", "val_roc_auc", "val_pr_auc"]
    history = {k: [] for k in history_keys}
    best_val_f1 = 0.0
    model_path = output_dir / "best_model.pth"

    total_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        logger.info("─── Epoch %d / %d ───", epoch, args.epochs)

        train_loss, tr_labels, tr_probs, tr_preds = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_m = compute_metrics(tr_labels, tr_probs, tr_preds)

        val_loss, va_labels, va_probs, va_preds = evaluate(model, val_loader, criterion, device)
        val_m = compute_metrics(va_labels, va_probs, va_preds)
        scheduler.step()

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_bal_acc"].append(train_m["balanced_acc"]); history["val_bal_acc"].append(val_m["balanced_acc"])
        history["train_f1_weighted"].append(train_m["f1_weighted"]); history["val_f1_weighted"].append(val_m["f1_weighted"])
        history["val_f1_macro"].append(val_m["f1_macro"])
        history["val_prec_0"].append(val_m["precision"][0]); history["val_prec_1"].append(val_m["precision"][1])
        history["val_rec_0"].append(val_m["recall"][0]); history["val_rec_1"].append(val_m["recall"][1])
        history["val_f1_0"].append(val_m["f1"][0]); history["val_f1_1"].append(val_m["f1"][1])
        history["val_roc_auc"].append(val_m["roc_auc"]); history["val_pr_auc"].append(val_m["pr_auc"])

        if val_m["f1_weighted"] > best_val_f1:
            best_val_f1 = val_m["f1_weighted"]
            torch.save(model.state_dict(), model_path)
            logger.info("  ★ New best model (Weighted-F1=%.4f) saved.", best_val_f1)

        update_dashboard(history, val_m, epoch, output_dir)

    logger.info("Training complete in %.1f min.", (time.time() - total_start) / 60)

    with open(output_dir / "final_metrics.txt", "w") as f:
        f.write(f"Mode: image_tensor\nContext channels: {args.context_channels}\nEpochs: {args.epochs}\nBest Weighted F1: {best_val_f1:.4f}\nFinal ROC-AUC: {history['val_roc_auc'][-1]:.4f}\nFinal PR-AUC: {history['val_pr_auc'][-1]:.4f}\nFinal Bal. Acc: {history['val_bal_acc'][-1]:.4f}\n")

if __name__ == "__main__":
    main()
