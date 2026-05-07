"""Two-stage trainer for the FiLM-conditioned FusedDeepSV model.

Stages
──────
* **Stage A** (default 2 epochs): backbone CNN frozen, FiLM generators only.
  Single param group at ``lr_film`` (default 1e-3).
* **Stage B** (remaining epochs): backbone unfrozen, joint training with two
  param groups: CNN at ``lr_cnn`` (default 1e-4), FiLM at ``lr_film``
  (default 1e-3).

Best checkpoint is selected by **validation F1** (positive class), not
accuracy. Loss is binary cross-entropy on the 2-class softmax head
(equivalent to ``nn.CrossEntropyLoss`` on logits, which is what the
underlying ``DeletionCNN`` exposes).

Stratified evaluation
─────────────────────
When a per-sample length and/or repeat-class is supplied to :meth:`validate`
(via the dataset, see :func:`stratify_by_length` and
:func:`stratify_by_repeat`), per-bucket precision / recall / F1 / mean
breakpoint distance are also reported.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.fused_cnn import FusedDeepSV

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# Stratification helpers
# ════════════════════════════════════════════════════════════════════════════

# Spec-defined deletion-length buckets, in bp.
LENGTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("50-200", 50, 200),
    ("200-500", 200, 500),
    ("500-1k", 500, 1_000),
    ("1k-5k", 1_000, 5_000),
    ("5k-10k", 5_000, 10_000),
)

# Repeat-class labels.
REPEAT_CLASSES: Tuple[str, ...] = ("unique", "simple-repeat", "segmental-dup")


def stratify_by_length(length_bp: int) -> Optional[str]:
    """Return the bucket name for a deletion length, or ``None`` if it falls
    outside all configured ranges (half-open ``[lo, hi)``)."""
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= length_bp < hi:
            return name
    return None


def stratify_by_repeat(
    chrom: str,
    start: int,
    end: int,
    repeat_bed: str,
    simple_repeat_bed: Optional[str] = None,
    segmental_dup_bed: Optional[str] = None,
) -> str:
    """Classify a deletion as unique / simple-repeat / segmental-dup.

    Uses ``pybedtools`` to intersect the deletion interval with the supplied
    annotation BEDs. Priority (when overlapping multiple): segmental-dup >
    simple-repeat > unique.

    Args:
        chrom: Chromosome name.
        start: Deletion start (0-based).
        end: Deletion end (exclusive).
        repeat_bed: General RepeatMasker BED used as fallback when the
            split BEDs aren't provided.
        simple_repeat_bed: Optional simple-repeat-only BED.
        segmental_dup_bed: Optional segmental-dup-only BED.
    """
    import pybedtools  # local import — heavy, optional dep

    interval = pybedtools.BedTool(f"{chrom}\t{start}\t{end}\n", from_string=True)

    if segmental_dup_bed and Path(segmental_dup_bed).exists():
        if interval.intersect(pybedtools.BedTool(segmental_dup_bed), u=True).count() > 0:
            return "segmental-dup"
    if simple_repeat_bed and Path(simple_repeat_bed).exists():
        if interval.intersect(pybedtools.BedTool(simple_repeat_bed), u=True).count() > 0:
            return "simple-repeat"
    if repeat_bed and Path(repeat_bed).exists():
        if interval.intersect(pybedtools.BedTool(repeat_bed), u=True).count() > 0:
            return "simple-repeat"
    return "unique"


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════


def _binary_prf(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """Precision / recall / F1 for the positive class (label 1)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════


class FiLMTrainer:
    """Two-stage trainer for :class:`FusedDeepSV`.

    Designed to be a drop-in companion to
    :class:`deepsv.training.trainer.ModelTrainer`: same interface shape, but
    consuming ``(image, embedding, label)`` triples from a
    :class:`deepsv.data.fused_dataset.FusedDataset`.
    """

    def __init__(
        self,
        model: FusedDeepSV,
        device: Optional[torch.device] = None,
    ) -> None:
        """Args:
            model: The :class:`FusedDeepSV` to train.
            device: Device for training. Defaults to CUDA if available.
        """
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer: Optional[optim.Optimizer] = None

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def _build_stage_a_optimizer(self, lr_film: float, weight_decay: float) -> None:
        """Stage A: only FiLM params are trainable."""
        self.model.freeze_backbone()
        self.optimizer = optim.Adam(
            self.model.film_parameters(),
            lr=lr_film,
            weight_decay=weight_decay,
        )

    def _build_stage_b_optimizer(
        self, lr_cnn: float, lr_film: float, weight_decay: float
    ) -> None:
        """Stage B: joint training with two param groups."""
        self.model.unfreeze_backbone()
        self.optimizer = optim.Adam(
            [
                {"params": list(self.model.cnn_parameters()), "lr": lr_cnn},
                {"params": list(self.model.film_parameters()), "lr": lr_film},
            ],
            weight_decay=weight_decay,
        )

    # ------------------------------------------------------------------
    # Train / validate loops
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """One epoch of training over ``(image, embedding, label)`` triples."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, embeddings, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images, embeddings)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                "loss": running_loss / max(1, (total // labels.size(0))),
                "acc": 100.0 * correct / max(1, total),
            })

        return {
            "loss": running_loss / max(1, len(dataloader)),
            "accuracy": 100.0 * correct / max(1, total),
        }

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        sample_lengths: Optional[Sequence[int]] = None,
        sample_repeat_classes: Optional[Sequence[str]] = None,
        sample_breakpoint_distances: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Validate, optionally with stratified metrics.

        .. important::
            Stratified metrics assume the iteration order of ``dataloader``
            matches the order of ``sample_lengths`` / ``sample_repeat_classes``
            / ``sample_breakpoint_distances``. The loader **must** therefore
            be constructed with ``shuffle=False`` (no ``RandomSampler``).
            This is checked and a warning is logged on mismatch.

        Args:
            dataloader: Validation loader (must be unshuffled if any
                stratified arg is supplied).
            sample_lengths: List aligned with the dataset; deletion length
                in bp per sample.
            sample_repeat_classes: List aligned with the dataset; one of
                ``"unique"`` / ``"simple-repeat"`` / ``"segmental-dup"``.
            sample_breakpoint_distances: List aligned with the dataset;
                external breakpoint-distance estimate per sample.
        """
        # Guard against silent misalignment when callers shuffle by accident.
        any_stratified = any(
            v is not None for v in (
                sample_lengths, sample_repeat_classes, sample_breakpoint_distances
            )
        )
        if any_stratified:
            from torch.utils.data import RandomSampler  # local import
            sampler = getattr(dataloader, "sampler", None)
            if isinstance(sampler, RandomSampler) or getattr(dataloader, "shuffle", False):
                logger.warning(
                    "validate(): dataloader appears to be shuffled but "
                    "stratified metrics were requested. Per-bucket numbers "
                    "may be misaligned. Pass shuffle=False on the val loader."
                )
        self.model.eval()
        running_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []

        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, embeddings, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            labels_dev = labels.to(self.device, non_blocking=True)

            outputs = self.model(images, embeddings)
            loss = self.criterion(outputs, labels_dev)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        precision, recall, f1 = _binary_prf(y_true, y_pred)
        acc = 100.0 * float((y_pred == y_true).mean()) if len(y_true) else 0.0

        metrics: Dict[str, float] = {
            "loss": running_loss / max(1, len(dataloader)),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        if sample_breakpoint_distances is not None:
            bps = np.asarray(sample_breakpoint_distances, dtype=np.float64)
            metrics["mean_bp_dist"] = float(bps.mean()) if bps.size else 0.0

        # Stratified reporting -------------------------------------------------
        n = len(y_true)

        if sample_lengths is not None and len(sample_lengths) == n:
            lengths = np.asarray(sample_lengths)
            for name, lo, hi in LENGTH_BUCKETS:
                mask = (lengths >= lo) & (lengths < hi)
                if mask.sum() == 0:
                    continue
                p, r, f = _binary_prf(y_true[mask], y_pred[mask])
                metrics[f"f1_len_{name}"] = f
                metrics[f"precision_len_{name}"] = p
                metrics[f"recall_len_{name}"] = r
                if sample_breakpoint_distances is not None:
                    metrics[f"mean_bp_dist_len_{name}"] = float(
                        np.asarray(sample_breakpoint_distances)[mask].mean()
                    )

        if sample_repeat_classes is not None and len(sample_repeat_classes) == n:
            classes = np.asarray(sample_repeat_classes)
            for cls in REPEAT_CLASSES:
                mask = classes == cls
                if mask.sum() == 0:
                    continue
                p, r, f = _binary_prf(y_true[mask], y_pred[mask])
                metrics[f"f1_repeat_{cls}"] = f
                metrics[f"precision_repeat_{cls}"] = p
                metrics[f"recall_repeat_{cls}"] = r
                if sample_breakpoint_distances is not None:
                    metrics[f"mean_bp_dist_repeat_{cls}"] = float(
                        np.asarray(sample_breakpoint_distances)[mask].mean()
                    )

        return metrics

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        stage_a_epochs: int = 2,
        lr_cnn: float = 1e-4,
        lr_film: float = 1e-3,
        weight_decay: float = 1e-6,
        save_path: Optional[Path] = None,
        validate_kwargs: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Run the two-stage training schedule.

        Returns the best-epoch validation metrics dict.
        """
        validate_kwargs = validate_kwargs or {}
        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        start = time.time()

        logger.info(
            "FiLMTrainer: %d total epochs, stage_a=%d, stage_b=%d on %s",
            num_epochs,
            stage_a_epochs,
            num_epochs - stage_a_epochs,
            self.device,
        )

        # ---- Stage A ----------------------------------------------------
        if stage_a_epochs > 0:
            logger.info("=== Stage A: FiLM-only (lr=%g) ===", lr_film)
            self._build_stage_a_optimizer(lr_film=lr_film, weight_decay=weight_decay)
            for epoch in range(stage_a_epochs):
                t0 = time.time()
                tm = self.train_epoch(train_loader)
                vm = self.validate(val_loader, **validate_kwargs) if val_loader else None
                self._log_epoch(epoch + 1, num_epochs, "A", t0, tm, vm)
                if vm and vm["f1"] > best_f1 and save_path:
                    best_f1 = vm["f1"]
                    best_metrics = vm
                    self._save(save_path)

        # ---- Stage B ----------------------------------------------------
        stage_b_epochs = num_epochs - stage_a_epochs
        if stage_b_epochs > 0:
            logger.info(
                "=== Stage B: joint (lr_cnn=%g, lr_film=%g) ===", lr_cnn, lr_film
            )
            self._build_stage_b_optimizer(
                lr_cnn=lr_cnn, lr_film=lr_film, weight_decay=weight_decay
            )
            for epoch in range(stage_a_epochs, num_epochs):
                t0 = time.time()
                tm = self.train_epoch(train_loader)
                vm = self.validate(val_loader, **validate_kwargs) if val_loader else None
                self._log_epoch(epoch + 1, num_epochs, "B", t0, tm, vm)
                if vm and vm["f1"] > best_f1 and save_path:
                    best_f1 = vm["f1"]
                    best_metrics = vm
                    self._save(save_path)

        logger.info(
            "Training done in %.1f min. Best val F1 = %.4f",
            (time.time() - start) / 60.0,
            best_f1,
        )
        return best_metrics

    # ------------------------------------------------------------------
    # Logging / IO
    # ------------------------------------------------------------------

    def _log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        stage: str,
        t0: float,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        dt = time.time() - t0
        msg = (
            f"[Stage {stage}] Epoch {epoch}/{total_epochs} | {dt:.1f}s | "
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.2f}%"
        )
        if val_metrics is not None:
            msg += (
                f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.2f}% "
                f"P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
                f"F1 {val_metrics['f1']:.3f}"
            )
        logger.info(msg)

    def _save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info("New best F1 — checkpoint saved to %s", path)

    def save_model(self, path: Path) -> None:
        """Save the full model state dict."""
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: Path) -> None:
        """Load a model state dict from disk."""
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        logger.info("Model loaded from %s", path)
