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
underlying CNN exposes).

Stratified evaluation
─────────────────────
reported.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from sklearn.metrics import roc_auc_score
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


# ════════════════════════════════════════════════════════════════════════════
# Metrics
# ════════════════════════════════════════════════════════════════════════════


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Precision / recall / F1 for the positive class (label 1)."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _best_threshold_tradeoff(
    y_true: np.ndarray, probs: np.ndarray
) -> Dict[str, float]:
    """Find the validation threshold that maximizes positive-class F1."""
    if y_true.size == 0 or probs.size == 0:
        return {
            "best_threshold": 0.5,
            "best_threshold_precision": 0.0,
            "best_threshold_recall": 0.0,
            "best_threshold_f1": 0.0,
        }

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probs)))
    best = {
        "best_threshold": 0.5,
        "best_threshold_precision": 0.0,
        "best_threshold_recall": 0.0,
        "best_threshold_f1": -1.0,
    }
    for threshold in thresholds:
        y_pred = (probs >= threshold).astype(np.int64)
        precision, recall, f1 = _binary_prf(y_true, y_pred)
        if f1 > best["best_threshold_f1"]:
            best = {
                "best_threshold": float(threshold),
                "best_threshold_precision": precision,
                "best_threshold_recall": recall,
                "best_threshold_f1": f1,
            }
    return best


def _bucket_f1_summary(metrics: Dict[str, float]) -> str:
    parts = []
    for name, _, _ in LENGTH_BUCKETS:
        key = f"f1_len_{name}"
        if key in metrics:
            parts.append(f"{name}={metrics[key]:.3f}")
    return ", ".join(parts)


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

    def _build_stage_a_optimizer(
        self, lr_film: float, film_weight_decay: float
    ) -> None:
        """Stage A: only FiLM params are trainable."""
        self.model.freeze_backbone()
        self.optimizer = optim.Adam(
            [
                {
                    "params": list(self.model.film_parameters()),
                    "lr": lr_film,
                    "weight_decay": film_weight_decay,
                    "name": "film",
                }
            ],
        )

    def _build_stage_b_optimizer(
        self,
        lr_cnn: float,
        lr_film: float,
        weight_decay: float,
        film_weight_decay: float,
    ) -> None:
        """Stage B: joint training with two param groups."""
        self.model.unfreeze_backbone()
        self.optimizer = optim.Adam(
            [
                {
                    "params": list(self.model.cnn_parameters()),
                    "lr": lr_cnn,
                    "weight_decay": weight_decay,
                    "name": "cnn",
                },
                {
                    "params": list(self.model.film_parameters()),
                    "lr": lr_film,
                    "weight_decay": film_weight_decay,
                    "name": "film",
                },
            ],
        )

    # ------------------------------------------------------------------
    # Train / validate loops
    # ------------------------------------------------------------------

    def train_epoch(self, dataloader: DataLoader, max_grad_norm: float = 1.0) -> Dict[str, float]:
        """One epoch of training over ``(image, embedding, label)`` triples."""
        self.model.train()
        running_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[float] = []
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, embeddings, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            labels_dev = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images, embeddings)
            loss = self.criterion(outputs, labels_dev)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            probs = torch.softmax(outputs.data, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

            pbar.set_postfix(
                {
                    "loss": running_loss / max(1, num_batches),
                }
            )

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        precision, recall, f1 = _binary_prf(y_true, y_pred)
        acc = 100.0 * float((y_pred == y_true).mean()) if len(y_true) else 0.0

        try:
            auc = roc_auc_score(y_true, all_probs)
        except ValueError:
            auc = 0.5

        return {
            "loss": running_loss / max(1, len(dataloader)),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
        sample_lengths: Optional[Sequence[int]] = None,
        sample_breakpoint_distances: Optional[Sequence[float]] = None,
    ) -> Dict[str, float]:
        """Validate, optionally with stratified metrics.

        .. important::
            Stratified metrics assume the iteration order of ``dataloader``
            matches the order of ``sample_lengths`` /
            ``sample_breakpoint_distances``. The loader **must** therefore
            be constructed with ``shuffle=False`` (no ``RandomSampler``).
            This is checked and a warning is logged on mismatch.

        Args:
            dataloader: Validation loader (must be unshuffled if any
                stratified arg is supplied).
            sample_lengths: List aligned with the dataset; deletion length
                in bp per sample.
            sample_breakpoint_distances: List aligned with the dataset;
                external breakpoint-distance estimate per sample.
        """
        # Guard against silent misalignment when callers shuffle by accident.
        any_stratified = any(
            v is not None for v in (sample_lengths, sample_breakpoint_distances)
        )
        if any_stratified:
            from torch.utils.data import RandomSampler  # local import

            sampler = getattr(dataloader, "sampler", None)
            if isinstance(sampler, RandomSampler) or getattr(
                dataloader, "shuffle", False
            ):
                logger.warning(
                    "validate(): dataloader appears to be shuffled but "
                    "stratified metrics were requested. Per-bucket numbers "
                    "may be misaligned. Pass shuffle=False on the val loader."
                )
        self.model.eval()
        running_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs: List[float] = []

        pbar = tqdm(dataloader, desc="Validating", leave=False)
        for images, embeddings, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            labels_dev = labels.to(self.device, non_blocking=True)

            outputs = self.model(images, embeddings)
            loss = self.criterion(outputs, labels_dev)
            running_loss += loss.item()

            probs = torch.softmax(outputs.data, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        probs_np = np.array(all_probs)
        precision, recall, f1 = _binary_prf(y_true, y_pred)
        acc = 100.0 * float((y_pred == y_true).mean()) if len(y_true) else 0.0

        try:
            auc = roc_auc_score(y_true, all_probs)
        except ValueError:
            auc = 0.5

        metrics: Dict[str, float] = {
            "loss": running_loss / max(1, len(dataloader)),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
        metrics.update(_best_threshold_tradeoff(y_true, probs_np))

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
        film_weight_decay: float = 1e-4,
        save_path: Optional[Path] = None,
        validate_kwargs: Optional[Dict] = None,
        max_grad_norm: float = 1.0,
        lr_patience: int = 3,
    ) -> Dict[str, float]:
        """Run the two-stage training schedule.

        Returns the best-epoch validation metrics dict.
        """
        # Compute class weights to handle imbalance
        if hasattr(train_loader.dataset, "labels"):
            dataset_labels = np.array(train_loader.dataset.labels)
            class_counts = np.bincount(dataset_labels)
            if len(class_counts) == 2:
                total = len(dataset_labels)
                weights = total / (2.0 * class_counts)
                class_weights = torch.FloatTensor(weights).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info(
                    f"Class imbalance handled. Counts: {class_counts}. Using weights: {weights}"
                )

        validate_kwargs = validate_kwargs or {}
        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        start = time.time()

        logger.info(
            "FiLMTrainer: %d total epochs, stage_a=%d, stage_b=%d on %s",
            num_epochs,
            min(stage_a_epochs, num_epochs),
            max(0, num_epochs - stage_a_epochs),
            self.device,
        )

        # ---- Stage A ----------------------------------------------------
        stage_a_epochs = min(stage_a_epochs, num_epochs)
        if stage_a_epochs > 0:
            logger.info(
                "=== Stage A: FiLM-only (lr=%g, weight_decay=%g) ===",
                lr_film,
                film_weight_decay,
            )
            self._build_stage_a_optimizer(
                lr_film=lr_film, film_weight_decay=film_weight_decay
            )
            scheduler_a = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=max(1, lr_patience // 2)
            )
            for epoch in range(stage_a_epochs):
                t0 = time.time()
                tm = self.train_epoch(train_loader, max_grad_norm=max_grad_norm)
                vm = (
                    self.validate(val_loader, **validate_kwargs) if val_loader else None
                )
                self._log_epoch(epoch + 1, num_epochs, "A", t0, tm, vm)

                old_lrs = self._lr_state()
                scheduler_a.step(vm["f1"] if vm else tm["f1"])
                new_lrs = self._lr_state()
                if old_lrs != new_lrs:
                    logger.info(
                        "Stage A: Learning rates reduced from %s to %s",
                        old_lrs,
                        new_lrs,
                    )

                if vm and vm["f1"] > best_f1:
                    best_f1 = vm["f1"]
                    best_metrics = vm
                    if save_path:
                        self._save(save_path)

        # ---- Stage B ----------------------------------------------------
        stage_b_epochs = num_epochs - stage_a_epochs
        if stage_b_epochs > 0:
            logger.info(
                "=== Stage B: joint (lr_cnn=%g, lr_film=%g, "
                "weight_decay_cnn=%g, weight_decay_film=%g) ===",
                lr_cnn,
                lr_film,
                weight_decay,
                film_weight_decay,
            )
            self._build_stage_b_optimizer(
                lr_cnn=lr_cnn,
                lr_film=lr_film,
                weight_decay=weight_decay,
                film_weight_decay=film_weight_decay,
            )
            scheduler_b = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=lr_patience
            )
            for epoch in range(stage_a_epochs, num_epochs):
                t0 = time.time()
                tm = self.train_epoch(train_loader, max_grad_norm=max_grad_norm)
                vm = (
                    self.validate(val_loader, **validate_kwargs) if val_loader else None
                )
                self._log_epoch(epoch + 1, num_epochs, "B", t0, tm, vm)

                old_lrs = self._lr_state()
                scheduler_b.step(vm["f1"] if vm else tm["f1"])
                new_lrs = self._lr_state()
                if old_lrs != new_lrs:
                    logger.info(
                        "Stage B: Learning rates reduced from %s to %s",
                        old_lrs,
                        new_lrs,
                    )

                if vm and vm["f1"] > best_f1:
                    best_f1 = vm["f1"]
                    best_metrics = vm
                    if save_path:
                        self._save(save_path)

        logger.info(
            "Training done in %.1f min. Best val F1 = %.4f",
            (time.time() - start) / 60.0,
            best_f1,
        )
        if best_metrics and "best_threshold" in best_metrics:
            logger.info(
                "Recommended validation threshold after training: %.4f "
                "(P=%.3f, R=%.3f, F1=%.3f on best validation epoch)",
                best_metrics["best_threshold"],
                best_metrics["best_threshold_precision"],
                best_metrics["best_threshold_recall"],
                best_metrics["best_threshold_f1"],
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
            f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.2f}% "
            f"P {train_metrics['precision']:.3f} R {train_metrics['recall']:.3f} "
            f"F1 {train_metrics['f1']:.3f} AUC {train_metrics['auc']:.3f}"
        )
        if val_metrics is not None:
            msg += (
                f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.2f}% "
                f"P {val_metrics['precision']:.3f} R {val_metrics['recall']:.3f} "
                f"F1 {val_metrics['f1']:.3f} AUC {val_metrics['auc']:.3f}"
            )
        logger.info(msg)
        if val_metrics is not None and "best_threshold" in val_metrics:
            logger.info(
                "Val threshold sweep: threshold=%.4f P=%.3f R=%.3f F1=%.3f",
                val_metrics["best_threshold"],
                val_metrics["best_threshold_precision"],
                val_metrics["best_threshold_recall"],
                val_metrics["best_threshold_f1"],
            )
        if val_metrics is not None:
            bucket_summary = _bucket_f1_summary(val_metrics)
            if bucket_summary:
                logger.info("Val F1 by deletion length: %s", bucket_summary)

    def _lr_state(self) -> Dict[str, float]:
        if self.optimizer is None:
            return {}
        return {
            pg.get("name", f"group{idx}"): float(pg["lr"])
            for idx, pg in enumerate(self.optimizer.param_groups)
        }

    def _save(self, path: Path) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info("New best F1 — checkpoint saved to %s", path)
