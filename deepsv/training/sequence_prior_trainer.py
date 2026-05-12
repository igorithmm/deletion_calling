"""Trainer for the sequence-only deletion prior model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.sequence_prior import SequenceDeletionPrior

logger = logging.getLogger(__name__)


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _best_threshold_tradeoff(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
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


class SequencePriorTrainer:
    """Train and validate :class:`SequenceDeletionPrior`."""

    def __init__(
        self,
        model: SequenceDeletionPrior,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler = None

    def setup_optimizer(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_patience: int = 3,
    ) -> None:
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.5,
            patience=lr_patience,
        )

    def _maybe_weight_loss(self, train_loader: DataLoader) -> None:
        labels = getattr(train_loader.dataset, "labels", None)
        if labels is None:
            return
        counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
        if np.any(counts == 0):
            logger.warning("Skipping class weights because counts are %s", counts)
            return
        total = int(counts.sum())
        weights = total / (2.0 * counts)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=self.device)
        )
        logger.info("Class counts=%s, weights=%s", counts.tolist(), weights.tolist())

    def train_epoch(
        self,
        dataloader: DataLoader,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        if self.optimizer is None:
            raise RuntimeError("Call setup_optimizer() before training.")

        self.model.train()
        running_loss = 0.0
        num_batches = 0
        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(dataloader, desc="Training sequence prior", leave=False)
        for embeddings, labels in pbar:
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1
            probs = torch.softmax(outputs.detach(), dim=1)[:, 1]
            preds = torch.argmax(outputs.detach(), dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            pbar.set_postfix({"loss": running_loss / max(1, num_batches)})

        return self._metrics(running_loss / max(1, len(dataloader)), all_labels, all_preds, all_probs)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []

        pbar = tqdm(dataloader, desc="Validating sequence prior", leave=False)
        for embeddings, labels in pbar:
            embeddings = embeddings.to(self.device, non_blocking=True).float()
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(embeddings)
            loss = self.criterion(outputs, labels)
            running_loss += float(loss.item())
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

        return self._metrics(running_loss / max(1, len(dataloader)), all_labels, all_preds, all_probs)

    def _metrics(self, loss: float, labels, preds, probs) -> Dict[str, float]:
        y_true = np.asarray(labels, dtype=np.int64)
        y_pred = np.asarray(preds, dtype=np.int64)
        y_prob = np.asarray(probs, dtype=np.float64)
        precision, recall, f1 = _binary_prf(y_true, y_pred)
        accuracy = 100.0 * float((y_true == y_pred).mean()) if y_true.size else 0.0
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = 0.5
        metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
        }
        metrics.update(_best_threshold_tradeoff(y_true, y_prob))
        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_path: Optional[Path] = None,
        checkpoint_metadata: Optional[Dict[str, object]] = None,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, float]:
        self._maybe_weight_loss(train_loader)
        best_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        checkpoint_metadata = checkpoint_metadata or {}
        start = time.time()

        for epoch in range(num_epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader, max_grad_norm=max_grad_norm)
            val_metrics = self.validate(val_loader) if val_loader is not None else None
            score_metrics = val_metrics or train_metrics

            if self.scheduler is not None:
                self.scheduler.step(score_metrics["f1"])

            msg = (
                f"Epoch {epoch + 1}/{num_epochs} | {time.time() - t0:.1f}s | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.2f}% "
                f"F1 {train_metrics['f1']:.3f} AUC {train_metrics['auc']:.3f}"
            )
            if val_metrics is not None:
                msg += (
                    f" | val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.2f}% "
                    f"F1 {val_metrics['f1']:.3f} AUC {val_metrics['auc']:.3f}"
                )
            logger.info(msg)

            if score_metrics["f1"] > best_f1:
                best_f1 = score_metrics["f1"]
                best_metrics = score_metrics
                if save_path is not None:
                    self._save(save_path, best_metrics, checkpoint_metadata)

        logger.info(
            "Sequence-prior training done in %.1f min. Best F1=%.4f",
            (time.time() - start) / 60.0,
            best_f1,
        )
        return best_metrics

    def _save(
        self,
        path: Path,
        metrics: Dict[str, float],
        metadata: Dict[str, object],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "model_config": self.model.config(),
            "best_metrics": metrics,
            **metadata,
        }
        torch.save(payload, path)
        logger.info("New best sequence-prior checkpoint saved to %s", path)
