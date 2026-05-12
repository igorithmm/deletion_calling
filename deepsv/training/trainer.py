"""Training utilities for the CNN model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, Dict, Sequence, Tuple
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import numpy as np

logger = logging.getLogger(__name__)

LENGTH_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ("50-200", 50, 200),
    ("200-500", 200, 500),
    ("500-1k", 500, 1_000),
    ("1k-5k", 1_000, 5_000),
    ("5k-10k", 5_000, 10_000),
)


def _binary_prf(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
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


def _metric_suffix(value: object) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "unknown"


def _neg_type_fpr_summary(metrics: Dict[str, float]) -> str:
    parts = []
    for key in sorted(metrics):
        if not key.startswith("fpr_neg_"):
            continue
        name = key.removeprefix("fpr_neg_")
        n_key = f"n_neg_{name}"
        if n_key in metrics:
            parts.append(f"{name}={metrics[key]:.3f} (n={int(metrics[n_key])})")
        else:
            parts.append(f"{name}={metrics[key]:.3f}")
    return ", ".join(parts)


class ImageDataset(Dataset):
    """Dataset for loading images and labels"""
    
    def __init__(self, image_paths: list, labels: list, transform=None):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: List of labels (0 or 1)
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class ModelTrainer:
    """Handles model training"""
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            device: Device to train on (CPU or CUDA)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
    
    def setup_optimizer(self,
                       learning_rate: float = 0.001,
                       weight_decay: float = 1e-6,
                       scheduler_step_size: Optional[int] = None):
        """
        Setup optimizer and learning rate scheduler
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            scheduler_step_size: Step size for learning rate decay
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        if scheduler_step_size:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_step_size,
                gamma=0.1
            )
    
    def train_epoch(self, dataloader: DataLoader, max_grad_norm: float = 1.0) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader for training data
            max_grad_norm: Maximum norm for gradient clipping
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            num_batches += 1
            probs = torch.softmax(outputs.data, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / num_batches
            })
        
        epoch_loss = running_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
        
        return {
            'loss': epoch_loss, 'accuracy': acc, 
            'precision': precision, 'recall': recall, 
            'f1': f1, 'auc': auc
        }
    
    def validate(
        self,
        dataloader: DataLoader,
        sample_lengths: Optional[Sequence[int]] = None,
        sample_neg_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        num_batches = 0
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                num_batches += 1
                probs = torch.softmax(outputs.data, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / num_batches
                })
        
        val_loss = running_loss / len(dataloader)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        probs_np = np.array(all_probs)

        acc = accuracy_score(all_labels, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
        
        metrics = {
            'loss': val_loss, 'accuracy': acc, 
            'precision': precision, 'recall': recall, 
            'f1': f1, 'auc': auc
        }
        metrics.update(_best_threshold_tradeoff(y_true, probs_np))

        if sample_lengths is not None and len(sample_lengths) == len(y_true):
            lengths = np.asarray(sample_lengths)
            for name, lo, hi in LENGTH_BUCKETS:
                mask = (lengths >= lo) & (lengths < hi)
                if mask.sum() == 0:
                    continue
                p, r, bucket_f1 = _binary_prf(y_true[mask], y_pred[mask])
                metrics[f"f1_len_{name}"] = bucket_f1
                metrics[f"precision_len_{name}"] = p
                metrics[f"recall_len_{name}"] = r

        if sample_neg_types is not None and len(sample_neg_types) == len(y_true):
            neg_types = np.asarray(sample_neg_types, dtype=object)
            for raw_name in sorted({str(x or "unknown") for x in neg_types[y_true == 0]}):
                if raw_name == "unknown":
                    type_mask = np.asarray([not x for x in neg_types], dtype=bool)
                else:
                    type_mask = neg_types == raw_name
                mask = (y_true == 0) & type_mask
                if mask.sum() == 0:
                    continue
                suffix = _metric_suffix(raw_name)
                fp = int(((y_pred == 1) & mask).sum())
                metrics[f"fpr_neg_{suffix}"] = fp / int(mask.sum())
                metrics[f"n_neg_{suffix}"] = float(mask.sum())

        return metrics
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10,
             save_path: Optional[Path] = None,
             validate_kwargs: Optional[Dict] = None,
             max_grad_norm: float = 1.0):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training
            val_loader: Optional DataLoader for validation
            num_epochs: Number of training epochs
            save_path: Path to save best model
            max_grad_norm: Maximum norm for gradient clipping
        """
        import time
        
        # Compute class weights to handle imbalance
        if hasattr(train_loader.dataset, 'labels'):
            dataset_labels = np.array(train_loader.dataset.labels)
            class_counts = np.bincount(dataset_labels)
            if len(class_counts) == 2:
                total = len(dataset_labels)
                weights = total / (2.0 * class_counts)
                class_weights = torch.FloatTensor(weights).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
                logger.info(f"Class imbalance handled. Counts: {class_counts}. Using weights: {weights}")
        
        validate_kwargs = validate_kwargs or {}
        best_val_f1 = -1.0
        best_val_metrics: Dict[str, float] = {}
        start_time = time.time()
        
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}...")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            logger.info(f"--- Epoch {epoch+1}/{num_epochs} Started ---")
            
            # Train
            train_metrics = self.train_epoch(train_loader, max_grad_norm=max_grad_norm)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader, **validate_kwargs)
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} Finished - Duration: {epoch_duration:.2f}s")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
                            f"P: {train_metrics['precision']:.3f}, R: {train_metrics['recall']:.3f}, "
                            f"F1: {train_metrics['f1']:.3f}, AUC: {train_metrics['auc']:.3f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                            f"P: {val_metrics['precision']:.3f}, R: {val_metrics['recall']:.3f}, "
                            f"F1: {val_metrics['f1']:.3f}, AUC: {val_metrics['auc']:.3f}")
                if "best_threshold" in val_metrics:
                    logger.info(
                        "Val threshold sweep: threshold=%.4f P=%.3f R=%.3f F1=%.3f",
                        val_metrics["best_threshold"],
                        val_metrics["best_threshold_precision"],
                        val_metrics["best_threshold_recall"],
                        val_metrics["best_threshold_f1"],
                    )
                bucket_summary = _bucket_f1_summary(val_metrics)
                if bucket_summary:
                    logger.info("Val F1 by deletion length: %s", bucket_summary)
                neg_summary = _neg_type_fpr_summary(val_metrics)
                if neg_summary:
                    logger.info("Val false-positive rate by negative type: %s", neg_summary)
                
                # Save best model based on F1 instead of accuracy
                if val_metrics['f1'] > best_val_f1:
                    best_val_f1 = val_metrics['f1']
                    best_val_metrics = val_metrics
                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        logger.info(f"!!! New Best Model: {best_val_f1:.3f} F1 Score - Saved to {save_path} !!!")
            else:
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                logger.info(f"Epoch {epoch+1}/{num_epochs} Finished - Duration: {epoch_duration:.2f}s")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
                            f"P: {train_metrics['precision']:.3f}, R: {train_metrics['recall']:.3f}, "
                            f"F1: {train_metrics['f1']:.3f}, AUC: {train_metrics['auc']:.3f}")
                # Save model at end of each epoch when no val_loader is available
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"Model saved to {save_path} (no validation set)")
            
            # Update learning rate
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
        total_duration = time.time() - start_time
        logger.info(f"Training completed in {total_duration/60:.2f} minutes.")
        if best_val_metrics and "best_threshold" in best_val_metrics:
            logger.info(
                "Recommended validation threshold after training: %.4f "
                "(P=%.3f, R=%.3f, F1=%.3f on best validation epoch)",
                best_val_metrics["best_threshold"],
                best_val_metrics["best_threshold_precision"],
                best_val_metrics["best_threshold_recall"],
                best_val_metrics["best_threshold_f1"],
            )
