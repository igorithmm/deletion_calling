"""Training utilities for the CNN model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, Dict
import logging
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, accuracy_score
import numpy as np

logger = logging.getLogger(__name__)


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
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: DataLoader for training data
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        running_loss = 0.0
        
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
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            probs = torch.softmax(outputs.data, dim=1)[:, 1]
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (len(all_labels) / labels.size(0))
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
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            dataloader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        running_loss = 0.0
        
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
                probs = torch.softmax(outputs.data, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (len(all_labels) / labels.size(0))
                })
        
        val_loss = running_loss / len(dataloader)
        acc = accuracy_score(all_labels, all_preds) * 100
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.5
        
        return {
            'loss': val_loss, 'accuracy': acc, 
            'precision': precision, 'recall': recall, 
            'f1': f1, 'auc': auc
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10,
             save_path: Optional[Path] = None):
        """
        Train the model
        
        Args:
            train_loader: DataLoader for training
            val_loader: Optional DataLoader for validation
            num_epochs: Number of training epochs
            save_path: Path to save best model
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
        
        best_val_f1 = 0.0
        start_time = time.time()
        
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}...")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            logger.info(f"--- Epoch {epoch+1}/{num_epochs} Started ---")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            if val_loader:
                val_metrics = self.validate(val_loader)
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} Finished - Duration: {epoch_duration:.2f}s")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
                            f"P: {train_metrics['precision']:.3f}, R: {train_metrics['recall']:.3f}, "
                            f"F1: {train_metrics['f1']:.3f}, AUC: {train_metrics['auc']:.3f}")
                logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%, "
                            f"P: {val_metrics['precision']:.3f}, R: {val_metrics['recall']:.3f}, "
                            f"F1: {val_metrics['f1']:.3f}, AUC: {val_metrics['auc']:.3f}")
                
                # Save best model based on F1 instead of accuracy
                if val_metrics['f1'] > best_val_f1 and save_path:
                    best_val_f1 = val_metrics['f1']
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"!!! New Best Model: {best_val_f1:.3f} F1 Score - Saved to {save_path} !!!")
            else:
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                logger.info(f"Epoch {epoch+1}/{num_epochs} Finished - Duration: {epoch_duration:.2f}s")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%, "
                            f"P: {train_metrics['precision']:.3f}, R: {train_metrics['recall']:.3f}, "
                            f"F1: {train_metrics['f1']:.3f}, AUC: {train_metrics['auc']:.3f}")
            
            # Update learning rate
            if self.scheduler:
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                new_lr = self.optimizer.param_groups[0]['lr']
                if old_lr != new_lr:
                    logger.info(f"Learning rate changed from {old_lr:.6f} to {new_lr:.6f}")
        
        total_duration = time.time() - start_time
        logger.info(f"Training completed in {total_duration/60:.2f} minutes.")
    
    def save_model(self, path: Path):
        """Save model state"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load model state"""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        logger.info(f"Model loaded from {path}")

