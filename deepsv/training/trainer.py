"""Training utilities for the CNN model"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Optional, Dict
import logging
from tqdm import tqdm

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
        correct = 0
        total = 0
        
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (total / labels.size(0)),
                'acc': 100 * correct / total
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / (total / labels.size(0)),
                    'acc': 100 * correct / total
                })
        
        val_loss = running_loss / len(dataloader)
        val_acc = 100 * correct / total
        
        return {'loss': val_loss, 'accuracy': val_acc}
    
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
        best_val_acc = 0.0
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
                
                logger.info(f"Epoch {epoch+1}/{num_epochs} Finished - "
                           f"Duration: {epoch_duration:.2f}s")
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}% | "
                           f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
                
                # Save best model
                if val_metrics['accuracy'] > best_val_acc and save_path:
                    best_val_acc = val_metrics['accuracy']
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"!!! New Best Model: {best_val_acc:.2f}% Accuracy - Saved to {save_path} !!!")
            else:
                epoch_end = time.time()
                epoch_duration = epoch_end - epoch_start
                logger.info(f"Epoch {epoch+1}/{num_epochs} Finished - "
                           f"Duration: {epoch_duration:.2f}s - "
                           f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.2f}%")
            
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

