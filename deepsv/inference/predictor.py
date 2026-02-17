"""Prediction and inference utilities"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Optional
import logging
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class DeletionPredictor:
    """Handles model inference for deletion detection"""
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 threshold: float = 0.5):
        """
        Initialize predictor
        
        Args:
            model: Trained PyTorch model
            device: Device for inference
            threshold: Classification threshold
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_image(self, image_path: Path) -> Tuple[float, int]:
        """
        Predict deletion for a single image
        
        Args:
            image_path: Path to image file
            
        Returns:
            Tuple of (probability, predicted_class)
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prob_deletion = probabilities[0][1].item()
            predicted = 1 if prob_deletion > self.threshold else 0
        
        return prob_deletion, predicted
    
    def predict_batch(self, image_paths: List[Path]) -> List[Tuple[float, int]]:
        """
        Predict deletions for a batch of images
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of (probability, predicted_class) tuples
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {e}")
                results.append((0.0, 0))
        
        return results

