"""PyTorch CNN model for deletion detection"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeletionCNN(nn.Module):
    """
    Convolutional Neural Network for detecting deletions from sequence read images
    
    Architecture inspired by AlexNet-style CNN with modifications for
    deletion detection task.
    """
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        """
        Initialize the CNN model
        
        Args:
            num_classes: Number of output classes (default: 2 for deletion/non-deletion)
            input_channels: Number of input channels (default: 3 for RGB)
        """
        super(DeletionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, 512)  # Adjusted based on input size
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    # ------------------------------------------------------------------
    # Named stages
    #
    # The forward pass is split into per-block helpers so that external
    # wrappers (e.g. FusedDeepSV) can inject FiLM modulation between the
    # named blocks without copying the architecture. The math executed by
    # `forward(x)` is unchanged: it is `_forward_features_with_hooks(x)`
    # with no hooks, which composes the helpers in the same order as the
    # original monolithic implementation.
    # ------------------------------------------------------------------

    def _block1(self, x: torch.Tensor) -> torch.Tensor:
        """Conv1 → ReLU → Pool1."""
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        return x

    def _block2(self, x: torch.Tensor) -> torch.Tensor:
        """Conv2 → ReLU → Pool2. Output channels: 256."""
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        return x

    def _block3(self, x: torch.Tensor) -> torch.Tensor:
        """Conv3 → ReLU. Output channels: 384."""
        x = F.relu(self.conv3(x))
        return x

    def _block4(self, x: torch.Tensor) -> torch.Tensor:
        """Conv4 → ReLU → Pool4."""
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        return x

    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten → fc1 → ReLU → drop → fc2 → ReLU → drop → fc3."""
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

    def _forward_features_with_hooks(
        self,
        x: torch.Tensor,
        hook2=None,
        hook3=None,
    ) -> torch.Tensor:
        """Forward pass with optional callables applied to block-2 / block-3 outputs.

        ``hook2`` (if provided) is called as ``hook2(feat) -> Tensor`` on the
        output of :meth:`_block2`; similarly for ``hook3``. When both hooks
        are ``None`` the math is byte-identical to the original
        :meth:`forward` implementation.

        Args:
            x: Input tensor ``(B, C, H, W)``.
            hook2: Optional callable applied after block 2.
            hook3: Optional callable applied after block 3.

        Returns:
            Output logits.
        """
        x = self._block1(x)

        x = self._block2(x)
        if hook2 is not None:
            x = hook2(x)

        x = self._block3(x)
        if hook3 is not None:
            x = hook3(x)

        x = self._block4(x)
        x = self._classify(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits
        """
        return self._forward_features_with_hooks(x)


class ModernDeletionCNN(nn.Module):
    """
    Modern CNN architecture with batch normalization and improved design.
    Optimized for efficiency: uses stride to reduce spatial dimensions early
    and scales channels progressively.
    """
    
    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super(ModernDeletionCNN, self).__init__()
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: Initial reduction (256x256 -> 64x64)
            # Conv1: Stride 2 reduces to 128x128
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            # Pool: Reduces to 64x64
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5: 8x8 -> 8x8 (Feature refinement)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

