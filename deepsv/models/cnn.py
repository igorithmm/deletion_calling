"""PyTorch CNN model for deletion detection"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernDeletionCNN(nn.Module):
    """Modern CNN architecture matching the legacy Keras model.

    The forward pass is decomposed into five named blocks so that
    wrapper models can inject FiLM modulation or hooks.
    """

    BLOCK3_CHANNELS: int = 96
    BLOCK4_CHANNELS: int = 96

    def __init__(self, num_classes: int = 2, input_channels: int = 3, n_base_filters: int = 96, dropout_rate: float = 0.3):
        super(ModernDeletionCNN, self).__init__()

        # ── Block 1 ──
        self.conv1_1 = nn.Conv2d(input_channels, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1_1 = nn.BatchNorm2d(n_base_filters)
        self.conv1_2 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1_2 = nn.BatchNorm2d(n_base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # ── Block 2 ──
        self.conv2_1 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(n_base_filters)
        self.conv2_2 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(n_base_filters)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # ── Block 3 ──
        self.conv3_1 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(n_base_filters)
        self.conv3_2 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(n_base_filters)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # ── Block 4 ──
        self.conv4_1 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(n_base_filters)
        self.conv4_2 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(n_base_filters)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        # ── Block 5 ──
        self.conv5 = nn.Conv2d(n_base_filters, n_base_filters, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(n_base_filters)
        self.drop5 = nn.Dropout(dropout_rate)

        # ── Classifier ──
        self.flatten = nn.Flatten()
        # For 255x255 or 256x256 input, 4 pools of 2x2 reduce spatial dimension to 16x16
        self.fc1 = nn.Linear(n_base_filters * 16 * 16, 256)
        self.dropout_fc = nn.Dropout(dropout_rate)
        # Note: Added final linear layer for class logits (old Keras code ended with Dropout)
        self.fc2 = nn.Linear(256, num_classes)

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

    def _block1(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn1_1(self.conv1_1(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn1_2(self.conv1_2(x)), 0.1, inplace=True)
        x = self.pool1(x)
        return x

    def _block2(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn2_1(self.conv2_1(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2_2(self.conv2_2(x)), 0.1, inplace=True)
        x = self.pool2(x)
        return x

    def _block3(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn3_1(self.conv3_1(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3_2(self.conv3_2(x)), 0.1, inplace=True)
        x = self.pool3(x)
        return x

    def _block4(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn4_1(self.conv4_1(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn4_2(self.conv4_2(x)), 0.1, inplace=True)
        x = self.pool4(x)
        return x

    def _block5(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
        x = self.drop5(x)
        return x

    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x

    def _forward_features_with_hooks(
        self,
        x: torch.Tensor,
        hook3=None,
        hook4=None,
    ) -> torch.Tensor:
        x = self._block1(x)
        x = self._block2(x)

        x = self._block3(x)
        if hook3 is not None:
            x = hook3(x)

        x = self._block4(x)
        if hook4 is not None:
            x = hook4(x)

        x = self._block5(x)
        x = self._classify(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output logits
        """
        return self._forward_features_with_hooks(x)
