"""Multichannel CNN for tensor-based SV detection (DeepSV3).

This module provides the BroadcastContextCNN architecture, which processes
a (C, H, W) tensor where:
  - C = 13 alignment channels + K genomic context channels
  - H = max reads (padded)
  - W = genomic window width (e.g. 50 bp)

The K context channels are constant along the H and W axes (broadcast from
a K-dimensional DNABERT-2 embedding).
"""
import torch
import torch.nn as nn

from deepsv.data.bam_handler import NUM_ALIGNMENT_CHANNELS


class BroadcastContextCNN(nn.Module):
    """2D CNN that processes alignment features + broadcast genomic context.

    Architecture:
        Conv2d blocks with BatchNorm + LeakyReLU, progressive channel
        scaling (32 → 64 → 128 → 256), adaptive average pooling,
        then a classifier head.

    The model uses AdaptiveAvgPool2d so it can handle variable H (read
    depth) without requiring a fixed spatial size.
    """

    def __init__(
        self,
        num_classes: int = 2,
        context_channels: int = 8,
        alignment_channels: int = NUM_ALIGNMENT_CHANNELS,
    ):
        """
        Args:
            num_classes: Output classes (default 2: deletion / non-deletion).
            context_channels: Number of PCA-reduced DNABERT-2 channels (K).
            alignment_channels: Number of alignment feature channels (13).
        """
        super().__init__()

        input_channels = alignment_channels + context_channels
        self.input_channels = input_channels

        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1: (C, H, W) → (32, H/2, W/2)
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            # Block 2: → (64, H/4, W/4)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: → (128, H/8, W/8)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: → (256, H/16, W/16)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            # Adaptive pooling to fixed spatial size regardless of H
            nn.AdaptiveAvgPool2d((4, 4)),

            # Refinement block (at fixed 4×4)
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.3),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization for conv layers, normal for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W) tensor where C = alignment + context channels.

        Returns:
            logits of shape (batch, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
