"""PyTorch CNN model for deletion detection"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernDeletionCNN(nn.Module):
    """Modern CNN architecture with batch normalization, LeakyReLU, and
    progressive channel scaling.

    The forward pass is decomposed into five named
    blocks so that :class:`~deepsv.models.fused_cnn.FusedDeepSV` (or any
    other wrapper) can inject FiLM modulation between them via the
    :meth:`_forward_features_with_hooks` method.

    FiLM injection points:

    * **hook3** — after block 3 (128 channels, 16×16). Mid-level features
      that benefit from sequence-aware rescaling.
    * **hook4** — after block 4 (256 channels, 8×8). High-level features
      before the refinement stage.

    Blocks 1–2 are too low-level (edges/textures), and block 5 is a
    refinement layer with dropout, so they are not modulated.

    Channel counts at hook sites (importable by ``fused_cnn.py``):

    * ``MODERN_BLOCK3_CHANNELS = 128``
    * ``MODERN_BLOCK4_CHANNELS = 256``
    """

    # Expose channel counts so FusedDeepSV can read them without magic
    # numbers.  Kept as class attributes for easy import.
    BLOCK3_CHANNELS: int = 128
    BLOCK4_CHANNELS: int = 256

    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super(ModernDeletionCNN, self).__init__()

        # ── Block 1: 256×256 → 64×64  (channels: input → 32) ──
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 2: 64×64 → 32×32  (channels: 32 → 64) ──
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 3: 32×32 → 16×16  (channels: 64 → 128) ── [hook3]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 4: 16×16 → 8×8  (channels: 128 → 256) ── [hook4]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # ── Block 5: 8×8 → 8×8  (channels: 256 → 256, refinement) ──
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.drop5 = nn.Dropout(0.3)

        # ── Classifier ──
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout_fc = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, num_classes)

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

    # ------------------------------------------------------------------
    # Named stages for FiLM compatibility
    # ------------------------------------------------------------------

    def _block1(self, x: torch.Tensor) -> torch.Tensor:
        """Conv1 → BN → LeakyReLU → Pool.  Output: 32×64×64."""
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)
        x = self.pool1(x)
        return x

    def _block2(self, x: torch.Tensor) -> torch.Tensor:
        """Conv2 → BN → LeakyReLU → Pool.  Output: 64×32×32."""
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = self.pool2(x)
        return x

    def _block3(self, x: torch.Tensor) -> torch.Tensor:
        """Conv3 → BN → LeakyReLU → Pool.  Output: 128×16×16."""
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1)
        x = self.pool3(x)
        return x

    def _block4(self, x: torch.Tensor) -> torch.Tensor:
        """Conv4 → BN → LeakyReLU → Pool.  Output: 256×8×8."""
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1)
        x = self.pool4(x)
        return x

    def _block5(self, x: torch.Tensor) -> torch.Tensor:
        """Conv5 → BN → LeakyReLU → Dropout.  Output: 256×8×8 (refinement)."""
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
        x = self.drop5(x)
        return x

    def _classify(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten → FC1 → ReLU → Dropout → FC2."""
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
        """Forward pass with optional FiLM hooks after blocks 3 and 4.

        When both hooks are ``None`` the result is identical to
        :meth:`forward`.

        Args:
            x: Input tensor ``(B, C, H, W)``.
            hook3: Optional callable ``(feat) → Tensor`` applied after block 3
                (128 channels).
            hook4: Optional callable ``(feat) → Tensor`` applied after block 4
                (256 channels).

        Returns:
            Output logits.
        """
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
