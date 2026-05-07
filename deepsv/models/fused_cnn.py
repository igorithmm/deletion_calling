"""FiLM-conditioned fusion of DeepSV image CNN with frozen sequence embeddings.

:class:`FusedDeepSV` — the main contribution. Wraps the existing
:class:`deepsv.models.cnn.DeletionCNN` and injects FiLM modulation
(γ/β predicted from a 256-dim HyenaDNA embedding) after blocks 2 and 3
of the CNN. The first block is too low-level and the fourth is too late
(spatial info already aggregated), so they are not modulated.

The fused model is initialised so that at step 0 it is a *bit-identical
identity wrapper* around the underlying CNN: the FiLM generators are
zero-initialised, so γ = β = 0 and the modulation reduces to the identity
``F_out = F * (1 + 0) + 0 = F``. Any deviation from baseline is therefore
purely learned.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .cnn import DeletionCNN
from .film import FiLMGenerator, apply_film


# Channel counts at the FiLM injection points in DeletionCNN. These match
# the conv2 / conv3 output channel definitions in cnn.py and must be kept
# in sync if the backbone is ever re-shaped.
_BLOCK2_CHANNELS = 256
_BLOCK3_CHANNELS = 384


class FusedDeepSV(nn.Module):
    """DeepSV CNN with FiLM modulation conditioned on a sequence embedding.

    The wrapped backbone is :class:`DeletionCNN`. Two FiLM generators map a
    256-dim embedding to ``(γ_2, β_2)`` for the block-2 feature map and
    ``(γ_3, β_3)`` for the block-3 feature map.

    Forward signature is ``(image, embedding) → logits``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 2,
        input_channels: int = 3,
        film_hidden_dim: int = 128,
        backbone: Optional[DeletionCNN] = None,
    ) -> None:
        """Args:
            embed_dim: Sequence embedding dimensionality (default 256 for
                HyenaDNA-small-32k).
            num_classes: Number of classifier outputs.
            input_channels: Number of image channels (3 for RGB pileups).
            film_hidden_dim: Hidden width of each FiLM generator MLP.
            backbone: Optionally inject a pre-existing :class:`DeletionCNN`
                instance (e.g. for transfer-learning from a saved M0
                checkpoint). If ``None``, a fresh CNN is constructed.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.cnn = backbone if backbone is not None else DeletionCNN(
            num_classes=num_classes, input_channels=input_channels
        )

        self.film2 = FiLMGenerator(
            embed_dim=embed_dim,
            num_channels=_BLOCK2_CHANNELS,
            hidden_dim=film_hidden_dim,
        )
        self.film3 = FiLMGenerator(
            embed_dim=embed_dim,
            num_channels=_BLOCK3_CHANNELS,
            hidden_dim=film_hidden_dim,
        )

    # ------------------------------------------------------------------
    # Parameter groups for two-stage training
    # ------------------------------------------------------------------

    def cnn_parameters(self):
        """Iterator over backbone parameters (used for the CNN param group)."""
        return self.cnn.parameters()

    def film_parameters(self):
        """Iterator over both FiLM generators' parameters."""
        for p in self.film2.parameters():
            yield p
        for p in self.film3.parameters():
            yield p

    def freeze_backbone(self) -> None:
        """Freeze the CNN backbone (Stage A of two-stage training)."""
        for p in self.cnn.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze the CNN backbone (Stage B of two-stage training)."""
        for p in self.cnn.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, image: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """Forward pass with FiLM injection after blocks 2 and 3.

        Args:
            image: Tensor of shape ``(B, C, H, W)``.
            embedding: Tensor of shape ``(B, embed_dim)``, already cast to the
                model's dtype/device by the dataloader / caller.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        gamma2, beta2 = self.film2(embedding)
        gamma3, beta3 = self.film3(embedding)

        def hook2(feat: torch.Tensor) -> torch.Tensor:
            return apply_film(feat, gamma2, beta2)

        def hook3(feat: torch.Tensor) -> torch.Tensor:
            return apply_film(feat, gamma3, beta3)

        return self.cnn._forward_features_with_hooks(image, hook2=hook2, hook3=hook3)

