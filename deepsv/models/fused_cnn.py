"""FiLM-conditioned fusion of DeepSV image CNN with frozen sequence embeddings.

:class:`FusedDeepSV` — the main contribution. Wraps a
:class:`~deepsv.models.cnn.ModernDeletionCNN` backbone and injects FiLM
modulation (γ/β predicted from a HyenaDNA embedding) at two mid-level
feature extraction points: after block 3 (96 ch) and block 4 (96 ch),
using ``hook3``/``hook4`` kwargs.

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

from .cnn import ModernDeletionCNN
from .film import FiLMGenerator, apply_film


# Channel counts at the FiLM injection points.  These must match the
# conv output channel definitions in cnn.py.
_HOOK_A_CHANNELS = ModernDeletionCNN.BLOCK3_CHANNELS  # 96
_HOOK_B_CHANNELS = ModernDeletionCNN.BLOCK4_CHANNELS  # 96


class FusedDeepSV(nn.Module):
    """DeepSV CNN with FiLM modulation conditioned on a sequence embedding.

    Two FiLM generators map a conditioning embedding to ``(γ_a, β_a)`` and
    ``(γ_b, β_b)`` which are injected after block 3 and block 4 of the
    :class:`ModernDeletionCNN` backbone (``hook3`` / ``hook4``).

    Forward signature is ``(image, embedding) → logits``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 2,
        input_channels: int = 3,
        film_hidden_dim: int = 128,
        film_dropout_rate: float = 0.1,
        backbone: Optional[ModernDeletionCNN] = None,
    ) -> None:
        """Args:
        embed_dim: Sequence embedding dimensionality (default 256 for
            HyenaDNA-small-32k).
        num_classes: Number of classifier outputs.
        input_channels: Number of image channels (3 for RGB pileups).
        film_hidden_dim: Hidden width of each FiLM generator MLP.
        film_dropout_rate: Dropout probability inside each FiLM generator.
        backbone: Optionally inject a pre-existing
            :class:`ModernDeletionCNN` instance (e.g. for
            transfer-learning from a saved M0 checkpoint).
            If ``None``, a new :class:`ModernDeletionCNN` is constructed.
        """
        super().__init__()

        self.embed_dim = embed_dim

        if backbone is not None:
            self.cnn = backbone
        else:
            self.cnn = ModernDeletionCNN(
                num_classes=num_classes,
                input_channels=input_channels,
            )

        # FiLM channel counts and hook kwarg names for ModernDeletionCNN.
        self._hook_a_kwarg = "hook3"
        self._hook_b_kwarg = "hook4"

        self.film_a = FiLMGenerator(
            embed_dim=embed_dim,
            num_channels=_HOOK_A_CHANNELS,
            hidden_dim=film_hidden_dim,
            dropout_rate=film_dropout_rate,
        )
        self.film_b = FiLMGenerator(
            embed_dim=embed_dim,
            num_channels=_HOOK_B_CHANNELS,
            hidden_dim=film_hidden_dim,
            dropout_rate=film_dropout_rate,
        )

    # ------------------------------------------------------------------
    # Parameter groups for two-stage training
    # ------------------------------------------------------------------

    def cnn_parameters(self):
        """Iterator over backbone parameters (used for the CNN param group)."""
        return self.cnn.parameters()

    def film_parameters(self):
        """Iterator over both FiLM generators' parameters."""
        for p in self.film_a.parameters():
            yield p
        for p in self.film_b.parameters():
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
        """Forward pass with FiLM injection at two mid-level backbone stages.

        Args:
            image: Tensor of shape ``(B, C, H, W)``.
            embedding: Tensor of shape ``(B, embed_dim)``, already cast to the
                model's dtype/device by the dataloader / caller.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        gamma_a, beta_a = self.film_a(embedding)
        gamma_b, beta_b = self.film_b(embedding)

        def hook_a(feat: torch.Tensor) -> torch.Tensor:
            return apply_film(feat, gamma_a, beta_a)

        def hook_b(feat: torch.Tensor) -> torch.Tensor:
            return apply_film(feat, gamma_b, beta_b)

        return self.cnn._forward_features_with_hooks(
            image,
            **{self._hook_a_kwarg: hook_a, self._hook_b_kwarg: hook_b},
        )
