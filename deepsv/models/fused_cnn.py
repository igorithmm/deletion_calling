"""Fusion of DeepSV image CNN with frozen sequence embeddings.

:class:`FusedDeepSV` — the main contribution. Wraps a
:class:`~deepsv.models.cnn.ModernDeletionCNN` backbone and can use two
embedding-fusion paths:

* FiLM modulation (gamma/beta predicted from a HyenaDNA embedding) at two
  mid-level feature extraction points: after block 3 and block 4.
* A late context-calibration path that directly shifts the positive deletion
  logit from sequence context and CNN classifier features.

The fused model is initialised so that at step 0 it is a *bit-identical
identity wrapper* around the underlying CNN: FiLM generators and context
calibration heads are zero-initialised, so any deviation from baseline is
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
_VALID_FUSION_MODES = ("film", "context", "film_context")


class _ZeroInitMLP(nn.Module):
    """Small residual MLP whose output is exactly zero at initialization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.fc1(x)))
        return self.fc2(x)


class FusedDeepSV(nn.Module):
    """DeepSV CNN with FiLM modulation conditioned on a sequence embedding.

    Depending on ``fusion_mode``, the model can apply FiLM, a late context
    deletion-logit calibrator, or both. The late path is designed for the
    common failure mode where the pileup image contains a deletion-like signal
    but the sequence context is a risky false-positive region.

    Forward signature is ``(image, embedding) → logits``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 2,
        input_channels: int = 3,
        film_hidden_dim: int = 128,
        film_dropout_rate: float = 0.1,
        fusion_mode: str = "film_context",
        context_hidden_dim: int = 128,
        context_dropout_rate: float = 0.1,
        backbone: Optional[ModernDeletionCNN] = None,
    ) -> None:
        """Args:
        embed_dim: Sequence embedding dimensionality (default 256 for
            HyenaDNA-small-32k).
        num_classes: Number of classifier outputs.
        input_channels: Number of image channels (3 for RGB pileups).
        film_hidden_dim: Hidden width of each FiLM generator MLP.
        film_dropout_rate: Dropout probability inside each FiLM generator.
        fusion_mode: One of ``"film"``, ``"context"``, or
            ``"film_context"``. ``"film_context"`` is the recommended default:
            it keeps the original FiLM pathway and adds direct context
            calibration of the deletion logit.
        context_hidden_dim: Hidden width for the late context-calibration MLPs.
        context_dropout_rate: Dropout probability in context-calibration MLPs.
        backbone: Optionally inject a pre-existing
            :class:`ModernDeletionCNN` instance (e.g. for
            transfer-learning from a saved M0 checkpoint).
            If ``None``, a new :class:`ModernDeletionCNN` is constructed.
        """
        super().__init__()

        if fusion_mode not in _VALID_FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {_VALID_FUSION_MODES}; got {fusion_mode!r}"
            )

        self.embed_dim = embed_dim
        self.fusion_mode = fusion_mode
        self.use_film = fusion_mode in ("film", "film_context")
        self.use_context = fusion_mode in ("context", "film_context")
        self.positive_class_idx = 1

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

        if self.use_film:
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
        else:
            self.film_a = None
            self.film_b = None

        if self.use_context:
            self.context_norm = nn.LayerNorm(embed_dim)
            self.context_projector = nn.Sequential(
                nn.Linear(embed_dim, context_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(context_dropout_rate),
            )
            self.context_delta = _ZeroInitMLP(
                input_dim=embed_dim,
                hidden_dim=context_hidden_dim,
                output_dim=1,
                dropout_rate=context_dropout_rate,
            )
            self.joint_delta = _ZeroInitMLP(
                input_dim=ModernDeletionCNN.CLASSIFIER_FEATURES + context_hidden_dim,
                hidden_dim=context_hidden_dim,
                output_dim=1,
                dropout_rate=context_dropout_rate,
            )
        else:
            self.context_norm = None
            self.context_projector = None
            self.context_delta = None
            self.joint_delta = None

    # ------------------------------------------------------------------
    # Parameter groups for two-stage training
    # ------------------------------------------------------------------

    def cnn_parameters(self):
        """Iterator over backbone parameters (used for the CNN param group)."""
        return self.cnn.parameters()

    def film_parameters(self):
        """Backward-compatible alias for all non-CNN fusion parameters."""
        yield from self.fusion_parameters()

    def fusion_parameters(self):
        """Iterator over FiLM and late context-calibration parameters."""
        modules = (
            self.film_a,
            self.film_b,
            self.context_norm,
            self.context_projector,
            self.context_delta,
            self.joint_delta,
        )
        for module in modules:
            if module is None:
                continue
            for p in module.parameters():
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
        """Forward pass with the configured embedding-fusion pathways.

        Args:
            image: Tensor of shape ``(B, C, H, W)``.
            embedding: Tensor of shape ``(B, embed_dim)``, already cast to the
                model's dtype/device by the dataloader / caller.

        Returns:
            Logits of shape ``(B, num_classes)``.
        """
        hook_kwargs = {}
        if self.use_film:
            gamma_a, beta_a = self.film_a(embedding)
            gamma_b, beta_b = self.film_b(embedding)

            def hook_a(feat: torch.Tensor) -> torch.Tensor:
                return apply_film(feat, gamma_a, beta_a)

            def hook_b(feat: torch.Tensor) -> torch.Tensor:
                return apply_film(feat, gamma_b, beta_b)

            hook_kwargs = {
                self._hook_a_kwarg: hook_a,
                self._hook_b_kwarg: hook_b,
            }

        if not self.use_context:
            return self.cnn._forward_features_with_hooks(image, **hook_kwargs)

        logits, cnn_features = self.cnn._forward_features_with_hooks(
            image,
            return_classifier_features=True,
            **hook_kwargs,
        )

        context_embedding = self.context_norm(embedding)
        context_features = self.context_projector(context_embedding)
        delta = self.context_delta(context_embedding) + self.joint_delta(
            torch.cat((cnn_features, context_features), dim=-1)
        )

        logits = logits.clone()
        logits[:, self.positive_class_idx] = (
            logits[:, self.positive_class_idx] + delta.squeeze(-1)
        )
        return logits
