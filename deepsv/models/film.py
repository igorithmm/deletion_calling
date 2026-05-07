"""FiLM (Feature-wise Linear Modulation) building blocks for DeepSV.

A FiLM generator maps a conditioning vector ``z`` (here, a HyenaDNA embedding)
to per-channel scale (``γ``) and shift (``β``) parameters, which are then
applied to a 2-D feature map ``F`` of shape ``(B, C, H, W)`` via::

    F_out = F * (1 + γ.view(B, C, 1, 1)) + β.view(B, C, 1, 1)

The ``(1 + γ)`` parameterisation lets us zero-initialise the final linear layer
of every generator: at step 0 both γ and β are zero, so the modulation reduces
to the identity ``F_out = F``. Training therefore *starts* from baseline
DeepSV behaviour and only deviates as the FiLM heads learn.

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning
Layer", AAAI 2018.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Two-layer MLP producing per-channel ``(γ, β)`` from a condition vector.

    Architecture::

        Linear(embed_dim → hidden_dim) → ReLU → Linear(hidden_dim → 2 * num_channels)

    The output is split along the last dim into ``γ`` and ``β``, each of
    shape ``(B, num_channels)``.

    The final ``Linear`` is zero-initialised (both weight and bias). This makes
    the generator output exactly zero at init, so :func:`apply_film` is the
    identity map at step 0.
    """

    def __init__(
        self,
        embed_dim: int,
        num_channels: int,
        hidden_dim: int = 128,
    ) -> None:
        """Args:
            embed_dim: Dimensionality of the conditioning vector (e.g. 256
                for HyenaDNA-small-32k embeddings).
            num_channels: Number of feature-map channels to modulate (``C_k``).
            hidden_dim: Width of the MLP hidden layer (default 128).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2 * num_channels)
        self.activation = nn.ReLU(inplace=True)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Standard init for fc1; **zero** init for fc2 (identity at step 0)."""
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map a batch of condition vectors to ``(γ, β)``.

        Args:
            z: Tensor of shape ``(B, embed_dim)``.

        Returns:
            A tuple ``(γ, β)``, each of shape ``(B, num_channels)``.
        """
        h = self.activation(self.fc1(z))
        out = self.fc2(h)  # (B, 2 * C)
        gamma, beta = out.chunk(2, dim=-1)  # each (B, C)
        return gamma, beta


def apply_film(
    feature_map: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply FiLM modulation to a 4-D feature map.

    Args:
        feature_map: Tensor of shape ``(B, C, H, W)``.
        gamma: Per-channel scale offsets of shape ``(B, C)``.
        beta: Per-channel shifts of shape ``(B, C)``.

    Returns:
        ``feature_map * (1 + γ) + β`` broadcast across spatial dims, shape
        ``(B, C, H, W)``.
    """
    if feature_map.dim() != 4:
        raise ValueError(
            f"feature_map must be 4-D (B, C, H, W); got shape {tuple(feature_map.shape)}"
        )
    if gamma.shape != beta.shape:
        raise ValueError(
            f"gamma and beta must share shape; got {tuple(gamma.shape)} vs {tuple(beta.shape)}"
        )
    if gamma.dim() != 2 or gamma.shape[0] != feature_map.shape[0] or gamma.shape[1] != feature_map.shape[1]:
        raise ValueError(
            f"gamma/beta must be (B, C)=({feature_map.shape[0]}, {feature_map.shape[1]}); "
            f"got {tuple(gamma.shape)}"
        )

    gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
    beta = beta.unsqueeze(-1).unsqueeze(-1)
    return feature_map * (1.0 + gamma) + beta
