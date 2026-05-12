"""Sequence-only deletion prior model.

The model consumes one reference-embedding vector, or a short window of
neighbouring reference-embedding vectors, and predicts whether the genomic
window is in a region with known population deletion evidence.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceDeletionPrior(nn.Module):
    """Small sequence model for reference-genome deletion priors.

    Args:
        embed_dim: Dimensionality of each H5 embedding vector.
        hidden_dim: Internal feature width.
        num_layers: Number of residual 1-D convolution blocks over the
            embedding-window axis.
        dropout_rate: Dropout probability.
        num_classes: Output classes. Defaults to 2 for CrossEntropyLoss.

    Input shapes:
        ``(B, embed_dim)`` or ``(B, T, embed_dim)``.

    Output:
        Logits of shape ``(B, num_classes)``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0; got {num_layers}")

        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout_rate)
        self.num_classes = int(num_classes)

        self.input_norm = nn.LayerNorm(self.embed_dim)
        self.input_proj = nn.Linear(self.embed_dim, self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

        blocks = []
        for _ in range(self.num_layers):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.hidden_dim,
                        self.hidden_dim,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.GELU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Conv1d(
                        self.hidden_dim,
                        self.hidden_dim,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.Dropout(self.dropout_rate),
                )
            )
        self.conv_blocks = nn.ModuleList(blocks)

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def config(self) -> Dict[str, object]:
        """Return constructor config for checkpoint metadata."""
        return {
            "embed_dim": self.embed_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3:
            raise ValueError(
                "SequenceDeletionPrior expects (B, D) or (B, T, D); "
                f"got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected embed_dim={self.embed_dim}; got input shape {tuple(x.shape)}"
            )

        x = self.input_norm(x)
        x = self.dropout(F.gelu(self.input_proj(x)))

        # Conv1d works over (B, C, T).
        x = x.transpose(1, 2)
        for block in self.conv_blocks:
            x = x + block(x)
        x = x.transpose(1, 2)

        avg_pool = x.mean(dim=1)
        max_pool = x.amax(dim=1)
        pooled = torch.cat((avg_pool, max_pool), dim=-1)
        return self.classifier(pooled)
