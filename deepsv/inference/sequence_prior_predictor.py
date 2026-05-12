"""Inference helper for sequence-only deletion-prior checkpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch

from ..data.fused_dataset import _read_embed_dim, _read_window_size, _resolve_chrom_key
from ..data.sequence_prior_dataset import chrom_sort_key, parse_chrom_list
from ..models.sequence_prior import SequenceDeletionPrior

logger = logging.getLogger(__name__)


def _load_checkpoint(path: str) -> Dict[str, object]:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Checkpoint {path!r} did not contain a dictionary payload.")


class SequencePriorPredictor:
    """Predict sequence-prior probability/logit for genomic positions."""

    def __init__(
        self,
        model: SequenceDeletionPrior,
        embeddings_h5: str,
        context_radius: int = 10,
        device: Optional[torch.device] = None,
        preload_chroms: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.embeddings_h5 = str(embeddings_h5)
        self.context_radius = int(context_radius)

        self._h5: Optional[h5py.File] = h5py.File(self.embeddings_h5, "r")
        self.window_size = _read_window_size(self._h5)
        sample_key = next(
            key
            for key in self._h5.keys()
            if isinstance(self._h5[key], h5py.Dataset) and len(self._h5[key].shape) == 2
        )
        self.embed_dim = _read_embed_dim(self._h5, sample_key)
        self._cache: Dict[str, np.ndarray] = {}
        self._chrom_key_map: Dict[str, str] = {}

        if preload_chroms:
            file_keys = list(self._h5.keys())
            for chrom in sorted(preload_chroms, key=chrom_sort_key):
                key = _resolve_chrom_key(file_keys, chrom)
                self._chrom_key_map[chrom] = key
                self._cache[chrom] = self._h5[key][:]
                logger.info(
                    "Preloaded sequence-prior embeddings %s (key=%s) shape=%s",
                    chrom,
                    key,
                    self._cache[chrom].shape,
                )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        embeddings_h5: str,
        context_radius: Optional[int] = None,
        device: Optional[torch.device] = None,
        preload_chroms: Optional[Sequence[str]] = None,
    ) -> "SequencePriorPredictor":
        payload = _load_checkpoint(checkpoint)
        config = payload.get("model_config", {})
        if not isinstance(config, dict):
            raise TypeError(f"Checkpoint {checkpoint!r} has invalid model_config.")
        model = SequenceDeletionPrior(**config)
        state = payload.get("state_dict", payload)
        if not isinstance(state, dict):
            raise TypeError(f"Checkpoint {checkpoint!r} has invalid state_dict.")
        model.load_state_dict(state)

        training_config = payload.get("training_config", {})
        if context_radius is None and isinstance(training_config, dict):
            context_radius = int(training_config.get("context_radius", 10))
        if context_radius is None:
            context_radius = 10

        return cls(
            model=model,
            embeddings_h5=embeddings_h5,
            context_radius=context_radius,
            device=device,
            preload_chroms=preload_chroms,
        )

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _array_for_chrom(self, chrom: str):
        if chrom in self._cache:
            return self._cache[chrom]
        if self._h5 is None:
            raise RuntimeError("SequencePriorPredictor: HDF5 file is closed.")
        if chrom not in self._chrom_key_map:
            self._chrom_key_map[chrom] = _resolve_chrom_key(list(self._h5.keys()), chrom)
        return self._h5[self._chrom_key_map[chrom]]

    def _embedding_window(self, chrom: str, position: int) -> np.ndarray:
        arr = self._array_for_chrom(chrom)
        n_windows = int(arr.shape[0])
        center = min(max(int(position) // self.window_size, 0), n_windows - 1)
        raw_start = center - self.context_radius
        raw_end = center + self.context_radius + 1
        start = max(0, raw_start)
        end = min(n_windows, raw_end)
        window = arr[start:end].astype(np.float32, copy=False)
        left_pad = max(0, -raw_start)
        right_pad = max(0, raw_end - n_windows)
        if left_pad:
            window = np.concatenate((np.repeat(window[:1], left_pad, axis=0), window), axis=0)
        if right_pad:
            window = np.concatenate((window, np.repeat(window[-1:], right_pad, axis=0)), axis=0)
        return window

    @torch.no_grad()
    def predict_batch(
        self,
        chroms: List[str],
        positions: List[int],
        batch_size: int = 1024,
    ) -> List[Tuple[float, float]]:
        """Return ``(prob_deletion_prior, logit_deletion_prior)`` per position."""
        if len(chroms) != len(positions):
            raise ValueError(
                "chroms and positions must have the same length; "
                f"got {len(chroms)} vs {len(positions)}"
            )

        results: List[Tuple[float, float]] = [(0.0, 0.0)] * len(chroms)
        for start in range(0, len(chroms), batch_size):
            end = min(len(chroms), start + batch_size)
            windows = [
                self._embedding_window(chroms[i], int(positions[i]))
                for i in range(start, end)
            ]
            batch = torch.from_numpy(np.stack(windows)).to(self.device).float()
            logits = self.model(batch)
            logit_delta = logits[:, 1] - logits[:, 0]
            probs = torch.sigmoid(logit_delta)
            for offset, (prob, logit) in enumerate(zip(probs.cpu(), logit_delta.cpu())):
                results[start + offset] = (float(prob.item()), float(logit.item()))
        return results


def load_sequence_prior_predictor(
    checkpoint: str,
    embeddings_h5: str,
    context_radius: Optional[int] = None,
    device: Optional[torch.device] = None,
    preload_chroms: Optional[object] = None,
) -> SequencePriorPredictor:
    parsed_preload = parse_chrom_list(preload_chroms)
    return SequencePriorPredictor.from_checkpoint(
        checkpoint=checkpoint,
        embeddings_h5=embeddings_h5,
        context_radius=context_radius,
        device=device,
        preload_chroms=parsed_preload,
    )
