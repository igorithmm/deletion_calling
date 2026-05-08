"""Inference for FusedDeepSV (image + HyenaDNA embedding).

Mirror of :class:`deepsv.inference.predictor.DeletionPredictor` but for the
two-modality fused model. Embeddings are read from the same HDF5 layout
produced by ``scripts/precompute_hyenadna_embeddings.py``: ``(chrom,
position)`` → ``embeddings[chrom][position // window_size]``.

The H5 file is opened on construction and held read-only for the predictor's
lifetime. Per-call lookups are O(1) array indexes; embeddings are cast from
``float16`` → ``float32`` before entering the model.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from ..data.fused_dataset import _read_embed_dim, _read_window_size, _resolve_chrom_key
from ..models.fused_cnn import FusedDeepSV

logger = logging.getLogger(__name__)


class FusedPredictor:
    """Inference handler for :class:`FusedDeepSV`.

    Args:
        model: A trained :class:`FusedDeepSV`.
        embeddings_h5: Path to the precomputed HyenaDNA HDF5 file.
        device: Inference device. Defaults to CUDA if available.
        threshold: Probability threshold for the positive class.
        preload_chroms: Optional list of chromosomes to load into RAM.
            If ``None``, embeddings are read lazily from disk per-call.
    """

    def __init__(
        self,
        model: FusedDeepSV,
        embeddings_h5: str,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        preload_chroms: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()
        self.threshold = threshold
        self.embeddings_h5 = str(embeddings_h5)

        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Open H5 to read attrs and (optionally) preload chrom arrays.
        self._h5: Optional[h5py.File] = h5py.File(self.embeddings_h5, "r")
        self.window_size = _read_window_size(self._h5)
        sample_key = next(iter(self._h5.keys()))
        self.embed_dim = _read_embed_dim(self._h5, sample_key)

        self._cache: Dict[str, np.ndarray] = {}
        self._chrom_key_map: Dict[str, str] = {}
        if preload_chroms:
            file_keys = list(self._h5.keys())
            for chrom in preload_chroms:
                key = _resolve_chrom_key(file_keys, chrom)
                self._chrom_key_map[chrom] = key
                self._cache[chrom] = self._h5[key][:]
                logger.info(
                    "Preloaded %s (key=%s) shape=%s", chrom, key, self._cache[chrom].shape
                )

    def close(self) -> None:
        """Release the HDF5 handle."""
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def _get_embedding(self, chrom: str, position: int) -> np.ndarray:
        window_idx = int(position) // self.window_size
        if chrom in self._cache:
            arr = self._cache[chrom]
            if window_idx < 0 or window_idx >= arr.shape[0]:
                raise IndexError(
                    f"window_idx={window_idx} out of range for {chrom} "
                    f"(n_windows={arr.shape[0]}); position={position}, "
                    f"window_size={self.window_size}."
                )
            return arr[window_idx].astype(np.float32, copy=False)
        if self._h5 is None:
            raise RuntimeError("FusedPredictor: HDF5 file is closed.")
        if chrom not in self._chrom_key_map:
            self._chrom_key_map[chrom] = _resolve_chrom_key(
                list(self._h5.keys()), chrom
            )
        key = self._chrom_key_map[chrom]
        n_windows = self._h5[key].shape[0]
        if window_idx < 0 or window_idx >= n_windows:
            raise IndexError(
                f"window_idx={window_idx} out of range for {chrom} "
                f"(n_windows={n_windows}); position={position}, "
                f"window_size={self.window_size}."
            )
        return self._h5[key][window_idx].astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, image_path: Path, chrom: str, position: int
    ) -> Tuple[float, int]:
        """Predict deletion probability for one ``(image, chrom, position)``.

        Returns:
            ``(prob_deletion, predicted_class)``.
        """
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        emb_np = self._get_embedding(chrom, position)
        emb_tensor = torch.from_numpy(emb_np).unsqueeze(0).to(self.device).float()

        outputs = self.model(image_tensor, emb_tensor)
        probs = torch.softmax(outputs, dim=1)
        prob_del = probs[0, 1].item()
        predicted = 1 if prob_del > self.threshold else 0
        return prob_del, predicted

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[Path],
        chroms: List[str],
        positions: List[int],
        batch_size: int = 64,
    ) -> List[Tuple[float, int]]:
        """Batched prediction over aligned lists of paths / chroms / positions."""
        n = len(image_paths)
        if not (len(chroms) == len(positions) == n):
            raise ValueError(
                "image_paths, chroms and positions must be the same length; "
                f"got {n}, {len(chroms)}, {len(positions)}."
            )

        # Pre-allocate one slot per input. Failed items keep the (0.0, 0)
        # sentinel; successes are filled in after the batched forward pass.
        # This preserves index alignment with the input lists.
        results: List[Tuple[float, int]] = [(0.0, 0)] * n

        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            valid_imgs: List[torch.Tensor] = []
            valid_embs: List[np.ndarray] = []
            valid_indices: List[int] = []

            for i in range(start, end):
                try:
                    img = Image.open(image_paths[i]).convert("RGB")
                    valid_imgs.append(self.transform(img))
                    valid_embs.append(self._get_embedding(chroms[i], positions[i]))
                    valid_indices.append(i)
                except Exception as e:
                    logger.error(
                        "Error preparing sample %s (chrom=%s pos=%s): %s",
                        image_paths[i], chroms[i], positions[i], e,
                    )
                    # results[i] stays (0.0, 0) by initialisation.

            if not valid_indices:
                continue

            img_batch = torch.stack(valid_imgs).to(self.device)
            emb_batch = torch.from_numpy(np.stack(valid_embs)).to(self.device).float()

            outputs = self.model(img_batch, emb_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().tolist()
            for idx, prob in zip(valid_indices, probs):
                results[idx] = (prob, 1 if prob > self.threshold else 0)

        return results
