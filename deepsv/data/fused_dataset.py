"""Image + HyenaDNA-embedding dataset for FiLM-conditioned DeepSV training.

Extends the path-based :class:`deepsv.training.trainer.ImageDataset` with a
second modality: a 256-dim HyenaDNA embedding looked up by ``(chrom,
window_idx)`` from a precomputed HDF5 file (see
``scripts/precompute_hyenadna_embeddings.py``).

Embedding lookup is::

    window_idx = position // window_size       # window_size from H5 attrs
    emb = embeddings[chrom][window_idx]        # (embed_dim,) float16

Per the project spec, the H5 file is opened read-only and entire chromosome
arrays for the requested chroms are pre-loaded into a RAM dict at dataset
init. The chr20–22 cache (~1.5 GB at 256 dim × 50 bp × ~150 Mb / chrom) fits
in T4 GPU host RAM and avoids per-batch HDF5 reads. Embeddings are cast
from ``float16`` → ``float32`` before being returned.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _resolve_chrom_key(h5_keys: Sequence[str], chrom: str) -> str:
    """Resolve a chromosome name to whatever key style the H5 file uses.

    Some pipelines write ``"chr1"`` keys, others write ``"1"``. Accept both
    on input and translate to whichever the file actually contains.
    """
    if chrom in h5_keys:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in h5_keys:
        return chrom[3:]
    if not chrom.startswith("chr") and f"chr{chrom}" in h5_keys:
        return f"chr{chrom}"
    raise KeyError(
        f"Chromosome {chrom!r} not found in HDF5 file. Available keys: "
        f"{sorted(h5_keys)}"
    )


def _read_window_size(h5_file: h5py.File) -> int:
    """Read the per-window bp size from H5 attrs.

    The spec example uses ``attrs/window_size`` while the actual producer
    (``precompute_hyenadna_embeddings.py``) uses ``window_bp`` at the
    file root. Accept both.
    """
    if "window_bp" in h5_file.attrs:
        return int(h5_file.attrs["window_bp"])
    if "window_size" in h5_file.attrs:
        return int(h5_file.attrs["window_size"])
    if "attrs" in h5_file and isinstance(h5_file["attrs"], h5py.Group):
        attrs_group = h5_file["attrs"]
        if "window_size" in attrs_group.attrs:
            return int(attrs_group.attrs["window_size"])
        if "window_bp" in attrs_group.attrs:
            return int(attrs_group.attrs["window_bp"])
    raise KeyError(
        "HDF5 embeddings file is missing 'window_bp' / 'window_size' attrs; "
        "cannot determine genomic window size."
    )


def _read_embed_dim(h5_file: h5py.File, sample_chrom_key: str) -> int:
    """Read embed_dim from attrs, falling back to the first dataset's shape."""
    if "embed_dim" in h5_file.attrs:
        return int(h5_file.attrs["embed_dim"])
    if "attrs" in h5_file and isinstance(h5_file["attrs"], h5py.Group):
        if "embed_dim" in h5_file["attrs"].attrs:
            return int(h5_file["attrs"].attrs["embed_dim"])
    return int(h5_file[sample_chrom_key].shape[1])


class FusedDataset(Dataset):
    """Dataset returning ``(image, embedding, label)`` triples.

    Args:
        image_paths: List of pileup-image file paths.
        labels: List of integer labels (0 = non-deletion, 1 = deletion).
        chroms: Per-sample chromosome name (e.g. ``"chr21"`` or ``"21"``).
        positions: Per-sample 0-based genomic position used to derive the
            window index. Should be the *centre* of the 50 bp window the
            image was rendered from.
        embeddings_h5: Path to the HDF5 file written by
            ``scripts/precompute_hyenadna_embeddings.py``.
        preload_chroms: Optional list of chromosomes to pre-load into RAM
            at init. If ``None``, the union of unique values in ``chroms``
            is used.
        transform: Optional torchvision-style image transform.
        embed_dtype: dtype of returned embeddings. Defaults to float32.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        chroms: List[str],
        positions: List[int],
        embeddings_h5: str,
        preload_chroms: Optional[Sequence[str]] = None,
        transform=None,
        embed_dtype: torch.dtype = torch.float32,
    ) -> None:
        n = len(image_paths)
        if not (len(labels) == len(chroms) == len(positions) == n):
            raise ValueError(
                "image_paths, labels, chroms and positions must all have the "
                f"same length; got {n}, {len(labels)}, {len(chroms)}, "
                f"{len(positions)}."
            )

        self.image_paths = image_paths
        self.labels = labels
        self.chroms = chroms
        self.positions = positions
        self.transform = transform
        self.embed_dtype = embed_dtype
        self.embeddings_h5 = str(embeddings_h5)

        chroms_to_load = (
            list(dict.fromkeys(preload_chroms))
            if preload_chroms is not None
            else list(dict.fromkeys(chroms))
        )
        if not chroms_to_load:
            raise ValueError(
                "FusedDataset requires at least one chromosome to load "
                "embeddings for; got an empty list."
            )

        logger.info(
            "Loading HyenaDNA embeddings from %s (chroms=%s) …",
            self.embeddings_h5,
            chroms_to_load,
        )

        self._embeddings: Dict[str, np.ndarray] = {}
        with h5py.File(self.embeddings_h5, "r") as f:
            self.window_size = _read_window_size(f)
            sample_key = _resolve_chrom_key(list(f.keys()), chroms_to_load[0])
            self.embed_dim = _read_embed_dim(f, sample_key)

            self._chrom_key_map: Dict[str, str] = {}
            file_keys = list(f.keys())
            for chrom in chroms_to_load:
                key = _resolve_chrom_key(file_keys, chrom)
                self._chrom_key_map[chrom] = key
                arr = f[key][:]
                self._embeddings[chrom] = arr
                logger.info(
                    "  %s → key=%s  shape=%s  dtype=%s  size=%.2f MB",
                    chrom,
                    key,
                    arr.shape,
                    arr.dtype,
                    arr.nbytes / 1e6,
                )

        unknown = [c for c in chroms if c not in self._embeddings]
        if unknown:
            raise ValueError(
                f"Sample chroms include {sorted(set(unknown))} but embeddings "
                f"are only loaded for {sorted(self._embeddings)}. Pass them "
                "via ``preload_chroms``."
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    # ------------------------------------------------------------------
    # Embedding lookup
    # ------------------------------------------------------------------

    def lookup_embedding(self, chrom: str, position: int) -> np.ndarray:
        """Fetch the embedding for ``(chrom, position)`` from the RAM cache.

        Args:
            chrom: Chromosome name as used at construction time.
            position: 0-based genomic position. The window index is
                ``position // window_size``.

        Returns:
            ``(embed_dim,)`` ``np.float32`` array.
        """
        if chrom not in self._embeddings:
            raise KeyError(
                f"No embeddings preloaded for {chrom!r}; available: "
                f"{sorted(self._embeddings)}"
            )
        arr = self._embeddings[chrom]
        window_idx = int(position) // self.window_size
        if window_idx < 0 or window_idx >= arr.shape[0]:
            raise IndexError(
                f"window_idx={window_idx} out of range for {chrom} "
                f"(n_windows={arr.shape[0]}); position={position}, "
                f"window_size={self.window_size}."
            )
        return arr[window_idx].astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        chrom = self.chroms[idx]
        position = self.positions[idx]
        emb_np = self.lookup_embedding(chrom, position)
        embedding = torch.from_numpy(emb_np).to(self.embed_dtype)

        return image, embedding, int(self.labels[idx])
