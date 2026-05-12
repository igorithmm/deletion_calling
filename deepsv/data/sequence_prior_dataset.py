"""Datasets and samplers for training sequence-only deletion priors."""

from __future__ import annotations

import logging
import math
import random
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .fused_dataset import _read_embed_dim, _read_window_size, _resolve_chrom_key
from .vcf_handler import Variant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriorSample:
    """One labelled H5 embedding-window sample."""

    chrom: str
    window_idx: int
    label: int


def canonical_chrom(chrom: str) -> str:
    """Normalize only for matching; preserves original names elsewhere."""
    text = str(chrom)
    return text[3:] if text.startswith("chr") else text


def chrom_sort_key(chrom: str) -> Tuple[int, object]:
    name = canonical_chrom(chrom)
    if name.isdigit():
        return (0, int(name))
    order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    return (0, order[name]) if name in order else (1, name)


def parse_chrom_list(value: Optional[object]) -> Optional[List[str]]:
    """Parse comma-separated chromosomes or pass through a sequence."""
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        parts = [str(p).strip() for p in value]  # type: ignore[arg-type]
    return [p for p in parts if p]


def list_h5_embedding_chroms(embeddings_h5: str) -> List[str]:
    """Return H5 dataset keys that look like chromosome embedding matrices."""
    with h5py.File(embeddings_h5, "r") as f:
        chroms = [
            key
            for key in f.keys()
            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 2
        ]
    return sorted(chroms, key=chrom_sort_key)


def _selected_h5_chrom_info(
    embeddings_h5: str,
    chroms: Optional[Sequence[str]],
) -> Tuple[Dict[str, str], Dict[str, int], int, int]:
    """Map canonical chrom name to sample chrom name and H5 length."""
    with h5py.File(embeddings_h5, "r") as f:
        file_keys = [
            key
            for key in f.keys()
            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) == 2
        ]
        if not file_keys:
            raise ValueError(f"{embeddings_h5!r} contains no 2-D chromosome datasets.")

        requested = list(chroms) if chroms is not None else sorted(file_keys, key=chrom_sort_key)
        sample_by_canon: Dict[str, str] = {}
        n_windows_by_canon: Dict[str, int] = {}

        for chrom in requested:
            key = _resolve_chrom_key(file_keys, chrom)
            canon = canonical_chrom(chrom)
            sample_by_canon[canon] = chrom
            n_windows_by_canon[canon] = int(f[key].shape[0])

        first_key = _resolve_chrom_key(file_keys, requested[0])
        window_size = _read_window_size(f)
        embed_dim = _read_embed_dim(f, first_key)

    return sample_by_canon, n_windows_by_canon, window_size, embed_dim


def _info_scalar(value: object) -> Optional[object]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        return value[0] if value else None
    return value


def _max_float_info(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)):
        if not value:
            return None
        return max(float(v) for v in value)
    return float(value)


def load_deletion_intervals_from_vcf(
    vcf_path: str,
    chroms: Optional[Sequence[str]] = None,
    min_length: int = 50,
    max_length: Optional[int] = 10_000,
    min_af: Optional[float] = None,
    require_pass: bool = False,
    limit: Optional[int] = None,
) -> List[Variant]:
    """Load deletion intervals from a 1000G-style SV VCF.

    Coordinates are returned as 0-based half-open intervals, matching pysam and
    the rest of the project.
    """
    import pysam

    wanted = None
    if chroms is not None:
        wanted = {canonical_chrom(c) for c in chroms}

    intervals: List[Variant] = []
    try:
        vcf_ctx = pysam.VariantFile(vcf_path)
    except (OSError, NotImplementedError, ValueError) as exc:
        logger.warning(
            "pysam could not open %s (%s); falling back to streaming VCF parser.",
            vcf_path,
            exc,
        )
        return _load_deletion_intervals_from_text_vcf(
            vcf_path=vcf_path,
            chroms=chroms,
            min_length=min_length,
            max_length=max_length,
            min_af=min_af,
            require_pass=require_pass,
            limit=limit,
        )

    with vcf_ctx as vcf:
        hg_samples = [s for s in vcf.header.samples if s.startswith(("HG", "NA"))]
        if not hg_samples:
            logger.warning(
                "Requested HG/NA-only filtering but no samples starting with 'HG' or 'NA' "
                "found in %s. Keeping all records.",
                vcf_path,
            )

        try:
            for record in vcf:
                if wanted is not None and canonical_chrom(record.chrom) not in wanted:
                    continue

                # Filter by sample genotype if any HG/NA samples exist
                if hg_samples:
                    has_hg = False
                    for s in hg_samples:
                        gt = record.samples[s].get("GT")
                        if gt and any(g is not None and g > 0 for g in gt):
                            has_hg = True
                            break
                    if not has_hg:
                        continue

                if require_pass:
                    filters = set(record.filter.keys())
                    if filters and filters != {"PASS"}:
                        continue

                raw_sv_type = _info_scalar(record.info.get("SVTYPE"))
                sv_type = str(raw_sv_type) if raw_sv_type is not None else ""
                alts = {str(alt).strip("<>") for alt in (record.alts or ())}
                if sv_type != "DEL" and "DEL" not in alts:
                    continue

                if min_af is not None:
                    af = _max_float_info(record.info.get("AF"))
                    if af is None or af < min_af:
                        continue

                start = max(0, int(record.start))
                end = int(record.stop)
                raw_end = _info_scalar(record.info.get("END"))
                if raw_end is not None:
                    end = int(raw_end)
                end = max(start + 1, end)

                length = end - start
                if length < min_length:
                    continue
                if max_length is not None and length > max_length:
                    continue

                intervals.append(Variant(record.chrom, start, end, "DEL"))
                if limit is not None and len(intervals) >= limit:
                    break
        except NotImplementedError as exc:
            logger.warning(
                "pysam failed while reading %s (%s); falling back to streaming VCF parser.",
                vcf_path,
                exc,
            )
            return _load_deletion_intervals_from_text_vcf(
                vcf_path=vcf_path,
                chroms=chroms,
                min_length=min_length,
                max_length=max_length,
                min_af=min_af,
                require_pass=require_pass,
                limit=limit,
            )

    return intervals


def _parse_info(info_text: str) -> Dict[str, object]:
    info: Dict[str, object] = {}
    if not info_text or info_text == ".":
        return info
    for token in info_text.split(";"):
        if not token:
            continue
        if "=" not in token:
            info[token] = True
            continue
        key, value = token.split("=", 1)
        info[key] = value
    return info


def _open_text_vcf(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")


def _load_deletion_intervals_from_text_vcf(
    vcf_path: str,
    chroms: Optional[Sequence[str]] = None,
    min_length: int = 50,
    max_length: Optional[int] = 10_000,
    min_af: Optional[float] = None,
    require_pass: bool = False,
    limit: Optional[int] = None,
) -> List[Variant]:
    wanted = None
    if chroms is not None:
        wanted = {canonical_chrom(c) for c in chroms}

    intervals: List[Variant] = []
    hg_samples: List[str] = []
    sample_indices: List[int] = []

    with _open_text_vcf(vcf_path) as handle:
        for line in handle:
            if not line:
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                # Parse header to find HG samples
                header = line.rstrip("\n").split("\t")
                if len(header) > 9:
                    for i, s in enumerate(header[9:]):
                        if s.startswith(("HG", "NA")):
                            hg_samples.append(s)
                            sample_indices.append(i + 9)
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue
            chrom, pos_text, _id, _ref, alt, _qual, filt, info_text = fields[:8]
            if wanted is not None and canonical_chrom(chrom) not in wanted:
                continue

            # Filter by HG/NA genotype
            if hg_samples:
                has_hg = False
                for idx in sample_indices:
                    if idx >= len(fields):
                        continue
                    # Simple check for '0/1', '1/1', '1|0' etc.
                    # GT is usually the first field in the sample column
                    gt_part = fields[idx].split(":")[0]
                    if "1" in gt_part or "2" in gt_part: # crude but effective for DEL
                        has_hg = True
                        break
                if not has_hg:
                    continue

            if require_pass and filt not in (".", "PASS"):
                continue

            info = _parse_info(info_text)
            sv_type = str(info.get("SVTYPE", ""))
            alts = {part.strip("<>") for part in alt.split(",")}
            if sv_type != "DEL" and "DEL" not in alts:
                continue

            if min_af is not None:
                af_text = info.get("AF")
                if af_text is None:
                    continue
                af = max(float(x) for x in str(af_text).split(",") if x)
                if af < min_af:
                    continue

            start = max(0, int(pos_text) - 1)
            if "END" in info:
                end = int(str(info["END"]).split(",")[0])
            elif "SVLEN" in info:
                svlen = int(str(info["SVLEN"]).split(",")[0])
                end = start + abs(svlen)
            else:
                continue
            end = max(start + 1, end)

            length = end - start
            if length < min_length:
                continue
            if max_length is not None and length > max_length:
                continue
            intervals.append(Variant(chrom, start, end, "DEL"))
            if limit is not None and len(intervals) >= limit:
                break
    return intervals


def _interval_positive_windows(
    interval: Variant,
    window_size: int,
    n_windows: int,
    positive_mode: str,
    max_windows_per_interval: int,
    positive_stride_windows: int,
) -> List[int]:
    start_w = max(0, int(interval.start) // window_size)
    end_w = min(n_windows - 1, max(0, int(interval.end - 1) // window_size))
    if end_w < start_w:
        return []

    center_w = (start_w + end_w) // 2
    if positive_mode == "center":
        windows = [center_w]
    elif positive_mode == "breakpoints":
        windows = [start_w, end_w]
    elif positive_mode == "breakpoints_center":
        windows = [start_w, center_w, end_w]
    elif positive_mode == "span":
        step = max(1, int(positive_stride_windows))
        windows = list(range(start_w, end_w + 1, step))
        if windows[-1] != end_w:
            windows.append(end_w)
    else:
        raise ValueError(
            "positive_mode must be one of center, breakpoints, "
            f"breakpoints_center, span; got {positive_mode!r}"
        )

    windows = sorted(set(windows))
    cap = max(1, int(max_windows_per_interval))
    if len(windows) > cap:
        positions = np.linspace(0, len(windows) - 1, num=cap)
        windows = sorted({windows[int(round(i))] for i in positions})
    return windows


def _mark_deletion_mask(
    intervals: Iterable[Variant],
    n_windows: int,
    window_size: int,
    margin_windows: int,
) -> np.ndarray:
    mask = np.zeros(n_windows, dtype=np.bool_)
    for interval in intervals:
        start_w = max(0, int(interval.start) // window_size - margin_windows)
        end_w = min(
            n_windows - 1,
            max(0, int(interval.end - 1) // window_size) + margin_windows,
        )
        if end_w >= start_w:
            mask[start_w : end_w + 1] = True
    return mask


def build_sequence_prior_samples(
    embeddings_h5: str,
    vcf_path: str,
    chroms: Optional[Sequence[str]] = None,
    positive_mode: str = "span",
    min_length: int = 50,
    max_length: Optional[int] = 10_000,
    min_af: Optional[float] = None,
    require_pass: bool = False,
    positive_stride_windows: int = 1,
    max_positive_windows_per_interval: int = 8,
    max_positive_samples: Optional[int] = 300_000,
    negative_ratio: float = 1.0,
    negative_margin_bp: int = 1_000,
    seed: int = 42,
    vcf_limit: Optional[int] = None,
) -> Tuple[List[PriorSample], Dict[str, object]]:
    """Build balanced prior samples from an embedding H5 and deletion VCF."""
    rng = random.Random(seed)
    parsed_chroms = parse_chrom_list(chroms)
    sample_by_canon, n_by_canon, window_size, embed_dim = _selected_h5_chrom_info(
        embeddings_h5,
        parsed_chroms,
    )
    selected_chroms = list(sample_by_canon.values())

    intervals = load_deletion_intervals_from_vcf(
        vcf_path,
        chroms=selected_chroms,
        min_length=min_length,
        max_length=max_length,
        min_af=min_af,
        require_pass=require_pass,
        limit=vcf_limit,
    )
    intervals_by_canon: Dict[str, List[Variant]] = {c: [] for c in sample_by_canon}
    for interval in intervals:
        canon = canonical_chrom(interval.chrom)
        if canon in intervals_by_canon:
            intervals_by_canon[canon].append(interval)

    positive_keys = set()
    positives: List[PriorSample] = []
    for canon, chrom_intervals in intervals_by_canon.items():
        n_windows = n_by_canon[canon]
        sample_chrom = sample_by_canon[canon]
        for interval in chrom_intervals:
            for window_idx in _interval_positive_windows(
                interval,
                window_size=window_size,
                n_windows=n_windows,
                positive_mode=positive_mode,
                max_windows_per_interval=max_positive_windows_per_interval,
                positive_stride_windows=positive_stride_windows,
            ):
                key = (canon, int(window_idx))
                if key in positive_keys:
                    continue
                positive_keys.add(key)
                positives.append(PriorSample(sample_chrom, int(window_idx), 1))

    if max_positive_samples is not None and max_positive_samples > 0:
        if len(positives) > max_positive_samples:
            positives = rng.sample(positives, max_positive_samples)
            positive_keys = {(canonical_chrom(s.chrom), s.window_idx) for s in positives}

    if not positives:
        raise ValueError(
            "No positive deletion-prior samples were built. Check chrom names, "
            "length filters, and VCF/H5 genome build."
        )

    margin_windows = int(math.ceil(max(0, negative_margin_bp) / window_size))
    masks_by_canon: Dict[str, np.ndarray] = {}
    eligible_counts: Dict[str, int] = {}
    for canon, n_windows in n_by_canon.items():
        mask = _mark_deletion_mask(
            intervals_by_canon.get(canon, []),
            n_windows=n_windows,
            window_size=window_size,
            margin_windows=margin_windows,
        )
        masks_by_canon[canon] = mask
        eligible_counts[canon] = int((~mask).sum())

    target_negatives = int(round(len(positives) * float(negative_ratio)))
    negatives: List[PriorSample] = []
    negative_keys = set()
    eligible_chroms = [c for c, count in eligible_counts.items() if count > 0]
    if target_negatives > 0 and not eligible_chroms:
        raise ValueError("No eligible negative windows remain after masking DEL intervals.")

    weights = [eligible_counts[c] for c in eligible_chroms]
    max_attempts = max(10_000, target_negatives * 100)
    attempts = 0
    while len(negatives) < target_negatives and attempts < max_attempts:
        attempts += 1
        canon = rng.choices(eligible_chroms, weights=weights, k=1)[0]
        n_windows = n_by_canon[canon]
        window_idx = rng.randrange(n_windows)
        if masks_by_canon[canon][window_idx]:
            continue
        key = (canon, int(window_idx))
        if key in positive_keys or key in negative_keys:
            continue
        negative_keys.add(key)
        negatives.append(PriorSample(sample_by_canon[canon], int(window_idx), 0))

    if len(negatives) < target_negatives:
        logger.warning(
            "Requested %d negatives but sampled %d after %d attempts.",
            target_negatives,
            len(negatives),
            attempts,
        )

    samples = positives + negatives
    rng.shuffle(samples)

    stats: Dict[str, object] = {
        "chroms": selected_chroms,
        "window_size": window_size,
        "embed_dim": embed_dim,
        "n_deletion_intervals": len(intervals),
        "n_positive_samples": len(positives),
        "n_negative_samples": len(negatives),
        "negative_margin_bp": negative_margin_bp,
        "positive_mode": positive_mode,
        "max_positive_windows_per_interval": max_positive_windows_per_interval,
        "min_length": min_length,
        "max_length": max_length,
        "min_af": min_af,
        "require_pass": require_pass,
    }
    return samples, stats


class SequencePriorDataset(Dataset):
    """Return ``(embedding_window, label)`` for sequence-prior training."""

    def __init__(
        self,
        samples: Sequence[PriorSample],
        embeddings_h5: str,
        context_radius: int = 10,
        preload_chroms: Optional[object] = None,
        embed_dtype: torch.dtype = torch.float32,
    ) -> None:
        if context_radius < 0:
            raise ValueError(f"context_radius must be >= 0; got {context_radius}")
        if not samples:
            raise ValueError("SequencePriorDataset requires at least one sample.")

        self.samples = list(samples)
        self.embeddings_h5 = str(embeddings_h5)
        self.context_radius = int(context_radius)
        self.embed_dtype = embed_dtype
        self.labels = [int(s.label) for s in self.samples]

        self._h5: Optional[h5py.File] = None
        self._file_keys: Optional[List[str]] = None
        self._chrom_key_map: Dict[str, str] = {}
        self._arrays: Dict[str, np.ndarray] = {}

        with h5py.File(self.embeddings_h5, "r") as f:
            self.window_size = _read_window_size(f)
            sample_key = _resolve_chrom_key(list(f.keys()), self.samples[0].chrom)
            self.embed_dim = _read_embed_dim(f, sample_key)

            preload_list: List[str] = []
            if preload_chroms is True:
                preload_list = sorted({s.chrom for s in self.samples}, key=chrom_sort_key)
            elif preload_chroms:
                preload_list = parse_chrom_list(preload_chroms) or []

            file_keys = list(f.keys())
            for chrom in preload_list:
                key = _resolve_chrom_key(file_keys, chrom)
                self._chrom_key_map[chrom] = key
                self._arrays[chrom] = f[key][:]
                logger.info(
                    "Preloaded prior embeddings %s (key=%s) shape=%s",
                    chrom,
                    key,
                    self._arrays[chrom].shape,
                )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_file_keys"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _ensure_h5(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.embeddings_h5, "r")
            self._file_keys = list(self._h5.keys())
        return self._h5

    def _array_for_chrom(self, chrom: str):
        if chrom in self._arrays:
            return self._arrays[chrom]

        h5 = self._ensure_h5()
        if chrom not in self._chrom_key_map:
            self._chrom_key_map[chrom] = _resolve_chrom_key(self._file_keys or list(h5.keys()), chrom)
        return h5[self._chrom_key_map[chrom]]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        arr = self._array_for_chrom(sample.chrom)
        n_windows = int(arr.shape[0])
        center = min(max(int(sample.window_idx), 0), n_windows - 1)
        raw_start = center - self.context_radius
        raw_end = center + self.context_radius + 1
        start = max(0, raw_start)
        end = min(n_windows, raw_end)
        x_np = arr[start:end].astype(np.float32, copy=False)
        left_pad = max(0, -raw_start)
        right_pad = max(0, raw_end - n_windows)
        if left_pad:
            x_np = np.concatenate((np.repeat(x_np[:1], left_pad, axis=0), x_np), axis=0)
        if right_pad:
            x_np = np.concatenate((x_np, np.repeat(x_np[-1:], right_pad, axis=0)), axis=0)
        x = torch.from_numpy(np.asarray(x_np)).to(self.embed_dtype)
        label = torch.tensor(int(sample.label), dtype=torch.long)
        return x, label
