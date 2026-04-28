"""DNABERT-2 genomic context extraction for DeepSV3.

This module handles:
  1. Extracting extended reference windows from a FASTA file.
  2. Running frozen DNABERT-2 inference to obtain 768-dim embeddings.
  3. Fitting PCA on training embeddings and transforming to K dimensions.
  4. Caching / loading precomputed embeddings from disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pysam
import torch

logger = logging.getLogger(__name__)

# Default model identifiers
DNABERT2_MODEL_ID = "zhihan1996/DNABERT-2-117M"
DNABERT2_LOCAL_PATH = "/datasets/igorno-genomes_1000/weights/dnabert2"

# Default context window: 50bp window + 999bp flanks each side = 2048bp
DEFAULT_CONTEXT_BP = 2048
DEFAULT_FLANK_BP = 999


class ReferenceGenome:
    """Thin wrapper around pysam.FastaFile for reference sequence access."""

    def __init__(self, fasta_path: str):
        self.fasta_path = fasta_path
        self._fasta: Optional[pysam.FastaFile] = None

    def __enter__(self):
        self._fasta = pysam.FastaFile(self.fasta_path)
        return self

    def __exit__(self, *exc):
        if self._fasta:
            self._fasta.close()

    @property
    def references(self):
        """List of chromosome/contig names in the reference."""
        return self._fasta.references if self._fasta else []

    def get_sequence(self, chrom: str, start: int, end: int) -> str:
        """Fetch the reference sequence for *chrom*:*start*–*end* (0-based, half-open).

        Coordinates are clamped to [0, chrom_length).
        """
        if not self._fasta:
            raise RuntimeError("FASTA file not opened. Use context manager.")
        chrom_len = self._fasta.get_reference_length(chrom)
        start = max(0, start)
        end = min(end, chrom_len)
        return self._fasta.fetch(chrom, start, end).upper()

    def get_extended_window(
        self, chrom: str, center: int, context_bp: int = DEFAULT_CONTEXT_BP
    ) -> str:
        """Extract an extended reference window centred at *center* (legacy)."""
        half = context_bp // 2
        return self.get_sequence(chrom, center - half, center + half)

    def get_window_with_flanks(
        self,
        chrom: str,
        start: int,
        end: int,
        flank_bp: int = DEFAULT_FLANK_BP,
    ) -> str:
        """Extract sequence: ``flank_bp`` upstream of *start* + window + ``flank_bp`` downstream of *end*.

        Matches DeepSV2.5 spec: 999 bp upstream + 50 bp window + 999 bp downstream = 2048 bp.

        Args:
            chrom: Chromosome name.
            start: Window start (0-based, inclusive).
            end: Window end (0-based, exclusive).
            flank_bp: Flank size in bp on each side (default 999).

        Returns:
            DNA string. May be shorter than ``flank_bp*2 + (end-start)`` near
            chromosome boundaries.
        """
        return self.get_sequence(chrom, start - flank_bp, end + flank_bp)


class DNABERT2Embedder:
    """Frozen DNABERT-2 model for extracting 768-dim genomic embeddings.

    Usage::

        embedder = DNABERT2Embedder(device="cuda")
        vec = embedder.embed_sequence("ATCGATCG...")  # → np.ndarray (768,)
    """

    def __init__(
        self,
        model_id: str = DNABERT2_MODEL_ID,
        device: str = "cpu",
    ):
        from transformers import AutoTokenizer, AutoModel

        # Prefer local path if it exists
        if Path(model_id).exists():
            resolved_path = model_id
            logger.info("Loading DNABERT-2 from local path: %s", resolved_path)
        elif Path(DNABERT2_LOCAL_PATH).exists():
            resolved_path = DNABERT2_LOCAL_PATH
            logger.info("Loading DNABERT-2 from default local path: %s", resolved_path)
        else:
            resolved_path = model_id
            logger.info("Loading DNABERT-2 from HuggingFace: %s", resolved_path)

        logger.info("Loading DNABERT-2 tokenizer …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, trust_remote_code=True
        )

        logger.info("Loading DNABERT-2 config and model (forcing PyTorch attention) …")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(resolved_path, trust_remote_code=True)
        # Force non-zero dropout to bypass Triton Flash Attention in bert_layers.py
        config.attention_probs_dropout_prob = 0.1
        
        # FIX: DNABERT-2 custom modeling code expects pad_token_id to be in the config
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id
        
        # DNABERT-2's bert_layers.py constructs ALiBi tensors inside __init__
        # via torch.arange(...) etc. Those calls inherit the default device.
        # In recent transformers + accelerate, the default device can be
        # silently switched to 'meta' during from_pretrained, which causes
        #   alibi = slopes.unsqueeze(...) * -relative_position
        # to mix a cpu tensor (slopes) with a meta tensor (relative_position)
        # and raise:
        #   "Tensor on device meta is not on the expected device cpu!"
        #
        # Forcing the default device to cpu for the duration of construction
        # ensures every implicit tensor inside the custom modeling code is
        # created on cpu, regardless of what transformers does globally.
        # low_cpu_mem_usage=False is also passed as belt-and-braces — it
        # disables the meta→materialise loading path on the transformers side.
        with torch.device("cpu"):
            self.model = AutoModel.from_pretrained(
                resolved_path,
                config=config,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
            )
        self.model.eval()
        self.device = torch.device(device)

        # Defensive guard: if any parameters/buffers are still on 'meta',
        # the model would silently produce garbage outputs after .to().
        meta_params = [n for n, p in self.model.named_parameters() if p.is_meta]
        meta_buffers = [n for n, b in self.model.named_buffers() if b.is_meta]
        if meta_params or meta_buffers:
            raise RuntimeError(
                f"DNABERT-2 still has meta tensors after construction: "
                f"params={meta_params[:3]}..., buffers={meta_buffers[:3]}... "
                "Likely a transformers/accelerate version mismatch — try "
                "pinning transformers==4.29.2 (the version DNABERT-2 ships with)."
            )

        self.model.to(self.device)

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        logger.info(
            "DNABERT-2 loaded (%.1f M params, device=%s)",
            sum(p.numel() for p in self.model.parameters()) / 1e6,
            self.device,
        )

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> np.ndarray:
        """Return the mean-pooled 768-dim embedding for a DNA sequence.

        Args:
            sequence: DNA string (A/C/G/T/N). Length should be ≤ ~2500 bp
                      to stay within the 512-token BPE budget.

        Returns:
            np.ndarray of shape (768,), dtype float32.
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # Custom DNABERT-2 model returns a tuple (hidden_states, pooled_output)
        if isinstance(outputs, (tuple, list)):
            hidden = outputs[0]
        else:
            hidden = outputs.last_hidden_state  # (1, num_tokens, 768)

        # Mean pooling over all tokens (excluding padding — none here)
        embedding = hidden.squeeze(0).mean(dim=0)  # (768,)
        return embedding.cpu().numpy().astype(np.float32)

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """Embed a batch of sequences. Returns (N, 768)."""
        embeddings = []
        for seq in sequences:
            embeddings.append(self.embed_sequence(seq))
        return np.stack(embeddings)


class GenomicContextExtractor:
    """High-level class that combines reference access + DNABERT-2 + PCA.

    Usage::

        ctx = GenomicContextExtractor(
            fasta_path="raw/hs37d5.fa",
            device="cuda",
            n_components=8,
        )
        ctx.fit_pca(training_embeddings)   # (N, 768) array
        vec = ctx.get_context_vector("1", center=10025)  # → (8,)
    """

    def __init__(
        self,
        fasta_path: str,
        model_id: str = DNABERT2_MODEL_ID,
        device: str = "cpu",
        n_components: int = 8,
        context_bp: int = DEFAULT_CONTEXT_BP,
        flank_bp: int = DEFAULT_FLANK_BP,
    ):
        self.fasta_path = fasta_path
        self.model_id = model_id
        self.device = device
        self.n_components = n_components
        self.context_bp = context_bp
        self.flank_bp = flank_bp

        self._ref: Optional[ReferenceGenome] = None
        self._embedder: Optional[DNABERT2Embedder] = None
        self._embedder_init_failed: bool = False
        self._pca = None  # sklearn PCA, fitted later
        # z-score parameters fitted alongside PCA
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    def _ensure_ref(self):
        if self._ref is None:
            self._ref = ReferenceGenome(self.fasta_path)
            try:
                self._ref.__enter__()
            except Exception:
                self._ref = None
                raise

    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        # Fail fast on a previous init failure: don't reload DNABERT-2 once
        # per window (otherwise a config error spams the log with hundreds of
        # "Loading DNABERT-2…" lines while every window silently fails).
        if self._embedder_init_failed:
            raise RuntimeError(
                "DNABERT-2 embedder failed to initialise on a previous attempt; "
                "fix the underlying error and restart."
            )
        try:
            self._embedder = DNABERT2Embedder(
                model_id=self.model_id, device=self.device
            )
        except Exception:
            self._embedder_init_failed = True
            raise

    def close(self):
        """Release resources."""
        if self._ref is not None:
            self._ref.__exit__(None, None, None)
            self._ref = None

    # ------------------------------------------------------------------
    # Raw embedding extraction (before PCA)
    # ------------------------------------------------------------------

    def get_raw_embedding(self, chrom: str, center: int) -> np.ndarray:
        """Legacy: extract a 768-dim embedding centred at *center*."""
        self._ensure_ref()
        self._ensure_embedder()
        seq = self._ref.get_extended_window(chrom, center, self.context_bp)
        if len(seq) < 10:
            return np.zeros(768, dtype=np.float32)
        return self._embedder.embed_sequence(seq)

    def get_raw_embedding_for_window(
        self, chrom: str, start: int, end: int
    ) -> np.ndarray:
        """Extract 768-dim embedding for a window using ``flank_bp`` flanks.

        Sequence layout: [start-flank_bp, end+flank_bp). For default flank=999
        and a 50bp window this yields 2048 bp, matching the DeepSV2.5 spec.
        """
        self._ensure_ref()
        self._ensure_embedder()
        seq = self._ref.get_window_with_flanks(chrom, start, end, self.flank_bp)
        if len(seq) < 10:
            return np.zeros(768, dtype=np.float32)
        return self._embedder.embed_sequence(seq)

    # ------------------------------------------------------------------
    # PCA fitting and transformation
    # ------------------------------------------------------------------

    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA + z-score scaler on training embeddings.

        Args:
            embeddings: (N, 768) array of raw DNABERT-2 embeddings.
        """
        from sklearn.decomposition import PCA

        logger.info(
            "Fitting PCA(%d) on %d embeddings …",
            self.n_components,
            embeddings.shape[0],
        )
        self._pca = PCA(n_components=self.n_components)
        reduced = self._pca.fit_transform(embeddings)
        self._scaler_mean = reduced.mean(axis=0).astype(np.float32)
        self._scaler_std = reduced.std(axis=0).astype(np.float32) + 1e-6
        explained = self._pca.explained_variance_ratio_.sum()
        logger.info(
            "PCA fitted. Explained variance: %.2f%% with %d components.",
            explained * 100,
            self.n_components,
        )
        logger.info(
            "Scaler stats — mean range [%.3f, %.3f], std range [%.3f, %.3f]",
            float(self._scaler_mean.min()), float(self._scaler_mean.max()),
            float(self._scaler_std.min()), float(self._scaler_std.max()),
        )

    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply fitted PCA followed by z-score normalisation.

        Args:
            embeddings: (N, 768) or (768,) array.

        Returns:
            (N, K) or (K,) array with K = n_components, z-scored.
        """
        if self._pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca() first.")
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings.reshape(1, -1)
        reduced = self._pca.transform(embeddings).astype(np.float32)
        if self._scaler_mean is not None and self._scaler_std is not None:
            reduced = (reduced - self._scaler_mean) / self._scaler_std
        return reduced.squeeze(0) if single else reduced

    def save_pca(self, path: str):
        """Persist the fitted PCA model + scaler to disk."""
        import joblib

        if self._pca is None:
            raise RuntimeError("PCA not fitted.")
        joblib.dump(
            {
                "pca": self._pca,
                "scaler_mean": self._scaler_mean,
                "scaler_std": self._scaler_std,
            },
            path,
        )
        logger.info("PCA + scaler saved to %s", path)

    def load_pca(self, path: str):
        """Load a previously fitted PCA model (and scaler if present)."""
        import joblib

        obj = joblib.load(path)
        if isinstance(obj, dict) and "pca" in obj:
            self._pca = obj["pca"]
            self._scaler_mean = obj.get("scaler_mean")
            self._scaler_std = obj.get("scaler_std")
        else:
            # Backward compat: legacy file stored only the PCA object
            self._pca = obj
            self._scaler_mean = None
            self._scaler_std = None
        logger.info(
            "PCA loaded from %s (n_components=%d, scaler=%s)",
            path,
            self._pca.n_components_,
            "yes" if self._scaler_mean is not None else "no",
        )

    # ------------------------------------------------------------------
    # Combined: raw → PCA-reduced context vector
    # ------------------------------------------------------------------

    def get_context_vector(self, chrom: str, center: int) -> np.ndarray:
        """Legacy: PCA-reduced context vector for a centred position."""
        raw = self.get_raw_embedding(chrom, center)
        return self.transform_pca(raw)

    def get_context_vector_for_window(
        self, chrom: str, start: int, end: int
    ) -> np.ndarray:
        """PCA-reduced context vector for a window, using ``flank_bp`` flanks."""
        raw = self.get_raw_embedding_for_window(chrom, start, end)
        return self.transform_pca(raw)
