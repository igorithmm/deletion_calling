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

# Default context window: 50bp center + 1225bp flanks = 2500bp
DEFAULT_CONTEXT_BP = 2500


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
        """Extract an extended reference window centred at *center*.

        Args:
            chrom: Chromosome name.
            center: Centre position (0-based).
            context_bp: Total window size in bp (default 2500).

        Returns:
            DNA string of length ≤ context_bp (may be shorter near
            chromosome boundaries).
        """
        half = context_bp // 2
        return self.get_sequence(chrom, center - half, center + half)


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
        
        self.model = AutoModel.from_pretrained(
            resolved_path, config=config, trust_remote_code=True
        )
        self.model.eval()
        self.device = torch.device(device)
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
    ):
        self.fasta_path = fasta_path
        self.model_id = model_id
        self.device = device
        self.n_components = n_components
        self.context_bp = context_bp

        self._ref: Optional[ReferenceGenome] = None
        self._embedder: Optional[DNABERT2Embedder] = None
        self._pca = None  # sklearn PCA, fitted later

    def _ensure_ref(self):
        if self._ref is None:
            self._ref = ReferenceGenome(self.fasta_path)
            self._ref.__enter__()

    def _ensure_embedder(self):
        if self._embedder is None:
            self._embedder = DNABERT2Embedder(
                model_id=self.model_id, device=self.device
            )

    def close(self):
        """Release resources."""
        if self._ref is not None:
            self._ref.__exit__(None, None, None)
            self._ref = None

    # ------------------------------------------------------------------
    # Raw embedding extraction (before PCA)
    # ------------------------------------------------------------------

    def get_raw_embedding(self, chrom: str, center: int) -> np.ndarray:
        """Extract a 768-dim embedding for a genomic position.

        Args:
            chrom: Chromosome name.
            center: Centre of the 50bp window (0-based).

        Returns:
            np.ndarray of shape (768,), dtype float32.
        """
        self._ensure_ref()
        self._ensure_embedder()
        seq = self._ref.get_extended_window(chrom, center, self.context_bp)
        if len(seq) < 10:
            # Degenerate region — return zeros
            return np.zeros(768, dtype=np.float32)
        return self._embedder.embed_sequence(seq)

    # ------------------------------------------------------------------
    # PCA fitting and transformation
    # ------------------------------------------------------------------

    def fit_pca(self, embeddings: np.ndarray):
        """Fit PCA on training embeddings.

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
        self._pca.fit(embeddings)
        explained = self._pca.explained_variance_ratio_.sum()
        logger.info(
            "PCA fitted. Explained variance: %.2f%% with %d components.",
            explained * 100,
            self.n_components,
        )

    def transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply fitted PCA.

        Args:
            embeddings: (N, 768) or (768,) array.

        Returns:
            (N, K) or (K,) array with K = n_components.
        """
        if self._pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca() first.")
        single = embeddings.ndim == 1
        if single:
            embeddings = embeddings.reshape(1, -1)
        reduced = self._pca.transform(embeddings).astype(np.float32)
        return reduced.squeeze(0) if single else reduced

    def save_pca(self, path: str):
        """Persist the fitted PCA model to disk."""
        import joblib

        if self._pca is None:
            raise RuntimeError("PCA not fitted.")
        joblib.dump(self._pca, path)
        logger.info("PCA model saved to %s", path)

    def load_pca(self, path: str):
        """Load a previously fitted PCA model from disk."""
        import joblib

        self._pca = joblib.load(path)
        logger.info(
            "PCA model loaded from %s (n_components=%d)",
            path,
            self._pca.n_components_,
        )

    # ------------------------------------------------------------------
    # Combined: raw → PCA-reduced context vector
    # ------------------------------------------------------------------

    def get_context_vector(self, chrom: str, center: int) -> np.ndarray:
        """Get the PCA-reduced context vector for a genomic position.

        Args:
            chrom: Chromosome name.
            center: Centre of the 50bp window.

        Returns:
            np.ndarray of shape (K,), dtype float32.
        """
        raw = self.get_raw_embedding(chrom, center)
        return self.transform_pca(raw)
