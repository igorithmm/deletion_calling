#!/usr/bin/env python3
"""Precompute HyenaDNA-small embeddings for the entire reference genome.

For each chromosome, the genome is divided into non-overlapping 30,000 bp core
regions. Each core is expanded by 1,000 bp of flanking sequence on either side
(filled with 'N' if past chromosome boundaries) to form a 32,000 bp window.
The window is tokenised character-by-character (1 bp = 1 token), passed through
HyenaDNA in a single forward pass, and the resulting (32000, 256) hidden states
are sliced to drop the flank tokens, reshaped to (600, 50, 256), and mean-pooled
along the 50-token axis to yield 600 embeddings of dimension 256 — one per
non-overlapping 50 bp window.

HDF5 layout
────────────
  /chr1   dataset  shape=(N_windows, 256)  dtype=float16
  /chr2   dataset  shape=(N_windows, 256)  dtype=float16
  ...
  attrs:
    model_id        — str, HuggingFace model identifier
    model_revision  — str, pinned git revision (commit SHA)
    window_bp       — int, output window size (50)
    flank_bp        — int, flank context per side (1000)
    core_bp         — int, core region per chunk (30000)
    seq_bp          — int, full input sequence per chunk (32000)
    embed_dim       — int, 256
    reference_md5   — str, MD5 of the reference FASTA
    genome_build    — str, optional human-readable build name
    date_utc        — str, ISO timestamp of generation

Look-up
───────
  import h5py
  with h5py.File("embeddings.h5", "r") as f:
      vec = f["chr21"][window_idx]    # (256,) float16
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import logging
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pyfaidx
import torch
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Pinned model identity ────────────────────────────────────────────────────
HYENADNA_MODEL_ID = "LongSafari/hyenadna-small-32k-seqlen-hf"
# Pin a specific commit so the weights cannot silently change. The user must
# verify / update this SHA after running once. Override via --model-revision.
HYENADNA_MODEL_REVISION = "main"  # PLACEHOLDER — override with --model-revision

# ── Geometry (matches the algorithm in the module docstring) ─────────────────
WINDOW_BP = 50          # output window: one embedding per 50 bp
CORE_BP = 30_000        # non-overlapping core per forward pass
FLANK_BP = 1_000        # flank context on each side
SEQ_BP = CORE_BP + 2 * FLANK_BP  # 32_000 — total input length
WINDOWS_PER_CHUNK = CORE_BP // WINDOW_BP  # 600
EMBED_DIM = 256

DEFAULT_CHROMS = [str(c) for c in range(1, 23)]


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_model(
    model_id: str,
    revision: str,
    device: str,
) -> tuple[AutoModel, AutoTokenizer]:
    """Load HyenaDNA at a pinned revision in bfloat16, eval mode, on `device`."""
    logger.info("Loading %s @ %s (trust_remote_code=True)", model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════
# Sequence utilities
# ═══════════════════════════════════════════════════════════════════════════

def fetch_chunk_sequence(
    chrom_seq: pyfaidx.FastaRecord,
    chunk_idx: int,
    chrom_len: int,
) -> str:
    """Return a 32_000-bp uppercase string for chunk `chunk_idx`.

    The core spans [chunk_idx * CORE_BP, (chunk_idx + 1) * CORE_BP); a 1_000 bp
    flank is added on each side. Anything past chromosome boundaries is
    padded with 'N' so the result is always exactly SEQ_BP characters.
    """
    core_start = chunk_idx * CORE_BP
    core_end = core_start + CORE_BP
    full_start = core_start - FLANK_BP
    full_end = core_end + FLANK_BP

    fetch_start = max(0, full_start)
    fetch_end = min(chrom_len, full_end)

    seq = str(chrom_seq[fetch_start:fetch_end]).upper()

    left_pad = fetch_start - full_start   # > 0 if we clipped on the left
    right_pad = full_end - fetch_end      # > 0 if we clipped on the right
    if left_pad or right_pad:
        seq = ("N" * left_pad) + seq + ("N" * right_pad)

    assert len(seq) == SEQ_BP, f"Expected {SEQ_BP} bp, got {len(seq)}"
    return seq


def fasta_md5(fasta_path: str, block_size: int = 1 << 20) -> str:
    """Compute MD5 of the FASTA file (raw bytes, including headers)."""
    h = hashlib.md5()
    with open(fasta_path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Per-chunk forward pass + pooling
# ═══════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def embed_chunk(
    seq: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
) -> np.ndarray:
    """Run one chunk through HyenaDNA and return (600, 256) float16 embeddings.

    Steps:
      1. Tokenise the 32_000-char string (character tokenizer → 32_000 tokens).
      2. Forward pass under bfloat16 autocast → (1, 32_000, 256) hidden states.
      3. Drop the flanks: keep positions [FLANK_BP : FLANK_BP + CORE_BP].
      4. Reshape to (1, 600, 50, 256) and mean over the 50-token axis.
    """
    encoded = tokenizer(
        seq,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)

    if input_ids.shape[1] != SEQ_BP:
        raise RuntimeError(
            f"Tokenizer produced {input_ids.shape[1]} tokens for a {SEQ_BP}-bp "
            "sequence; expected 1 token per base. Check the tokenizer config."
        )

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        outputs = model(input_ids)

    # HyenaDNA-hf returns a BaseModelOutput with last_hidden_state of (1, L, D).
    hidden = outputs.last_hidden_state  # (1, 32000, 256)

    core = hidden[:, FLANK_BP : FLANK_BP + CORE_BP, :]  # (1, 30000, 256)
    pooled = core.reshape(1, WINDOWS_PER_CHUNK, WINDOW_BP, EMBED_DIM).mean(dim=2)
    pooled = pooled.squeeze(0).to(torch.float16).cpu().numpy()  # (600, 256)
    return pooled


# ═══════════════════════════════════════════════════════════════════════════
# Main precompute loop
# ═══════════════════════════════════════════════════════════════════════════

def precompute(
    fasta_path: str,
    output_path: str,
    model_id: str = HYENADNA_MODEL_ID,
    model_revision: str = HYENADNA_MODEL_REVISION,
    device: str = "cuda",
    chroms: list[str] | None = None,
    genome_build: str = "",
    resume: bool = False,
):
    """Generate embeddings for every chromosome and write them to HDF5."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fasta = pyfaidx.Fasta(fasta_path, sequence_always_upper=False, as_raw=True)
    available = set(fasta.keys())

    if chroms is None:
        chroms_to_process = []
        for c in DEFAULT_CHROMS:
            if c in available:
                chroms_to_process.append(c)
            elif f"chr{c}" in available:
                chroms_to_process.append(f"chr{c}")
        if not chroms_to_process:
            logger.warning("No default autosomes found; processing ALL contigs.")
            chroms_to_process = list(available)
    else:
        chroms_to_process = [c for c in chroms if c in available]
        missing = set(chroms) - set(chroms_to_process)
        if missing:
            logger.warning("Chromosomes not found in FASTA: %s", missing)

    logger.info(
        "Will process %d chromosomes: %s",
        len(chroms_to_process),
        ", ".join(chroms_to_process),
    )

    logger.info("Computing reference MD5 …")
    ref_md5 = fasta_md5(fasta_path)
    logger.info("Reference MD5: %s", ref_md5)

    model, tokenizer = load_model(model_id, model_revision, device)

    hdf = h5py.File(str(output_file), "a")
    hdf.attrs["model_id"] = model_id
    hdf.attrs["model_revision"] = model_revision
    hdf.attrs["window_bp"] = WINDOW_BP
    hdf.attrs["flank_bp"] = FLANK_BP
    hdf.attrs["core_bp"] = CORE_BP
    hdf.attrs["seq_bp"] = SEQ_BP
    hdf.attrs["embed_dim"] = EMBED_DIM
    hdf.attrs["reference_md5"] = ref_md5
    hdf.attrs["genome_build"] = genome_build
    hdf.attrs["date_utc"] = _dt.datetime.utcnow().isoformat(timespec="seconds")

    for chrom in chroms_to_process:
        chrom_record = fasta[chrom]
        chrom_len = len(chrom_record)
        n_windows = math.ceil(chrom_len / WINDOW_BP)
        n_chunks = math.ceil(chrom_len / CORE_BP)

        if chrom in hdf:
            if resume:
                logger.info("Skipping %s — already present", chrom)
                continue
            del hdf[chrom]

        logger.info(
            "Processing %s  length=%s bp  windows=%s  chunks=%s",
            chrom,
            f"{chrom_len:,}",
            f"{n_windows:,}",
            f"{n_chunks:,}",
        )

        chunk_rows = min(1024, n_windows)
        dataset = hdf.create_dataset(
            chrom,
            shape=(n_windows, EMBED_DIM),
            dtype=np.float16,
            chunks=(chunk_rows, EMBED_DIM),
        )

        for k in range(n_chunks):
            seq = fetch_chunk_sequence(chrom_record, k, chrom_len)
            pooled = embed_chunk(seq, model, tokenizer, device)  # (600, 256)

            row_start = k * WINDOWS_PER_CHUNK
            row_end = min(row_start + WINDOWS_PER_CHUNK, n_windows)
            n_to_write = row_end - row_start
            dataset[row_start:row_end] = pooled[:n_to_write]

            if (k + 1) % 50 == 0 or k == n_chunks - 1:
                logger.info("  %s  chunk %d/%d", chrom, k + 1, n_chunks)

        hdf.flush()
        logger.info("  ✓ %s done", chrom)

    fasta.close()
    hdf.close()
    logger.info("═══ All done. Output: %s ═══", output_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Precompute HyenaDNA embeddings for a reference genome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fasta", required=True, help="Path to reference FASTA")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument(
        "--model-id", default=HYENADNA_MODEL_ID,
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--model-revision", default=HYENADNA_MODEL_REVISION,
        help="Pinned git revision (commit SHA). Avoid 'main' for reproducibility.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="PyTorch device (cuda / cuda:0 / cpu)",
    )
    parser.add_argument(
        "--chrom", nargs="*",
        help="Chromosome(s) to process. Default: autosomes 1-22",
    )
    parser.add_argument(
        "--genome-build", default="",
        help="Human-readable genome build name (e.g. GRCh37, hs37d5)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip chromosomes already present in the output HDF5",
    )

    args = parser.parse_args()

    if args.model_revision == "main":
        logger.warning(
            "model_revision='main' is not pinned — weights may change silently. "
            "Pass --model-revision <commit-sha> for reproducibility."
        )

    precompute(
        fasta_path=args.fasta,
        output_path=args.output,
        model_id=args.model_id,
        model_revision=args.model_revision,
        device=args.device,
        chroms=args.chrom,
        genome_build=args.genome_build,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
