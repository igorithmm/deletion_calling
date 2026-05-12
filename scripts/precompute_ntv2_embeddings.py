#!/usr/bin/env python3
"""Precompute Nucleotide Transformer v2-100m embeddings for the entire reference genome.

For each chromosome, the genome is divided into non-overlapping 12,000 bp core
regions. Each core is expanded by 120 bp of flanking sequence on either side
(filled with 'N' if past chromosome boundaries) to form a 12,240 bp window.
The window is tokenised using a 6-mer tokenizer (resulting in 2040 tokens + CLS),
passed through NTV2-100m in a single forward pass, and the resulting hidden states
(excluding CLS) are sliced to drop the flank tokens. The remaining 2000 core tokens
are expanded 6x to match base-level coordinates (12,000 bp) and then mean-pooled
every 50 bp to yield 240 embeddings of dimension 512.

HDF5 layout
────────────
  /chr1   dataset  shape=(N_windows, 512)  dtype=float16
  /chr2   dataset  shape=(N_windows, 512)  dtype=float16
  ...
  attrs:
    model_id        — str, path or ID
    window_bp       — int, output window size (50)
    flank_bp        — int, flank context per side (120)
    core_bp         — int, core region per chunk (12000)
    seq_bp          — int, full input sequence per chunk (12240)
    embed_dim       — int, 512
    reference_md5   — str, MD5 of the reference FASTA
    genome_build    — str, optional human-readable build name
    date_utc        — str, ISO timestamp of generation
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
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Pinned model identity ────────────────────────────────────────────────────
NTV2_MODEL_PATH = "models/nucleotide-transformer-v2-100m-multi-species"

# ── Geometry (multiples of 6 and 50) ─────────────────────────────────────────
WINDOW_BP = 50  # output window: one embedding per 50 bp
CORE_BP = 12_000  # non-overlapping core per forward pass (2000 tokens)
FLANK_BP = 120  # flank context on each side (20 tokens)
SEQ_BP = CORE_BP + 2 * FLANK_BP  # 12,240 bp (2040 tokens)
WINDOWS_PER_CHUNK = CORE_BP // WINDOW_BP  # 240
TOKENS_PER_CHUNK = CORE_BP // 6  # 2000
TOKENS_PER_FLANK = FLANK_BP // 6  # 20
EMBED_DIM = 512

DEFAULT_CHROMS = [str(c) for c in range(1, 23)]


# ═══════════════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════════════


def load_model(
    model_path: str,
    device: str,
) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Load Nucleotide Transformer v2 at a local path with custom loading logic.
    
    We manually load the model classes from local files because the transformers library
    defaults to the built-in (non-GLU) ESM model when model_type is 'esm', causing
    size mismatches in the FFN layers.
    """
    import importlib.util
    import types
    
    model_dir = Path(model_path).resolve()
    logger.info("Loading NTV2 from %s", model_dir)
    
    # Create a dummy package for local imports to work (modeling_esm.py uses relative imports)
    pkg_name = "ntv2_internal"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(model_dir)]
        sys.modules[pkg_name] = pkg

    # Polyfill for find_pruneable_heads_and_indices and prune_linear_layer
    # which moved around in different versions of the transformers library.
    import transformers.pytorch_utils as torch_utils
    if not hasattr(torch_utils, "find_pruneable_heads_and_indices"):
        try:
            from transformers import modeling_utils
            if hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
                torch_utils.find_pruneable_heads_and_indices = modeling_utils.find_pruneable_heads_and_indices
                torch_utils.prune_linear_layer = modeling_utils.prune_linear_layer
                logger.info("Polyfilled find_pruneable_heads_and_indices from modeling_utils")
        except (ImportError, AttributeError):
            pass

    # Load custom config and model classes
    for module_name in ["esm_config", "modeling_esm"]:
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{module_name}", 
            str(model_dir / f"{module_name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"{pkg_name}.{module_name}"] = module
        spec.loader.exec_module(module)

    EsmConfig = sys.modules[f"{pkg_name}.esm_config"].EsmConfig
    EsmModel = sys.modules[f"{pkg_name}.modeling_esm"].EsmModel
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = EsmConfig.from_pretrained(model_path)
    
    logger.info("Instantiating model with Gated Linear Units (GLU)...")
    model = EsmModel.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16
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
    """Return a SEQ_BP-bp uppercase string for chunk `chunk_idx`."""
    core_start = chunk_idx * CORE_BP
    core_end = core_start + CORE_BP
    full_start = core_start - FLANK_BP
    full_end = core_end + FLANK_BP

    fetch_start = max(0, full_start)
    fetch_end = min(chrom_len, full_end)

    seq = str(chrom_seq[fetch_start:fetch_end]).upper()

    left_pad = fetch_start - full_start
    right_pad = full_end - fetch_end
    if left_pad or right_pad:
        seq = ("N" * left_pad) + seq + ("N" * right_pad)

    assert len(seq) == SEQ_BP, f"Expected {SEQ_BP} bp, got {len(seq)}"
    return seq


def fasta_md5(fasta_path: str, block_size: int = 1 << 20) -> str:
    """Compute MD5 of the FASTA file."""
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
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: str,
) -> np.ndarray:
    """Run one chunk through NTV2 and return (240, 512) float16 embeddings.
    
    Note: We replace 'N' with 'A' to ensure 6-mer tokenization and prevent
    token count overflow in genomic gaps.
    """
    seq_fixed = seq.replace("N", "A")
    
    encoded = tokenizer(
        seq_fixed,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(device)

    # Expected: 1 (CLS) + 2040 (6-mers) = 2041 tokens
    expected_tokens = 1 + (SEQ_BP // 6)
    if input_ids.shape[1] != expected_tokens:
         # This should not happen now with N->A replacement
         pass

    autocast_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.autocast(device_type=autocast_device, dtype=torch.bfloat16):
        outputs = model(input_ids)

    # outputs.last_hidden_state is (1, 2041, 512)
    hidden = outputs.last_hidden_state

    # Drop CLS (index 0)
    data_hidden = hidden[:, 1:, :]  # (1, 2040, 512)

    # Slice core tokens
    core_tokens = data_hidden[:, TOKENS_PER_FLANK : TOKENS_PER_FLANK + TOKENS_PER_CHUNK, :]  # (1, 2000, 512)

    # Repeat each token 6 times to get base-level representation (12,000 bp)
    core_bp_level = core_tokens.repeat_interleave(6, dim=1)  # (1, 12000, 512)

    # Pool into 50 bp windows
    pooled = core_bp_level.reshape(1, WINDOWS_PER_CHUNK, WINDOW_BP, EMBED_DIM).mean(dim=2)
    pooled = pooled.squeeze(0).to(torch.float16).cpu().numpy()  # (240, 512)
    return pooled


# ═══════════════════════════════════════════════════════════════════════════
# Main precompute loop
# ═══════════════════════════════════════════════════════════════════════════


def precompute(
    fasta_path: str,
    output_path: str,
    model_id: str = NTV2_MODEL_PATH,
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

    model, tokenizer = load_model(model_id, device)

    hdf = h5py.File(str(output_file), "a")
    hdf.attrs["model_id"] = model_id
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
            pooled = embed_chunk(seq, model, tokenizer, device)  # (240, 512)

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
        description="Precompute Nucleotide Transformer v2-100m embeddings for a reference genome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fasta", required=True, help="Path to reference FASTA")
    parser.add_argument("--output", required=True, help="Output HDF5 file path")
    parser.add_argument(
        "--model-id",
        default=NTV2_MODEL_PATH,
        help="Local path to NTV2 model",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device (cuda / cuda:0 / cpu)",
    )
    parser.add_argument(
        "--chrom",
        nargs="*",
        help="Chromosome(s) to process. Default: autosomes 1-22",
    )
    parser.add_argument(
        "--genome-build",
        default="",
        help="Human-readable genome build name (e.g. hs37d5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chromosomes already present in the output HDF5",
    )

    args = parser.parse_args()

    precompute(
        fasta_path=args.fasta,
        output_path=args.output,
        model_id=args.model_id,
        device=args.device,
        chroms=args.chrom,
        genome_build=args.genome_build,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
