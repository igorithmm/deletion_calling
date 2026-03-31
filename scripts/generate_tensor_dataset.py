#!/usr/bin/env python3
"""Generate multichannel tensor dataset from BAM, VCF, and reference FASTA.

This script produces .pt files containing:
  - alignment:  float32 tensor (13, H, W) — per-read alignment features
  - context:    float32 vector (K,)       — PCA-reduced DNABERT-2 embedding
  - label:      int                        — 0 (non-deletion) or 1 (deletion)
  - chrom:      str
  - start:      int
  - end:        int

Workflow:
  1. Parse VCF to get deletion and non-deletion regions (reuses existing VCFHandler).
  2. For each 50bp window:
     a. Extract alignment tensor from BAM (13 channels × max_reads × W).
     b. Extract 2500bp reference context → DNABERT-2 → mean-pool → 768-dim.
  3. (Optional) Fit PCA on training embeddings and transform all to K dims.
  4. Save each sample as a .pt file.

Usage:
  # Step 1: Generate raw embeddings + alignment tensors
  python generate_tensor_dataset.py --bam raw/NA12878.bam --vcf raw/variants.vcf.gz \\
        --fasta raw/hs37d5.fa --output data/tensor_dataset --sample NA12878

  # Step 2: Fit PCA and compress context vectors (after generating raw data)
  python generate_tensor_dataset.py --fit-pca data/tensor_dataset \\
        --n-components 8 --output data/tensor_dataset
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import VCFHandler, DeletionSize, Variant
from deepsv.data.genomic_context import GenomicContextExtractor, ReferenceGenome

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _collect_windows(
    variants: List[Variant],
    region_size: int = 50,
) -> List[Tuple[str, int, int]]:
    """Split each variant region into non-overlapping windows of *region_size*.

    Returns list of (chrom, start, end) tuples.
    """
    windows = []
    for v in variants:
        pos = v.start
        while pos < v.end:
            w_end = min(pos + region_size, v.end)
            if w_end - pos >= region_size // 2:  # skip tiny trailing windows
                windows.append((v.chrom, pos, w_end))
            pos += region_size
    return windows


def _generate_samples(
    bam_path: str,
    fasta_path: str,
    windows: List[Tuple[str, int, int]],
    label: int,
    output_dir: Path,
    max_reads: int = 100,
    use_dnabert: bool = True,
    dnabert_device: str = "cpu",
    limit: int = None,
):
    """Generate .pt samples for a set of windows.

    Each file contains the alignment tensor and the raw 768-dim context
    embedding (PCA is applied separately after all samples are generated).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialise context extractor (loads DNABERT-2 if needed)
    ctx_extractor = None
    if use_dnabert:
        ctx_extractor = GenomicContextExtractor(
            fasta_path=fasta_path,
            device=dnabert_device,
        )

    saved = 0
    with BAMHandler(bam_path) as bam:
        for idx, (chrom, start, end) in enumerate(windows):
            if limit and idx >= limit:
                logger.info("Reached limit of %d. Stopping.", limit)
                break

            width = end - start
            if width < 5:
                continue

            try:
                # Alignment tensor: (13, max_reads, W)
                aln_tensor = bam.get_alignment_tensor(
                    chrom, start, end, max_reads=max_reads
                )

                # Check that we actually have reads
                if aln_tensor.sum() == 0:
                    continue

                sample = {
                    "alignment": torch.from_numpy(aln_tensor),
                    "label": label,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                }

                # Context embedding: (768,) — raw, pre-PCA
                if ctx_extractor is not None:
                    center = (start + end) // 2
                    raw_emb = ctx_extractor.get_raw_embedding(chrom, center)
                    sample["context_raw"] = torch.from_numpy(raw_emb)

                # Save
                label_str = "del" if label == 1 else "non_del"
                fname = f"{label_str}_{chrom}_{start}_{end}.pt"
                torch.save(sample, output_dir / fname)
                saved += 1

                if saved % 200 == 0:
                    logger.info(
                        "  [%s] Saved %d / %d windows (current: %s:%d-%d)",
                        label_str, saved, len(windows), chrom, start, end,
                    )

            except Exception as e:
                logger.error(
                    "Error at %s:%d-%d: %s", chrom, start, end, e
                )

    if ctx_extractor is not None:
        ctx_extractor.close()

    logger.info("Saved %d %s samples to %s", saved,
                "deletion" if label == 1 else "non-deletion", output_dir)
    return saved


# ═══════════════════════════════════════════════════════════════════════════
# PCA fitting on pre-generated embeddings
# ═══════════════════════════════════════════════════════════════════════════

def fit_and_apply_pca(dataset_dir: Path, n_components: int = 8):
    """Fit PCA on training embeddings and add compressed context to all .pt files.

    1. Scan all .pt files for context_raw vectors.
    2. Fit PCA on the training split (chr 1–11).
    3. Transform ALL embeddings and add a `context` key.
    4. Save the PCA model alongside the dataset.
    """
    from sklearn.decomposition import PCA
    import joblib

    pt_files = sorted(dataset_dir.rglob("*.pt"))
    if not pt_files:
        logger.error("No .pt files found in %s", dataset_dir)
        return

    # Collect raw embeddings
    logger.info("Collecting raw embeddings from %d files …", len(pt_files))
    train_chroms = {str(c) for c in range(1, 12)}  # chr 1-11
    train_embeddings = []
    for f in pt_files:
        data = torch.load(f, weights_only=False)
        if "context_raw" not in data:
            continue
        if data["chrom"] in train_chroms:
            train_embeddings.append(data["context_raw"].numpy())

    if len(train_embeddings) == 0:
        logger.error("No training embeddings found.")
        return

    all_train = np.stack(train_embeddings)
    logger.info("Fitting PCA(%d) on %d training embeddings …",
                n_components, len(all_train))

    pca = PCA(n_components=n_components)
    pca.fit(all_train)
    explained = pca.explained_variance_ratio_.sum() * 100
    logger.info("PCA explained variance: %.1f%%", explained)

    # Save PCA model
    pca_path = dataset_dir / "pca_model.joblib"
    joblib.dump(pca, pca_path)
    logger.info("PCA model saved to %s", pca_path)

    # Transform and update all files
    logger.info("Applying PCA to all %d files …", len(pt_files))
    for f in pt_files:
        data = torch.load(f, weights_only=False)
        if "context_raw" not in data:
            continue
        raw = data["context_raw"].numpy().reshape(1, -1)
        reduced = pca.transform(raw).astype(np.float32).squeeze(0)
        data["context"] = torch.from_numpy(reduced)
        torch.save(data, f)

    logger.info("Done. All files updated with %d-dim context vectors.", n_components)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate multichannel tensor dataset for DeepSV3."
    )
    sub = parser.add_subparsers(dest="command")

    # --- generate sub-command ---
    gen = sub.add_parser("generate", help="Generate alignment tensors + DNABERT-2 embeddings")
    gen.add_argument("--bam", required=True, help="Path to BAM file")
    gen.add_argument("--vcf", required=True, help="Path to VCF file")
    gen.add_argument("--fasta", required=True, help="Path to reference FASTA")
    gen.add_argument("--output", required=True, help="Output directory")
    gen.add_argument("--sample", help="Sample ID for multi-sample VCF")
    gen.add_argument("--size", choices=["small", "medium", "large", "very_large", "all"],
                     default="all", help="Deletion size category")
    gen.add_argument("--chrom", help="Filter by chromosome")
    gen.add_argument("--max-reads", type=int, default=100, help="Max reads (H dim)")
    gen.add_argument("--region-size", type=int, default=50, help="Window width (W dim)")
    gen.add_argument("--max-length", type=int, default=10000,
                     help="Max variant length to process")
    gen.add_argument("--limit", type=int, help="Limit number of windows")
    gen.add_argument("--no-dnabert", action="store_true",
                     help="Skip DNABERT-2 context extraction (alignment only)")
    gen.add_argument("--device", default="cpu",
                     help="Device for DNABERT-2 inference (cpu/cuda)")

    # --- pca sub-command ---
    pca_cmd = sub.add_parser("pca", help="Fit PCA on raw embeddings and compress")
    pca_cmd.add_argument("--dataset", required=True, help="Dataset directory")
    pca_cmd.add_argument("--n-components", type=int, default=8,
                         help="PCA components (K)")

    args = parser.parse_args()

    if args.command == "generate":
        _run_generate(args)
    elif args.command == "pca":
        fit_and_apply_pca(Path(args.dataset), args.n_components)
    else:
        parser.print_help()


def _run_generate(args):
    """Execute the generate sub-command."""
    from deepsv.processing.refinement import BoundaryRefiner
    refiner = BoundaryRefiner()

    size_map = {
        "small": DeletionSize.SMALL,
        "medium": DeletionSize.MEDIUM,
        "large": DeletionSize.LARGE,
        "very_large": DeletionSize.VERY_LARGE,
    }

    # Load variants
    vcf_handler = VCFHandler(args.vcf)
    vcf_handler.load_variants(sample_id=args.sample)

    if args.size == "all":
        del_variants = []
        for cat in DeletionSize:
            del_variants.extend(vcf_handler.get_variants_by_size(cat))
    else:
        del_variants = vcf_handler.get_variants_by_size(size_map[args.size])

    logger.info("Total deletion variants: %d", len(del_variants))

    # Filter
    if args.chrom:
        del_variants = [v for v in del_variants if v.chrom == args.chrom]
    if args.max_length:
        del_variants = [v for v in del_variants if v.length <= args.max_length]

    logger.info("After filtering: %d deletion variants", len(del_variants))

    # Refine boundaries
    with BAMHandler(args.bam) as bam:
        refined = []
        for v in del_variants:
            try:
                refined.append(refiner.refine_boundaries(bam, v))
            except Exception:
                refined.append(v)
        del_variants = refined

    # Generate non-deletion anchor regions
    non_del_up = vcf_handler.get_non_deletion_regions(del_variants, "up")
    non_del_down = vcf_handler.get_non_deletion_regions(del_variants, "down")
    non_del_variants = non_del_up + non_del_down

    # Create windows
    del_windows = _collect_windows(del_variants, args.region_size)
    non_del_windows = _collect_windows(non_del_variants, args.region_size)

    logger.info("Windows — Deletion: %d, Non-deletion: %d",
                len(del_windows), len(non_del_windows))

    output = Path(args.output)

    # Generate deletion samples
    logger.info("═══ Generating deletion samples ═══")
    _generate_samples(
        args.bam, args.fasta, del_windows, label=1,
        output_dir=output / "deletion",
        max_reads=args.max_reads,
        use_dnabert=not args.no_dnabert,
        dnabert_device=args.device,
        limit=args.limit,
    )

    # Generate non-deletion samples
    logger.info("═══ Generating non-deletion samples ═══")
    _generate_samples(
        args.bam, args.fasta, non_del_windows, label=0,
        output_dir=output / "non_deletion",
        max_reads=args.max_reads,
        use_dnabert=not args.no_dnabert,
        dnabert_device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
