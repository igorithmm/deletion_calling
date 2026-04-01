#!/usr/bin/env python3
"""Generate combined DeepSV2.5 dataset (RGB image tensors + DNABERT-2 embeddings).

This script produces .pt files containing:
  - image:      float32 tensor (3, H, W) — RGB image of alignment
  - context:    float32 vector (K,)       — PCA-reduced DNABERT-2 embedding
  - label:      int                        — 0 (non-deletion) or 1 (deletion)
  - chrom:      str
  - start:      int
  - end:        int

Workflow:
  1. Parse VCF to get deletion and non-deletion regions.
  2. For each window:
     a. Extract RGB image using DeepSV2 ImageGenerator.
     b. Convert image to (3, H, W) tensor.
     c. Extract 2500bp reference context → DNABERT-2 → mean-pool → 768-dim.
  3. (Optional) Fit PCA on embeddings and transform to K dims.
  4. Save each sample.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent.parent))

from deepsv.data.bam_handler import BAMHandler
from deepsv.data.vcf_handler import VCFHandler, DeletionSize, Variant
from deepsv.data.genomic_context import GenomicContextExtractor
from deepsv.visualization.image_generator import ImageGenerator
from deepsv.processing.refinement import BoundaryRefiner

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

# Helpers
def _collect_windows(variants: List[Variant], region_size: int = 50) -> List[Tuple[str, int, int]]:
    windows = []
    for v in variants:
        pos = v.start
        while pos < v.end:
            w_end = min(pos + region_size, v.end)
            if w_end - pos >= region_size // 2:
                windows.append((v.chrom, pos, w_end))
            pos += region_size
    return windows

def _generate_samples(bam_path, fasta_path, windows, label, output_dir, sample_name="sample", coloring_mode="standard", use_dnabert=True, dnabert_device="cpu", limit=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx_extractor = GenomicContextExtractor(fasta_path, device=dnabert_device) if use_dnabert else None
    image_gen = ImageGenerator(coloring_mode=coloring_mode)
    to_tensor = transforms.ToTensor()
    
    saved = 0
    with BAMHandler(bam_path) as bam:
        for idx, (chrom, start, end) in enumerate(windows):
            if limit and idx >= limit:
                break
            if end - start < 5:
                continue
            
            try:
                pileup_data = bam.get_pileup_data(chrom, start, end)
                clipping_data = bam.get_clipping_info(chrom, start, end)
                
                if not pileup_data:
                    continue
                    
                image = image_gen.generate_image(pileup_data, clipping_data, start, end - start)
                img_tensor = to_tensor(image)  # (3, H, W) float32
                
                sample = {
                    "image": img_tensor,
                    "label": label,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "sample": sample_name,
                }
                
                if ctx_extractor:
                    center = (start + end) // 2
                    raw_emb = ctx_extractor.get_raw_embedding(chrom, center)
                    sample["context_raw"] = torch.from_numpy(raw_emb)
                
                label_str = "del" if label == 1 else "non_del"
                fname = f"{sample_name}_{label_str}_{chrom}_{start}_{end}.pt"
                torch.save(sample, output_dir / fname)
                saved += 1
                
                if saved % 200 == 0:
                    logger.info("  [%s] Saved %d / %d windows (current: %s:%d-%d)", label_str, saved, len(windows), chrom, start, end)
            except Exception as e:
                logger.error("Error at %s:%d-%d: %s", chrom, start, end, e)
                
    if ctx_extractor:
        ctx_extractor.close()
    
    logger.info("Saved %d samples to %s", saved, output_dir)
    return saved


def fit_and_apply_pca(dataset_dir: Path, n_components: int = 8):
    from sklearn.decomposition import PCA
    import joblib

    pt_files = sorted(dataset_dir.rglob("*.pt"))
    if not pt_files:
        logger.error("No .pt files found in %s", dataset_dir)
        return

    logger.info("Collecting raw embeddings from %d files …", len(pt_files))
    train_chroms = {str(c) for c in range(1, 12)}
    train_embeddings = []
    
    all_indices = []
    # Process in chunks or individually, ensuring we don't keep images in memory
    for i, f in enumerate(tqdm(pt_files, desc="Collecting embeddings")):
        try:
            # We load the full file, but don't append it to any list
            data = torch.load(f, weights_only=False)
            if "context_raw" in data and data.get("chrom") in train_chroms:
                # We copy the vector and forget the 'data' dict (and its image)
                train_embeddings.append(data["context_raw"].numpy().copy())
            
            # Help garbage collector
            del data
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    if not train_embeddings:
        logger.error("No training embeddings found in chromosomes 1-11.")
        return

    all_train = np.stack(train_embeddings)
    logger.info("Fitting PCA(%d) on %d embeddings...", n_components, len(all_train))
    pca = PCA(n_components=n_components)
    pca.fit(all_train)
    joblib.dump(pca, dataset_dir / "pca_model.joblib")
    
    # We can clear train_embeddings now to save more space
    del train_embeddings
    del all_train

    logger.info("Applying PCA to all %d files …", len(pt_files))
    for f in tqdm(pt_files, desc="Updating files"):
        try:
            data = torch.load(f, weights_only=False)
            if "context_raw" in data:
                raw = data["context_raw"].numpy().reshape(1, -1)
                reduced = pca.transform(raw).astype(np.float32).squeeze(0)
                data["context"] = torch.from_numpy(reduced)
                torch.save(data, f)
            del data
        except Exception as e:
            logger.warning(f"Failed to update {f}: {e}")
            
    logger.info("Done. All files updated.")

def main():
    parser = argparse.ArgumentParser(description="Generate image+tensor dataset for DeepSV2.5.")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate RGB image tensors + DNABERT-2 embeddings")
    gen.add_argument("--bam", required=True)
    gen.add_argument("--vcf", required=True)
    gen.add_argument("--fasta", required=True)
    gen.add_argument("--output", required=True)
    gen.add_argument("--sample")
    gen.add_argument("--size", default="all", help="Deletion size category or comma-separated list: small,medium,large,very_large,all")
    gen.add_argument("--chrom")
    gen.add_argument("--region-size", type=int, default=50)
    gen.add_argument("--max-length", type=int, default=10000)
    gen.add_argument("--limit", type=int)
    gen.add_argument("--coloring-mode", choices=["standard", "kmer"], default="standard")
    gen.add_argument("--no-dnabert", action="store_true")
    gen.add_argument("--device", default="cpu")
    gen.add_argument("--exclude-sex", action="store_true", help="Exclude chromosomes X and Y")
    gen.add_argument("--balance", action="store_true", help="Balance non-deletion windows to match deletion windows")

    pca_cmd = sub.add_parser("pca", help="Fit PCA and compress context")
    pca_cmd.add_argument("--dataset", required=True)
    pca_cmd.add_argument("--n-components", type=int, default=8)

    args = parser.parse_args()

    if args.command == "generate":
        refiner = BoundaryRefiner()
        vcf_handler = VCFHandler(args.vcf)
        vcf_handler.load_variants(sample_id=args.sample)

        size_map = {"small": DeletionSize.SMALL, "medium": DeletionSize.MEDIUM, "large": DeletionSize.LARGE, "very_large": DeletionSize.VERY_LARGE}
        del_variants = []
        
        selected_sizes = args.size.split(",")
        if "all" in selected_sizes:
            for cat in DeletionSize:
                del_variants.extend(vcf_handler.get_variants_by_size(cat))
        else:
            for s in selected_sizes:
                if s in size_map:
                    del_variants.extend(vcf_handler.get_variants_by_size(size_map[s]))
                else:
                    logger.warning(f"Unknown size category: {s}")

        if args.chrom: del_variants = [v for v in del_variants if v.chrom == args.chrom]
        if args.max_length: del_variants = [v for v in del_variants if v.length <= args.max_length]

        if args.exclude_sex:
            sex_chroms = {"X", "Y", "chrX", "chrY"}
            initial_count = len(del_variants)
            del_variants = [v for v in del_variants if v.chrom not in sex_chroms]
            logger.info(f"Excluded {initial_count - len(del_variants)} sex chromosome variants.")

        with BAMHandler(args.bam) as bam:
            # ... boundaries ...
            refined = []
            for v in tqdm(del_variants, desc="Refining boundaries"):
                try:
                    refined.append(refiner.refine_boundaries(bam, v))
                except Exception:
                    refined.append(v)
            del_variants = refined

        non_del_variants = vcf_handler.get_non_deletion_regions(del_variants, "up") + vcf_handler.get_non_deletion_regions(del_variants, "down")

        del_windows = _collect_windows(del_variants, args.region_size)
        non_del_windows = _collect_windows(non_del_variants, args.region_size)

        if args.balance:
            import random
            random.seed(42)
            if len(non_del_windows) > len(del_windows):
                random.shuffle(non_del_windows)
                non_del_windows = non_del_windows[:len(del_windows)]
                logger.info(f"Balanced: Using {len(non_del_windows)} non-deletion windows to match {len(del_windows)} deletions.")
            elif len(del_windows) > len(non_del_windows):
                random.shuffle(del_windows)
                del_windows = del_windows[:len(non_del_windows)]
                logger.info(f"Balanced: Using {len(del_windows)} deletion windows to match {len(non_del_windows)} non-deletions.")

        out = Path(args.output)
        sample_name = args.sample if args.sample else "sample"
        _generate_samples(args.bam, args.fasta, del_windows, 1, out / "deletion", sample_name, args.coloring_mode, not args.no_dnabert, args.device, args.limit)
        _generate_samples(args.bam, args.fasta, non_del_windows, 0, out / "non_deletion", sample_name, args.coloring_mode, not args.no_dnabert, args.device, args.limit)

    elif args.command == "pca":
        fit_and_apply_pca(Path(args.dataset), args.n_components)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
