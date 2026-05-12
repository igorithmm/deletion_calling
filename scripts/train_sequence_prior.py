#!/usr/bin/env python3
"""Train a sequence-only deletion prior from reference embeddings and SV VCF.

The model learns a population deletion prior:

    reference H5 embeddings + known DEL regions from a 1000G-style VCF
    -> P(reference window lies in a deletion-prone region)

This is intentionally separate from the image CNN. At inference time the prior
can be mixed with an image/fused model via logit addition.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsv.data.sequence_prior_dataset import (  # noqa: E402
    SequencePriorDataset,
    build_sequence_prior_samples,
    canonical_chrom,
    chrom_sort_key,
    list_h5_embedding_chroms,
    parse_chrom_list,
)
from deepsv.models import SequenceDeletionPrior  # noqa: E402
from deepsv.training import SequencePriorTrainer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _default_val_chrom(selected_chroms: List[str]) -> List[str]:
    for chrom in selected_chroms:
        if canonical_chrom(chrom) == "22":
            return [chrom]
    return [selected_chroms[-1]]


def _resolve_split(args: argparse.Namespace) -> tuple[List[str], List[str], List[str]]:
    h5_chroms = list_h5_embedding_chroms(args.embeddings)
    selected = parse_chrom_list(args.chroms) or h5_chroms
    selected = sorted(selected, key=chrom_sort_key)

    val_chroms = parse_chrom_list(args.val_chroms) or _default_val_chrom(selected)
    val_set = {canonical_chrom(c) for c in val_chroms}

    train_chroms = parse_chrom_list(args.train_chroms)
    if train_chroms is None:
        train_chroms = [c for c in selected if canonical_chrom(c) not in val_set]

    if not train_chroms:
        raise ValueError("No training chromosomes selected.")
    if not val_chroms:
        raise ValueError("No validation chromosomes selected.")
    return selected, sorted(train_chroms, key=chrom_sort_key), sorted(val_chroms, key=chrom_sort_key)


def _positive_sample_cap(value: int) -> Optional[int]:
    return None if value <= 0 else value


def build_loader(
    embeddings_h5: str,
    vcf_path: str,
    chroms: List[str],
    args: argparse.Namespace,
    seed: int,
    shuffle: bool,
) -> tuple[DataLoader, dict]:
    samples, stats = build_sequence_prior_samples(
        embeddings_h5=embeddings_h5,
        vcf_path=vcf_path,
        chroms=chroms,
        positive_mode=args.positive_mode,
        min_length=args.min_length,
        max_length=None if args.max_length <= 0 else args.max_length,
        min_af=args.min_af,
        require_pass=args.require_pass,
        positive_stride_windows=args.positive_stride_windows,
        max_positive_windows_per_interval=args.max_positive_windows_per_interval,
        max_positive_samples=_positive_sample_cap(args.max_positive_samples),
        negative_ratio=args.negative_ratio,
        negative_margin_bp=args.negative_margin_bp,
        seed=seed,
        vcf_limit=None if args.vcf_limit <= 0 else args.vcf_limit,
    )
    dataset = SequencePriorDataset(
        samples=samples,
        embeddings_h5=embeddings_h5,
        context_radius=args.context_radius,
        preload_chroms=chroms if args.preload_chroms else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings", required=True, help="Reference embedding HDF5")
    p.add_argument(
        "--vcf",
        required=True,
        help="1000G-style SV VCF.gz with DEL records, e.g. raw/ALL...vcf.gz",
    )
    p.add_argument("--output", required=True, help="Path to save best checkpoint")
    p.add_argument(
        "--chroms",
        default=None,
        help="Optional comma-separated chrom subset. Defaults to all H5 datasets.",
    )
    p.add_argument(
        "--train-chroms",
        default=None,
        help="Comma-separated training chromosomes. Defaults to chroms minus val.",
    )
    p.add_argument(
        "--val-chroms",
        default=None,
        help="Comma-separated validation chromosomes. Defaults to chr22/22 if present.",
    )
    p.add_argument(
        "--positive-mode",
        choices=["center", "breakpoints", "breakpoints_center", "span"],
        default="span",
        help="How to turn each DEL interval into positive H5 windows.",
    )
    p.add_argument("--min-length", type=int, default=50)
    p.add_argument(
        "--max-length",
        type=int,
        default=10_000,
        help="Skip longer DELs; set <=0 to disable.",
    )
    p.add_argument("--min-af", type=float, default=None)
    p.add_argument("--require-pass", action="store_true")
    p.add_argument("--positive-stride-windows", type=int, default=1)
    p.add_argument("--max-positive-windows-per-interval", type=int, default=8)
    p.add_argument(
        "--max-positive-samples",
        type=int,
        default=300_000,
        help="Cap positives after sampling; set <=0 to disable.",
    )
    p.add_argument("--negative-ratio", type=float, default=1.0)
    p.add_argument("--negative-margin-bp", type=int, default=1_000)
    p.add_argument("--context-radius", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-patience", type=int, default=3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--preload-chroms",
        action="store_true",
        help="Preload selected chromosome embeddings into RAM.",
    )
    p.add_argument(
        "--vcf-limit",
        type=int,
        default=0,
        help="Debug cap on loaded DEL intervals; <=0 means no cap.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    selected, train_chroms, val_chroms = _resolve_split(args)
    logger.info("Selected H5 chroms: %s", ",".join(selected))
    logger.info("Train chroms: %s", ",".join(train_chroms))
    logger.info("Val chroms: %s", ",".join(val_chroms))

    train_loader, train_stats = build_loader(
        args.embeddings,
        args.vcf,
        train_chroms,
        args,
        seed=args.seed,
        shuffle=True,
    )
    val_loader, val_stats = build_loader(
        args.embeddings,
        args.vcf,
        val_chroms,
        args,
        seed=args.seed + 1,
        shuffle=False,
    )
    logger.info("Train sampling stats: %s", train_stats)
    logger.info("Val sampling stats: %s", val_stats)

    model = SequenceDeletionPrior(
        embed_dim=int(train_stats["embed_dim"]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        num_classes=2,
    )
    trainer = SequencePriorTrainer(model)
    trainer.setup_optimizer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=Path(args.output),
        checkpoint_metadata={
            "training_config": {
                "context_radius": args.context_radius,
                "positive_mode": args.positive_mode,
                "min_length": args.min_length,
                "max_length": args.max_length,
                "min_af": args.min_af,
                "negative_ratio": args.negative_ratio,
                "negative_margin_bp": args.negative_margin_bp,
                "train_chroms": train_chroms,
                "val_chroms": val_chroms,
                "vcf": args.vcf,
                "embeddings": args.embeddings,
            },
            "train_sampling_stats": train_stats,
            "val_sampling_stats": val_stats,
        },
        max_grad_norm=args.max_grad_norm,
    )


if __name__ == "__main__":
    main()
