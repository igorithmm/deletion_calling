#!/usr/bin/env python3
"""Step 4 (training): train one of the two CADC models.

Models
──────
* **M0** (``--model cnn``) — :class:`ModernDeletionCNN`, image only.
  Standard accuracy-best training via :class:`ModelTrainer`.
  Does NOT require HyenaDNA embeddings.
* **M1** (``--model fused``) — :class:`FusedDeepSV` with FiLM modulation
  conditioned on a HyenaDNA embedding. Two-stage F1-best training via
  :class:`FiLMTrainer`.

Inputs
──────
* ``--manifest`` — CSV produced by ``generate_fused_dataset.py`` with columns
  ``image_path,chrom,position,label,length``.
* ``--embeddings`` — HDF5 file produced by ``precompute_hyenadna_embeddings.py``.
  Required only for ``--model fused``.
* ``--train-chroms`` / ``--val-chroms`` — chromosome split for train/val.

Output
──────
* Best checkpoint at ``--output``.
* Training log to stdout.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsv.data import FusedDataset
from deepsv.models import ModernDeletionCNN, FusedDeepSV
from deepsv.training import FiLMTrainer
from deepsv.training.trainer import ImageDataset, ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest loading + train/val split
# ─────────────────────────────────────────────────────────────────────────────


def load_manifest(path: str) -> List[dict]:
    """Read the manifest CSV into a list of row dicts."""
    rows: List[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["position"] = int(r["position"])
            r["label"] = int(r["label"])
            r["length"] = int(r["length"])
            rows.append(r)
    return rows


def split_rows(
    rows: List[dict],
    train_chroms: Sequence[str],
    val_chroms: Sequence[str],
) -> Tuple[List[dict], List[dict]]:
    """Split rows by chromosome. Chromosome names are matched literally."""
    train_set = set(train_chroms)
    val_set = set(val_chroms)
    train = [r for r in rows if r["chrom"] in train_set]
    val = [r for r in rows if r["chrom"] in val_set]
    return train, val


def manifest_to_lists(rows: List[dict]) -> Dict[str, list]:
    return {
        "image_paths": [r["image_path"] for r in rows],
        "labels": [r["label"] for r in rows],
        "chroms": [r["chrom"] for r in rows],
        "positions": [r["position"] for r in rows],
        "lengths": [r["length"] for r in rows],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build everything
# ─────────────────────────────────────────────────────────────────────────────


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_loaders(
    train_rows: List[dict],
    val_rows: List[dict],
    embeddings_h5: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, FusedDataset, FusedDataset]:
    """Build train + val loaders. Both datasets preload their own chroms."""
    tr = manifest_to_lists(train_rows)
    va = manifest_to_lists(val_rows)

    train_ds = FusedDataset(
        image_paths=tr["image_paths"],
        labels=tr["labels"],
        chroms=tr["chroms"],
        positions=tr["positions"],
        embeddings_h5=embeddings_h5,
        preload_chroms=sorted(set(tr["chroms"])),
        transform=IMAGE_TRANSFORM,
    )
    val_ds = FusedDataset(
        image_paths=va["image_paths"],
        labels=va["labels"],
        chroms=va["chroms"],
        positions=va["positions"],
        embeddings_h5=embeddings_h5,
        preload_chroms=sorted(set(va["chroms"])),
        transform=IMAGE_TRANSFORM,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    # Validation loader MUST be unshuffled so stratified metrics align.
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds


# ─────────────────────────────────────────────────────────────────────────────
# Per-model training paths
# ─────────────────────────────────────────────────────────────────────────────


def train_m0(
    train_rows, val_rows, args
) -> None:
    """Image-only baseline: uses :class:`ImageDataset` (no embeddings needed)."""
    tr = manifest_to_lists(train_rows)
    va = manifest_to_lists(val_rows)

    train_ds = ImageDataset(
        image_paths=tr["image_paths"],
        labels=tr["labels"],
        transform=IMAGE_TRANSFORM,
    )
    val_ds = ImageDataset(
        image_paths=va["image_paths"],
        labels=va["labels"],
        transform=IMAGE_TRANSFORM,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = ModernDeletionCNN(num_classes=2)
    trainer = ModelTrainer(model)
    trainer.setup_optimizer(learning_rate=args.lr_cnn, weight_decay=args.weight_decay)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_path=Path(args.output),
    )


def train_m1(train_rows, val_rows, args) -> None:
    """Fused CNN + FiLM (main contribution)."""
    train_loader, val_loader, _, val_ds = build_loaders(
        train_rows, val_rows, args.embeddings, args.batch_size, args.num_workers
    )

    val_va = manifest_to_lists(val_rows)
    validate_kwargs = {
        "sample_lengths": val_va["lengths"],
    }

    model = FusedDeepSV(embed_dim=args.embed_dim, num_classes=2)
    trainer = FiLMTrainer(model)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        stage_a_epochs=args.stage_a_epochs,
        lr_cnn=args.lr_cnn,
        lr_film=args.lr_film,
        weight_decay=args.weight_decay,
        save_path=Path(args.output),
        validate_kwargs=validate_kwargs,
    )




# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, help="CSV from generate_fused_dataset.py")
    p.add_argument(
        "--embeddings", default=None,
        help="HyenaDNA embeddings HDF5 (required only for --model fused)",
    )
    p.add_argument(
        "--model", choices=["cnn", "fused"], default="fused",
        help="cnn=M0 (image only), fused=M1 (image + HyenaDNA FiLM, default)",
    )
    p.add_argument("--output", required=True, help="Path to save the best checkpoint")
    p.add_argument(
        "--train-chroms", required=True,
        help="Comma-separated chromosomes for training (e.g. '1,2,3,...,11')",
    )
    p.add_argument(
        "--val-chroms", required=True,
        help="Comma-separated chromosomes for validation (e.g. '12,...,22')",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--stage-a-epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--lr-cnn", type=float, default=1e-4)
    p.add_argument("--lr-film", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading manifest from %s …", args.manifest)
    rows = load_manifest(args.manifest)
    train_chroms = [c.strip() for c in args.train_chroms.split(",") if c.strip()]
    val_chroms = [c.strip() for c in args.val_chroms.split(",") if c.strip()]

    train_rows, val_rows = split_rows(rows, train_chroms, val_chroms)
    logger.info(
        "Split: train=%d (%s) val=%d (%s)",
        len(train_rows), train_chroms, len(val_rows), val_chroms,
    )
    if not train_rows or not val_rows:
        raise SystemExit(
            "Empty train or val split. Check that --train-chroms / --val-chroms "
            "match the chrom values in the manifest."
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    if args.model == "fused" and not args.embeddings:
        raise SystemExit(
            "--embeddings is required for --model fused. "
            "Use --model cnn for image-only training."
        )

    if args.model == "cnn":
        train_m0(train_rows, val_rows, args)
    elif args.model == "fused":
        train_m1(train_rows, val_rows, args)


if __name__ == "__main__":
    main()
