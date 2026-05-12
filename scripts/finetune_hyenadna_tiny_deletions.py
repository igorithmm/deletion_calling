#!/usr/bin/env python3
"""Fine-tune a local HyenaDNA-tiny model for deletion-window classification.

The script is intentionally offline-friendly: it loads the model/tokenizer from
``--model-dir`` with ``local_files_only=True`` and never contacts Hugging Face.

Training data is built from the manifest written by
``scripts/generate_fused_dataset.py``:

* positives are rows with ``label=1``;
* negatives are the exact generated negatives in that manifest, preserving
  ``neg_type`` values such as ``anchor_up``, ``anchor_down``, ``regional`` and
  ``random``.

The resulting checkpoint is saved with ``save_pretrained`` and can be loaded
later from ``--output-dir/best_model`` using ``trust_remote_code=True``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyfaidx
import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceWindowSample:
    chrom: str
    position: int
    label: int
    length: int = 0
    neg_type: str = ""


def canonical_chrom(chrom: str) -> str:
    text = str(chrom)
    return text[3:] if text.startswith("chr") else text


def _chrom_sort_key(chrom: str) -> Tuple[int, object]:
    name = canonical_chrom(chrom)
    if name.isdigit():
        return (0, int(name))
    order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    return (0, order[name]) if name in order else (1, name)


def _resolve_fasta_chrom(available: Sequence[str], chrom: str) -> str:
    if chrom in available:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in available:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in available:
        return prefixed
    canon = canonical_chrom(chrom)
    for name in available:
        if canonical_chrom(name) == canon:
            return name
    raise KeyError(f"Chromosome {chrom!r} not found in FASTA.")


def parse_chrom_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_manifest(path: str, keep_all_samples: bool = False) -> List[dict]:
    """Read rows from a fused manifest.

    Mirrors ``scripts/train_fused_model.py``: by default rows are kept only
    when an image path segment looks like an HG/NA sample. This makes combined
    manifests produced from several samples behave the same across CNN/fused
    and sequence-only training.
    """
    rows: List[dict] = []
    n_total = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chrom", "position", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path!r} is missing required columns: {sorted(missing)}")
        for row in reader:
            n_total += 1
            if not keep_all_samples:
                image_path = row.get("image_path", "")
                path_parts = Path(image_path).parts
                if not any(p.startswith(("HG", "NA")) for p in path_parts):
                    continue
            row["position"] = int(row["position"])
            row["label"] = int(row["label"])
            row["length"] = int(row.get("length") or 0)
            row["neg_type"] = row.get("neg_type", "")
            rows.append(row)
    logger.info("Loaded %d / %d rows from %s", len(rows), n_total, path)
    return rows


def split_rows(
    rows: Sequence[dict],
    train_chroms: Sequence[str],
    val_chroms: Sequence[str],
) -> Tuple[List[dict], List[dict]]:
    """Split manifest rows by chromosome, matching train_fused_model.py literally."""
    train_set = set(train_chroms)
    val_set = set(val_chroms)
    train = [row for row in rows if row["chrom"] in train_set]
    val = [row for row in rows if row["chrom"] in val_set]
    return train, val


def rows_to_samples(rows: Sequence[dict], seed: int, max_samples: int = 0) -> List[SequenceWindowSample]:
    samples = [
        SequenceWindowSample(
            chrom=str(row["chrom"]),
            position=int(row["position"]),
            label=int(row["label"]),
            length=int(row.get("length", 0)),
            neg_type=str(row.get("neg_type", "")),
        )
        for row in rows
    ]
    if max_samples > 0 and len(samples) > max_samples:
        samples = random.Random(seed).sample(samples, max_samples)
    return samples


def sample_stats(samples: Sequence[SequenceWindowSample]) -> Dict[str, object]:
    labels = Counter(sample.label for sample in samples)
    neg_types = Counter(sample.neg_type or "positive" for sample in samples if sample.label == 0)
    chroms = sorted({sample.chrom for sample in samples}, key=_chrom_sort_key)
    stats = {
        "chroms": chroms,
        "n_samples": len(samples),
        "n_positive_samples": int(labels.get(1, 0)),
        "n_negative_samples": int(labels.get(0, 0)),
        "negative_types": dict(sorted(neg_types.items())),
    }
    return stats


class DeletionSequenceDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        samples: Sequence[SequenceWindowSample],
        tokenizer,
        sequence_bp: int,
        max_tokens: int,
        window_bp: int = 50,
    ) -> None:
        if sequence_bp <= 0:
            raise ValueError("sequence_bp must be > 0")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        self.fasta_path = str(fasta_path)
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.sequence_bp = int(sequence_bp)
        self.max_tokens = int(max_tokens)
        self.window_bp = int(window_bp)
        self.labels = [int(s.label) for s in self.samples]
        self._fasta: Optional[pyfaidx.Fasta] = None
        self._chrom_map: Optional[Dict[str, str]] = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fasta"] = None
        state["_chrom_map"] = None
        return state

    def __len__(self) -> int:
        return len(self.samples)

    def close(self) -> None:
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    def _ensure_fasta(self) -> pyfaidx.Fasta:
        if self._fasta is None:
            self._fasta = pyfaidx.Fasta(self.fasta_path, as_raw=True, sequence_always_upper=False)
        return self._fasta

    def _fasta_chrom(self, chrom: str) -> str:
        fasta = self._ensure_fasta()
        if self._chrom_map is None:
            self._chrom_map = {}
        if chrom not in self._chrom_map:
            self._chrom_map[chrom] = _resolve_fasta_chrom(list(fasta.keys()), chrom)
        return self._chrom_map[chrom]

    def _fetch_sequence(self, sample: SequenceWindowSample) -> str:
        fasta = self._ensure_fasta()
        chrom_seq = fasta[self._fasta_chrom(sample.chrom)]
        chrom_len = len(chrom_seq)
        left_len = self.sequence_bp // 2
        center_bp = int(sample.position) + max(0, self.window_bp // 2)
        start = center_bp - left_len
        end = start + self.sequence_bp
        fetch_start = max(0, start)
        fetch_end = min(chrom_len, end)
        seq = str(chrom_seq[fetch_start:fetch_end]).upper()
        left_pad = fetch_start - start
        right_pad = end - fetch_end
        if left_pad or right_pad:
            seq = ("N" * left_pad) + seq + ("N" * right_pad)
        return seq

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        seq = self._fetch_sequence(sample)
        encoded = self.tokenizer(
            seq,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_attention_mask=False,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(sample.label), dtype=torch.long)
        return item


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def _binary_metrics(labels: Sequence[int], probs: Sequence[float], preds: Sequence[int]) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    y_pred = np.asarray(preds, dtype=np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    return {
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
    }


def _class_weights(labels: Sequence[int], device: torch.device) -> Optional[torch.Tensor]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if np.any(counts == 0):
        logger.warning("Skipping class weights because class counts are %s", counts.tolist())
        return None
    weights = counts.sum() / (2.0 * counts)
    logger.info("Class counts=%s weights=%s", counts.tolist(), weights.tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    class_weights: Optional[torch.Tensor],
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    running_loss = 0.0
    labels_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[float] = []
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc="Training HyenaDNA", leave=False)
    for step, batch in enumerate(pbar):
        labels = batch["labels"].to(device, non_blocking=True)
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "labels"
        }
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()
        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item()) * grad_accum_steps
        probs = torch.softmax(logits.detach(), dim=-1)[:, 1]
        preds = torch.argmax(logits.detach(), dim=-1)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())
        pbar.set_postfix({"loss": running_loss / (step + 1)})

    metrics = _binary_metrics(labels_all, probs_all, preds_all)
    metrics["loss"] = running_loss / max(1, len(dataloader))
    return metrics


@torch.no_grad()
def evaluate(model, dataloader: DataLoader, device: torch.device, class_weights: Optional[torch.Tensor]) -> Dict[str, float]:
    model.eval()
    running_loss = 0.0
    labels_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[float] = []

    pbar = tqdm(dataloader, desc="Validating HyenaDNA", leave=False)
    for step, batch in enumerate(pbar):
        labels = batch["labels"].to(device, non_blocking=True)
        inputs = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
            if key != "labels"
        }
        outputs = model(**inputs)
        logits = outputs.logits
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
        running_loss += float(loss.item())
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)
        labels_all.extend(labels.cpu().tolist())
        preds_all.extend(preds.cpu().tolist())
        probs_all.extend(probs.cpu().tolist())

    metrics = _binary_metrics(labels_all, probs_all, preds_all)
    metrics["loss"] = running_loss / max(1, len(dataloader))
    return metrics


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default="models/hyenadna-tiny-16k-seqlen-d128-hf",
        help="Local HyenaDNA HF directory. No internet is used.",
    )
    parser.add_argument("--fasta", required=True, help="Reference FASTA path.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="CSV from scripts/generate_fused_dataset.py, or a combined manifest.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and metrics.")
    parser.add_argument(
        "--train-chroms",
        required=True,
        help="Comma-separated training chromosomes, matched literally against manifest chrom values.",
    )
    parser.add_argument(
        "--val-chroms",
        required=True,
        help="Comma-separated validation chromosomes, matched literally against manifest chrom values.",
    )
    parser.add_argument("--sequence-bp", type=int, default=16_000, help="Input DNA window length.")
    parser.add_argument("--max-tokens", type=int, default=16_000, help="Tokenizer max_length.")
    parser.add_argument(
        "--window-bp",
        type=int,
        default=50,
        help="Manifest window size. position + window_bp/2 is used as sequence center.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional debug cap after manifest split; <=0 means no cap.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=0,
        help="Optional debug cap after manifest split; <=0 means no cap.",
    )
    parser.add_argument(
        "--keep-all-samples",
        action="store_true",
        help="Do not filter manifest rows to image paths containing HG*/NA* sample segments.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA fp16 autocast.")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Train only the sequence-classification head.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    logger.info("Loading manifest from %s", args.manifest)
    rows = load_manifest(args.manifest, keep_all_samples=args.keep_all_samples)
    train_chroms = parse_chrom_list(args.train_chroms)
    val_chroms = parse_chrom_list(args.val_chroms)
    train_rows, val_rows = split_rows(rows, train_chroms, val_chroms)
    logger.info("Train chroms: %s", ",".join(train_chroms))
    logger.info("Val chroms: %s", ",".join(val_chroms))
    logger.info("Split rows: train=%d val=%d", len(train_rows), len(val_rows))
    if not train_rows or not val_rows:
        raise SystemExit(
            "Empty train or val split. Check that --train-chroms / --val-chroms "
            "match the chrom values in the manifest."
        )

    train_samples = rows_to_samples(train_rows, seed=args.seed, max_samples=args.max_train_samples)
    val_samples = rows_to_samples(val_rows, seed=args.seed + 1, max_samples=args.max_val_samples)
    train_stats = sample_stats(train_samples)
    val_stats = sample_stats(val_samples)
    logger.info("Train manifest stats: %s", train_stats)
    logger.info("Val manifest stats: %s", val_stats)

    logger.info("Checking FASTA chromosome names in %s", args.fasta)
    fasta = pyfaidx.Fasta(args.fasta, as_raw=True, sequence_always_upper=False)
    try:
        available = list(fasta.keys())
        for chrom in sorted({s.chrom for s in train_samples + val_samples}, key=_chrom_sort_key):
            resolved = _resolve_fasta_chrom(available, chrom)
            if resolved != chrom:
                logger.info("Manifest chrom %s maps to FASTA chrom %s", chrom, resolved)
    finally:
        fasta.close()

    logger.info("Loading local model/tokenizer from %s", args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    config = AutoConfig.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        num_labels=2,
        id2label={0: "non_deletion", 1: "deletion"},
        label2id={"non_deletion": 0, "deletion": 1},
        problem_type="single_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    )

    if args.freeze_backbone:
        logger.info("Freezing HyenaDNA backbone; training classification head only.")
        for name, param in model.named_parameters():
            if not name.startswith("score."):
                param.requires_grad = False

    model.to(device)

    train_ds = DeletionSequenceDataset(
        args.fasta,
        train_samples,
        tokenizer,
        sequence_bp=args.sequence_bp,
        max_tokens=args.max_tokens,
        window_bp=args.window_bp,
    )
    val_ds = DeletionSequenceDataset(
        args.fasta,
        val_samples,
        tokenizer,
        sequence_bp=args.sequence_bp,
        max_tokens=args.max_tokens,
        window_bp=args.window_bp,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_steps = max(1, args.epochs * updates_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    class_weights = None if args.no_class_weights else _class_weights(train_ds.labels, device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    metadata = {
        "model_dir": args.model_dir,
        "fasta": args.fasta,
        "manifest": args.manifest,
        "training_config": vars(args),
        "train_manifest_stats": train_stats,
        "val_manifest_stats": val_stats,
    }
    save_json(output_dir / "training_metadata.json", metadata)

    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    start_time = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_weights=class_weights,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=max(1, args.grad_accum_steps),
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = evaluate(model, val_loader, device, class_weights)
        logger.info(
            "Epoch %d/%d | %.1fs | train loss %.4f F1 %.3f AUC %.3f | "
            "val loss %.4f F1 %.3f AUC %.3f",
            epoch + 1,
            args.epochs,
            time.time() - epoch_start,
            train_metrics["loss"],
            train_metrics["f1"],
            train_metrics["auc"],
            val_metrics["loss"],
            val_metrics["f1"],
            val_metrics["auc"],
        )

        epoch_payload = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
        }
        save_json(output_dir / f"metrics_epoch_{epoch + 1:03d}.json", epoch_payload)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            best_dir = output_dir / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir, safe_serialization=False)
            tokenizer.save_pretrained(best_dir)
            save_json(
                best_dir / "training_summary.json",
                {
                    "best_epoch": epoch + 1,
                    "best_val_metrics": best_metrics,
                    "metadata": metadata,
                },
            )
            logger.info("Saved new best model to %s", best_dir)

    final_dir = output_dir / "last_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir, safe_serialization=False)
    tokenizer.save_pretrained(final_dir)
    save_json(
        output_dir / "final_metrics.json",
        {
            "best_val_metrics": best_metrics,
            "elapsed_minutes": (time.time() - start_time) / 60.0,
        },
    )
    train_ds.close()
    val_ds.close()
    logger.info("Done. Best F1=%.4f. Best checkpoint: %s", best_f1, output_dir / "best_model")


if __name__ == "__main__":
    main()
