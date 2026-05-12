#!/usr/bin/env python3
"""End-to-end CNN + HyenaDNA training for deletion detection.

Input comes from ``generate_fused_dataset.py`` manifests:
PNG pileup image + FASTA sequence window around the same 50 bp manifest
position -> joint deletion / non-deletion classifier.
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
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepsv.models import ModernDeletionCNN  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@dataclass(frozen=True)
class ManifestSample:
    image_path: str
    chrom: str
    position: int
    label: int
    length: int = 0
    neg_type: str = ""


def canonical_chrom(chrom: str) -> str:
    text = str(chrom)
    return text[3:] if text.startswith("chr") else text


def chrom_sort_key(chrom: str) -> Tuple[int, object]:
    name = canonical_chrom(chrom)
    if name.isdigit():
        return (0, int(name))
    order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    return (0, order[name]) if name in order else (1, name)


def parse_chrom_list(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_fasta_chrom(available: Sequence[str], chrom: str) -> str:
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


def load_manifest(path: str, keep_all_samples: bool = False) -> List[dict]:
    rows: List[dict] = []
    n_total = 0
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "chrom", "position", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest {path!r} is missing required columns: {sorted(missing)}")
        for row in reader:
            n_total += 1
            if not keep_all_samples:
                parts = Path(row["image_path"]).parts
                if not any(part.startswith(("HG", "NA")) for part in parts):
                    continue
            row["position"] = int(row["position"])
            row["label"] = int(row["label"])
            row["length"] = int(row.get("length") or 0)
            row["neg_type"] = row.get("neg_type", "")
            rows.append(row)
    logger.info("Loaded %d / %d rows from %s", len(rows), n_total, path)
    return rows


def split_rows(rows: Sequence[dict], train_chroms: Sequence[str], val_chroms: Sequence[str]):
    train_set = set(train_chroms)
    val_set = set(val_chroms)
    return [r for r in rows if r["chrom"] in train_set], [r for r in rows if r["chrom"] in val_set]


def rows_to_samples(rows: Sequence[dict], seed: int, max_samples: int = 0) -> List[ManifestSample]:
    samples = [
        ManifestSample(
            image_path=str(r["image_path"]),
            chrom=str(r["chrom"]),
            position=int(r["position"]),
            label=int(r["label"]),
            length=int(r.get("length", 0)),
            neg_type=str(r.get("neg_type", "")),
        )
        for r in rows
    ]
    if max_samples > 0 and len(samples) > max_samples:
        samples = random.Random(seed).sample(samples, max_samples)
    return samples


def sample_stats(samples: Sequence[ManifestSample]) -> Dict[str, object]:
    labels = Counter(s.label for s in samples)
    neg_types = Counter(s.neg_type or "unknown" for s in samples if s.label == 0)
    return {
        "chroms": sorted({s.chrom for s in samples}, key=chrom_sort_key),
        "n_samples": len(samples),
        "n_positive_samples": int(labels.get(1, 0)),
        "n_negative_samples": int(labels.get(0, 0)),
        "negative_types": dict(sorted(neg_types.items())),
    }


class E2EManifestDataset(Dataset):
    def __init__(
        self,
        fasta_path: str,
        samples: Sequence[ManifestSample],
        tokenizer,
        sequence_bp: int,
        max_tokens: int,
        window_bp: int,
        transform=IMAGE_TRANSFORM,
    ) -> None:
        self.fasta_path = str(fasta_path)
        self.samples = list(samples)
        self.tokenizer = tokenizer
        self.sequence_bp = int(sequence_bp)
        self.max_tokens = int(max_tokens)
        self.window_bp = int(window_bp)
        self.transform = transform
        self.labels = [int(s.label) for s in samples]
        self._fasta: Optional[pyfaidx.Fasta] = None
        self._chrom_map: Dict[str, str] = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fasta"] = None
        state["_chrom_map"] = {}
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
        if chrom not in self._chrom_map:
            self._chrom_map[chrom] = resolve_fasta_chrom(list(fasta.keys()), chrom)
        return self._chrom_map[chrom]

    def _fetch_sequence(self, sample: ManifestSample) -> str:
        fasta = self._ensure_fasta()
        chrom_seq = fasta[self._fasta_chrom(sample.chrom)]
        chrom_len = len(chrom_seq)
        center_bp = int(sample.position) + max(0, self.window_bp // 2)
        start = center_bp - self.sequence_bp // 2
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
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        encoded = self.tokenizer(
            self._fetch_sequence(sample),
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=self.max_tokens,
            return_attention_mask=False,
            return_tensors="pt",
        )
        return {
            "image": image,
            "input_ids": encoded["input_ids"].squeeze(0),
            "label": torch.tensor(int(sample.label), dtype=torch.long),
        }


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "image": torch.stack([x["image"] for x in batch], dim=0),
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
    }


class EndToEndHyenaCNN(nn.Module):
    def __init__(
        self,
        hyena_model,
        cnn: Optional[ModernDeletionCNN],
        hyena_dim: int,
        fusion_hidden_dim: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.cnn = cnn if cnn is not None else ModernDeletionCNN(num_classes=2)
        self.hyena = hyena_model
        self.hyena_norm = nn.LayerNorm(hyena_dim)
        self.fusion = nn.Sequential(
            nn.Linear(ModernDeletionCNN.CLASSIFIER_FEATURES + hyena_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_hidden_dim, 2),
        )

    def set_hyena_trainable(self, trainable: bool) -> None:
        for p in self.hyena.parameters():
            p.requires_grad = trainable

    def set_cnn_trainable(self, trainable: bool) -> None:
        for p in self.cnn.parameters():
            p.requires_grad = trainable

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        _, cnn_features = self.cnn._forward_features_with_hooks(image, return_classifier_features=True)
        hyena_outputs = self.hyena(input_ids=input_ids, return_dict=True)
        hyena_features = hyena_outputs.last_hidden_state.mean(dim=1)
        hyena_features = self.hyena_norm(hyena_features.float())
        return self.fusion(torch.cat((cnn_features, hyena_features), dim=-1))


def load_torch_state_dict(path: str) -> Dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint {path!r} did not contain a state_dict.")
    return {k.removeprefix("module."): v for k, v in state.items() if isinstance(k, str) and torch.is_tensor(v)}


def load_cnn_backbone(path: Optional[str]) -> Optional[ModernDeletionCNN]:
    if not path:
        return None
    cnn = ModernDeletionCNN(num_classes=2)
    state = load_torch_state_dict(path)
    if any(k.startswith("cnn.") for k in state):
        state = {k.removeprefix("cnn."): v for k, v in state.items() if k.startswith("cnn.")}
    missing, unexpected = cnn.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Could not initialize CNN from {path!r}; missing keys: {missing}")
    if unexpected:
        logger.warning("Ignoring unexpected CNN checkpoint keys from %s: %s", path, unexpected)
    logger.info("Initialized CNN backbone from %s", path)
    return cnn


def load_hyena_backbone(model_dir: str, init_checkpoint: Optional[str] = None):
    load_dir = init_checkpoint or model_dir
    config = AutoConfig.from_pretrained(load_dir, trust_remote_code=True, local_files_only=True)
    if init_checkpoint:
        logger.info("Loading fine-tuned HyenaDNA classifier from %s", init_checkpoint)
        classifier = AutoModelForSequenceClassification.from_pretrained(
            init_checkpoint,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
        )
        if not hasattr(classifier, "hyena"):
            raise TypeError(f"{init_checkpoint!r} does not expose a .hyena backbone.")
        return classifier.hyena, config

    logger.info("Loading base HyenaDNA from %s", model_dir)
    return (
        AutoModel.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
        ),
        config,
    )


def class_weights(labels: Sequence[int], device: torch.device) -> Optional[torch.Tensor]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=2)
    if np.any(counts == 0):
        logger.warning("Skipping class weights because class counts are %s", counts.tolist())
        return None
    weights = counts.sum() / (2.0 * counts)
    logger.info("Class counts=%s weights=%s", counts.tolist(), weights.tolist())
    return torch.tensor(weights, dtype=torch.float32, device=device)


def binary_metrics(labels, probs, preds) -> Dict[str, float]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probs, dtype=np.float64)
    y_pred = np.asarray(preds, dtype=np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    accuracy = 100.0 * float((y_true == y_pred).mean()) if y_true.size else 0.0
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.5
    return {"accuracy": accuracy, "precision": float(precision), "recall": float(recall), "f1": float(f1), "auc": auc}


def build_optimizer(model: EndToEndHyenaCNN, args: argparse.Namespace) -> torch.optim.Optimizer:
    groups = [
        {"params": [p for p in model.cnn.parameters() if p.requires_grad], "lr": args.lr_cnn, "weight_decay": args.weight_decay},
        {"params": [p for p in model.hyena.parameters() if p.requires_grad], "lr": args.lr_hyena, "weight_decay": args.hyena_weight_decay},
        {
            "params": [p for m in (model.hyena_norm, model.fusion) for p in m.parameters() if p.requires_grad],
            "lr": args.lr_fusion,
            "weight_decay": args.weight_decay,
        },
    ]
    return torch.optim.AdamW([g for g in groups if g["params"]])


def run_epoch(model, dataloader, device, loss_weights, amp_dtype, optimizer=None, scheduler=None, scaler=None, grad_accum_steps=1, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    labels_all: List[int] = []
    probs_all: List[float] = []
    preds_all: List[int] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    desc = "Training E2E" if training else "Validating E2E"
    pbar = tqdm(dataloader, desc=desc, leave=False)
    context = torch.enable_grad() if training else torch.no_grad()
    use_amp = amp_dtype is not None and device.type == "cuda"
    with context:
        for step, batch in enumerate(pbar):
            image = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(image, input_ids)
                loss = nn.functional.cross_entropy(logits, labels, weight=loss_weights)
                step_loss = loss / grad_accum_steps if training else loss
            if training:
                scaler.scale(step_loss).backward()
                if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                    if max_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item())
            probs = torch.softmax(logits.detach(), dim=-1)[:, 1]
            preds = torch.argmax(logits.detach(), dim=-1)
            labels_all.extend(labels.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            pbar.set_postfix({"loss": running_loss / (step + 1)})
    metrics = binary_metrics(labels_all, probs_all, preds_all)
    metrics["loss"] = running_loss / max(1, len(dataloader))
    return metrics


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def save_checkpoint(path: Path, model: EndToEndHyenaCNN, tokenizer, args, metrics, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "metrics": metrics, "config": vars(args)}, path)
    tokenizer.save_pretrained(path.parent / "tokenizer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--fasta", required=True)
    p.add_argument("--hyena-model-dir", default="models/hyenadna-tiny-16k-seqlen-d128-hf")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-chroms", required=True)
    p.add_argument("--val-chroms", required=True)
    p.add_argument("--init-cnn-checkpoint", default=None)
    p.add_argument("--init-hyena-checkpoint", default=None, help="Optional sequence-only save_pretrained dir")
    p.add_argument("--sequence-bp", type=int, default=16_000)
    p.add_argument("--max-tokens", type=int, default=16_000)
    p.add_argument("--window-bp", type=int, default=50)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--freeze-hyena-epochs", type=int, default=1)
    p.add_argument("--freeze-cnn-epochs", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum-steps", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr-cnn", type=float, default=1e-4)
    p.add_argument("--lr-hyena", type=float, default=1e-5)
    p.add_argument("--lr-fusion", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hyena-weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--fusion-hidden-dim", type=int, default=256)
    p.add_argument("--fusion-dropout", type=float, default=0.2)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--keep-all-samples", action="store_true")
    p.add_argument("--no-class-weights", action="store_true")
    return p.parse_args()


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
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "none": None}[args.amp_dtype]
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_dtype == torch.float16))

    rows = load_manifest(args.manifest, keep_all_samples=args.keep_all_samples)
    train_rows, val_rows = split_rows(rows, parse_chrom_list(args.train_chroms), parse_chrom_list(args.val_chroms))
    if not train_rows or not val_rows:
        raise SystemExit("Empty train or val split. Check manifest chrom names.")
    train_samples = rows_to_samples(train_rows, args.seed, args.max_train_samples)
    val_samples = rows_to_samples(val_rows, args.seed + 1, args.max_val_samples)
    train_stats = sample_stats(train_samples)
    val_stats = sample_stats(val_samples)
    logger.info("Train stats: %s", train_stats)
    logger.info("Val stats: %s", val_stats)

    fasta = pyfaidx.Fasta(args.fasta, as_raw=True, sequence_always_upper=False)
    try:
        available = list(fasta.keys())
        for chrom in sorted({s.chrom for s in train_samples + val_samples}, key=chrom_sort_key):
            resolved = resolve_fasta_chrom(available, chrom)
            if resolved != chrom:
                logger.info("Manifest chrom %s maps to FASTA chrom %s", chrom, resolved)
    finally:
        fasta.close()

    tokenizer_dir = args.init_hyena_checkpoint or args.hyena_model_dir
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="left",
    )
    hyena, hyena_config = load_hyena_backbone(args.hyena_model_dir, args.init_hyena_checkpoint)
    model = EndToEndHyenaCNN(
        hyena_model=hyena,
        cnn=load_cnn_backbone(args.init_cnn_checkpoint),
        hyena_dim=int(getattr(hyena_config, "d_model", 128)),
        fusion_hidden_dim=args.fusion_hidden_dim,
        dropout_rate=args.fusion_dropout,
    ).to(device)

    train_ds = E2EManifestDataset(args.fasta, train_samples, tokenizer, args.sequence_bp, args.max_tokens, args.window_bp)
    val_ds = E2EManifestDataset(args.fasta, val_samples, tokenizer, args.sequence_bp, args.max_tokens, args.window_bp)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=collate_batch)

    loss_weights = None if args.no_class_weights else class_weights(train_ds.labels, device)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    warmup_steps = int(args.epochs * updates_per_epoch * args.warmup_ratio)
    save_json(output_dir / "training_metadata.json", {"training_config": vars(args), "train_stats": train_stats, "val_stats": val_stats})

    best_f1 = -1.0
    best_metrics: Dict[str, float] = {}
    previous_state = None
    optimizer = None
    scheduler = None
    start_time = time.time()
    for epoch in range(args.epochs):
        trainable_state = (epoch >= args.freeze_hyena_epochs, epoch >= args.freeze_cnn_epochs)
        if trainable_state != previous_state:
            hyena_trainable, cnn_trainable = trainable_state
            logger.info("Epoch %d: hyena_trainable=%s cnn_trainable=%s", epoch + 1, hyena_trainable, cnn_trainable)
            model.set_hyena_trainable(hyena_trainable)
            model.set_cnn_trainable(cnn_trainable)
            optimizer = build_optimizer(model, args)
            remaining_updates = max(1, (args.epochs - epoch) * updates_per_epoch)
            scheduler = get_linear_schedule_with_warmup(optimizer, min(warmup_steps, remaining_updates), remaining_updates)
            previous_state = trainable_state

        assert optimizer is not None and scheduler is not None
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, device, loss_weights, amp_dtype, optimizer, scheduler, scaler, max(1, args.grad_accum_steps), args.max_grad_norm)
        val_metrics = run_epoch(model, val_loader, device, loss_weights, amp_dtype)
        logger.info(
            "Epoch %d/%d | %.1fs | train loss %.4f F1 %.3f AUC %.3f | val loss %.4f F1 %.3f AUC %.3f",
            epoch + 1, args.epochs, time.time() - t0,
            train_metrics["loss"], train_metrics["f1"], train_metrics["auc"],
            val_metrics["loss"], val_metrics["f1"], val_metrics["auc"],
        )
        save_json(output_dir / f"metrics_epoch_{epoch + 1:03d}.json", {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_metrics = val_metrics
            save_checkpoint(output_dir / "best_model.pth", model, tokenizer, args, best_metrics, epoch + 1)
            logger.info("Saved new best checkpoint to %s", output_dir / "best_model.pth")

    save_checkpoint(output_dir / "last_model.pth", model, tokenizer, args, best_metrics, args.epochs)
    save_json(output_dir / "final_metrics.json", {"best_val_metrics": best_metrics, "elapsed_minutes": (time.time() - start_time) / 60.0})
    train_ds.close()
    val_ds.close()
    logger.info("Done. Best F1=%.4f", best_f1)


if __name__ == "__main__":
    main()
