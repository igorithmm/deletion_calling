#!/usr/bin/env python3
"""Step 4 (inference): run a trained model over a manifest and write predictions.

Reads:
  * Manifest CSV from ``generate_fused_dataset.py``
  * Trained checkpoint from ``train_fused_model.py``
  * HyenaDNA embeddings HDF5 (only needed for ``--model fused``)

Writes a predictions CSV with one row per input window:

    image_path, chrom, position, label, length,
    prob_deletion, predicted_class

Optionally, neighbouring positive windows are merged into deletion calls
written to a minimal VCF (``--vcf-out``).

Examples
--------
    # Inference with the fused (M1) model
    python3 scripts/call_fused_deletions.py \\
        --manifest data/fused/NA12878/manifest.csv \\
        --model fused \\
        --checkpoint models/fused_best.pth \\
        --embeddings data/hyenadna_embeddings.h5 \\
        --predictions-out runs/NA12878_predictions.csv \\
        --vcf-out runs/NA12878.vcf

    # Inference with the CNN-only (M0) model — no embeddings needed
    python3 scripts/call_fused_deletions.py \\
        --manifest data/fused/NA12878/manifest.csv \\
        --model cnn \\
        --checkpoint models/image_only_model.pth \\
        --predictions-out runs/NA12878_cnn_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepsv.inference import FusedPredictor
from deepsv.inference.predictor import DeletionPredictor
from deepsv.models import ModernDeletionCNN, FusedDeepSV

# Reuse manifest/checkpoint helpers from the training script.
from train_fused_model import load_manifest, load_torch_state_dict  # type: ignore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


WINDOW_BP = 50  # must match generation


# ─────────────────────────────────────────────────────────────────────────────
# Inference paths per model
# ─────────────────────────────────────────────────────────────────────────────


def predict_m0(
    rows: List[dict], checkpoint: str, threshold: float, batch_size: int
) -> List[Tuple[float, int]]:
    """Image-only inference."""
    model = ModernDeletionCNN(num_classes=2)
    state = load_torch_state_dict(checkpoint)
    model.load_state_dict(state)
    predictor = DeletionPredictor(model=model, threshold=threshold)
    image_paths = [Path(r["image_path"]) for r in rows]
    return predictor.predict_batch(image_paths)


def predict_m1(
    rows: List[dict],
    checkpoint: str,
    embeddings_h5: str,
    threshold: float,
    batch_size: int,
    embed_dim: int,
    fusion_mode: str,
    context_hidden_dim: int,
    context_dropout: float,
) -> List[Tuple[float, int]]:
    """Fused (image + embedding) inference."""
    model = FusedDeepSV(
        embed_dim=embed_dim,
        num_classes=2,
        fusion_mode=fusion_mode,
        context_hidden_dim=context_hidden_dim,
        context_dropout_rate=context_dropout,
    )
    state = load_torch_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(
            "Checkpoint is missing %d model keys; newly initialized keys stay at "
            "their default values. First few missing keys: %s",
            len(missing),
            missing[:8],
        )
    if unexpected:
        logger.warning(
            "Checkpoint has %d unexpected keys ignored by this model. First few: %s",
            len(unexpected),
            unexpected[:8],
        )

    chroms_in_manifest = sorted({r["chrom"] for r in rows})
    with FusedPredictor(
        model=model,
        embeddings_h5=embeddings_h5,
        threshold=threshold,
        preload_chroms=chroms_in_manifest,
    ) as predictor:
        return predictor.predict_batch(
            image_paths=[Path(r["image_path"]) for r in rows],
            chroms=[r["chrom"] for r in rows],
            positions=[r["position"] for r in rows],
            batch_size=batch_size,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Merge adjacent positive windows into VCF calls
# ─────────────────────────────────────────────────────────────────────────────


def merge_to_calls(rows: List[dict], preds: List[Tuple[float, int]]) -> List[dict]:
    """Merge adjacent positive windows on the same chromosome into calls.

    Windows are considered adjacent if they share a chromosome and start
    positions differ by exactly ``WINDOW_BP``. A call is emitted with the
    span from the first window's start to the last window's end + WINDOW_BP.
    """
    paired = [{**r, "prob": p, "pred": c} for r, (p, c) in zip(rows, preds)]
    paired.sort(key=lambda r: (r["chrom"], r["position"]))

    calls: List[dict] = []
    cur: Optional[dict] = None
    for r in paired:
        if r["pred"] != 1:
            if cur is not None:
                calls.append(cur)
                cur = None
            continue
        if cur is None:
            cur = {
                "chrom": r["chrom"],
                "start": r["position"],
                "end": r["position"] + WINDOW_BP,
                "max_prob": r["prob"],
                "n_windows": 1,
            }
        elif r["chrom"] == cur["chrom"] and r["position"] == cur["end"]:
            cur["end"] = r["position"] + WINDOW_BP
            cur["max_prob"] = max(cur["max_prob"], r["prob"])
            cur["n_windows"] += 1
        else:
            calls.append(cur)
            cur = {
                "chrom": r["chrom"],
                "start": r["position"],
                "end": r["position"] + WINDOW_BP,
                "max_prob": r["prob"],
                "n_windows": 1,
            }
    if cur is not None:
        calls.append(cur)
    return calls


def write_vcf(calls: List[dict], path: str, sample: str = "SAMPLE") -> None:
    """Write a minimal VCF with one DEL line per merged call."""
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">\n')
        f.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">\n')
        f.write('##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">\n')
        f.write(
            '##INFO=<ID=NWIN,Number=1,Type=Integer,Description="Number of merged windows">\n'
        )
        f.write(
            '##INFO=<ID=MAXPROB,Number=1,Type=Float,Description="Max P(DEL) among merged windows">\n'
        )
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for i, c in enumerate(calls, 1):
            length = c["end"] - c["start"]
            info = (
                f"SVTYPE=DEL;END={c['end']};SVLEN=-{length};"
                f"NWIN={c['n_windows']};MAXPROB={c['max_prob']:.4f}"
            )
            qual = f"{c['max_prob'] * 100:.1f}"
            f.write(
                f"{c['chrom']}\t{c['start']}\tDEL_{i}\tN\t<DEL>\t{qual}\tPASS\t{info}\n"
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True)
    p.add_argument("--checkpoint", required=True, help="Trained model state_dict")
    p.add_argument("--model", choices=["cnn", "fused"], default="fused")
    p.add_argument(
        "--embeddings", default=None, help="HDF5 file (required for --model fused)"
    )
    p.add_argument(
        "--predictions-out",
        required=True,
        help="Output CSV with per-window predictions",
    )
    p.add_argument("--vcf-out", default=None, help="Optional VCF with merged DEL calls")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument(
        "--fusion-mode",
        choices=["film", "context", "film_context"],
        default="film_context",
        help=(
            "Embedding fusion used by the fused checkpoint. Match the training "
            "setting when using non-default modes."
        ),
    )
    p.add_argument(
        "--context-hidden-dim",
        type=int,
        default=128,
        help="Hidden width for context calibration heads (default: 128)",
    )
    p.add_argument(
        "--context-dropout",
        type=float,
        default=0.1,
        help="Dropout probability for context heads; inactive during eval.",
    )
    p.add_argument("--sample-id", default="SAMPLE")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    logger.info("Loaded %d rows from %s", len(rows), args.manifest)

    if args.model == "fused" and not args.embeddings:
        raise SystemExit("--embeddings is required for --model fused")

    if args.model == "cnn":
        preds = predict_m0(rows, args.checkpoint, args.threshold, args.batch_size)
    else:  # fused
        preds = predict_m1(
            rows,
            args.checkpoint,
            args.embeddings,
            args.threshold,
            args.batch_size,
            args.embed_dim,
            args.fusion_mode,
            args.context_hidden_dim,
            args.context_dropout,
        )

    # Per-window predictions CSV.
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.predictions_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image_path",
                "chrom",
                "position",
                "label",
                "length",
                "prob_deletion",
                "predicted_class",
            ]
        )
        for r, (prob, cls) in zip(rows, preds):
            w.writerow(
                [
                    r["image_path"],
                    r["chrom"],
                    r["position"],
                    r["label"],
                    r["length"],
                    f"{prob:.6f}",
                    cls,
                ]
            )
    logger.info("Wrote per-window predictions to %s", args.predictions_out)

    # Optional VCF output.
    if args.vcf_out:
        calls = merge_to_calls(rows, preds)
        Path(args.vcf_out).parent.mkdir(parents=True, exist_ok=True)
        write_vcf(calls, args.vcf_out, sample=args.sample_id)
        logger.info("Wrote %d merged DEL calls to %s", len(calls), args.vcf_out)


if __name__ == "__main__":
    main()
