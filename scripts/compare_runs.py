#!/usr/bin/env python3
"""Compare two training runs side-by-side.

Reads ``metrics.jsonl`` and ``config.json`` from each run directory and
produces:
  - ``comparison.png``: overlaid val-loss / val-F1 / val-ROC-AUC / val-PR-AUC curves.
  - ``comparison.txt``: best-epoch summary per run, plus delta.

Usage::

    python scripts/compare_runs.py \
        --run-a models/image_only \
        --run-b models/image_dnabert \
        --output reports/compare_3ch_vs_11ch
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = [
    ("val_loss", "Val Loss", "lower"),
    ("val_bal_acc", "Val Balanced Accuracy", "higher"),
    ("val_f1_weighted", "Val Weighted F1", "higher"),
    ("val_f1_macro", "Val Macro F1", "higher"),
    ("val_f1_1", "Val F1 (Deletion class)", "higher"),
    ("val_roc_auc", "Val ROC-AUC", "higher"),
    ("val_pr_auc", "Val PR-AUC", "higher"),
    ("val_recall_1", "Val Recall (Deletion)", "higher"),
]


def load_run(run_dir: Path):
    cfg_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found.")
    config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"run_name": run_dir.name}
    epochs = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if line:
            epochs.append(json.loads(line))
    return config, epochs


def best_epoch(epochs, key, direction):
    if not epochs:
        return None
    fn = min if direction == "lower" else max
    return fn(epochs, key=lambda e: e.get(key, float("inf") if direction == "lower" else float("-inf")))


def plot_overlay(runs, output_path):
    n = len(METRICS)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), constrained_layout=True)
    fig.suptitle("Training run comparison", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for ax, (key, title, _direction) in zip(axes, METRICS):
        for (cfg, epochs), color in zip(runs, colors):
            xs = [e["epoch"] for e in epochs]
            ys = [e.get(key) for e in epochs]
            ax.plot(xs, ys, "o-", color=color, lw=2, markersize=4, label=cfg.get("run_name", "?"))
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[len(METRICS):]:
        ax.axis("off")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary(runs, output_path):
    lines = []
    lines.append("=" * 78)
    lines.append("Run comparison summary")
    lines.append("=" * 78)
    lines.append("")

    for cfg, epochs in runs:
        lines.append(f"Run: {cfg.get('run_name', '?')}")
        lines.append(f"  input_channels:  {cfg.get('input_channels')}")
        lines.append(f"  context_channels:{cfg.get('context_channels')}")
        lines.append(f"  n_train / n_val: {cfg.get('n_train')} / {cfg.get('n_val')}")
        lines.append(f"  epochs run:      {len(epochs)}")
        for key, title, direction in METRICS:
            best = best_epoch(epochs, key, direction)
            if best is None:
                continue
            lines.append(f"  best {title:<32} = {best.get(key):.4f}  (epoch {best.get('epoch')})")
        lines.append("")

    if len(runs) == 2:
        lines.append("-" * 78)
        lines.append(f"Delta (Run B − Run A) on best-epoch values")
        lines.append(f"  A = {runs[0][0].get('run_name')}")
        lines.append(f"  B = {runs[1][0].get('run_name')}")
        lines.append("-" * 78)
        for key, title, direction in METRICS:
            a = best_epoch(runs[0][1], key, direction)
            b = best_epoch(runs[1][1], key, direction)
            if a is None or b is None:
                continue
            delta = b.get(key) - a.get(key)
            sign = "+" if delta >= 0 else ""
            arrow = "↑" if (delta > 0 and direction == "higher") or (delta < 0 and direction == "lower") else "↓"
            lines.append(f"  {title:<32}  A={a.get(key):.4f}  B={b.get(key):.4f}  Δ={sign}{delta:.4f}  {arrow}")

    output_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare two training runs.")
    parser.add_argument("--run-a", required=True, help="Directory of run A (must contain metrics.jsonl).")
    parser.add_argument("--run-b", required=True, help="Directory of run B.")
    parser.add_argument("--output", required=True, help="Output directory for comparison artifacts.")
    args = parser.parse_args()

    runs = [load_run(Path(args.run_a)), load_run(Path(args.run_b))]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    plot_overlay(runs, out / "comparison.png")
    write_summary(runs, out / "comparison.txt")
    print(f"Saved comparison to {out}/")
    print((out / "comparison.txt").read_text())


if __name__ == "__main__":
    main()
