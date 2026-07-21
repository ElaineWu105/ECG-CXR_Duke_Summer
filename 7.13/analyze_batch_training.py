#!/usr/bin/env python3
"""Summarize batch-level training and epoch-level validation for Cases 1/2/3."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def rolling(values, width):
    values = np.asarray(values, dtype=float)
    if width <= 1 or len(values) < width:
        return values
    kernel = np.ones(width) / width
    return np.convolve(values, kernel, mode="valid")


def read_batches(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in row:
            row[key] = number(row[key])
    return rows


def discover(root):
    found = defaultdict(list)
    for path in root.glob("**/results.json"):
        data = json.load(path.open())
        name = data.get("spec", {}).get("name", "")
        case_match = re.match(r"(case[123])_", name)
        n_match = re.search(r"_n(\d+)$", name)
        if case_match and n_match:
            found[case_match.group(1)].append((int(n_match.group(1)), path, data))
    for case in ("case1", "case2", "case3"):
        found[case].sort()
    return found


def epoch_rows(case, n, result_path, result, batches):
    grouped = defaultdict(list)
    for row in batches:
        grouped[int(row["epoch"])].append(row)
    history = {int(row["epoch"]): row for row in result.get("history", [])}
    output = []
    for epoch in sorted(grouped):
        rows = grouped[epoch]
        val = history.get(epoch, {}).get("val", {}).get("cross_patient", {})
        output.append({
            "case": case, "n": n, "epoch": epoch, "num_batches": len(rows),
            "train_loss_mean": np.nanmean([r["loss"] for r in rows]),
            "train_loss_std": np.nanstd([r["loss"] for r in rows]),
            "train_batch_R@1_mean": np.nanmean([r["train_cross_patient_R@1"] for r in rows]),
            "train_batch_R@1_std": np.nanstd([r["train_cross_patient_R@1"] for r in rows]),
            "train_batch_R@5_mean": np.nanmean([r["train_cross_patient_R@5"] for r in rows]),
            "train_batch_R@5_std": np.nanstd([r["train_cross_patient_R@5"] for r in rows]),
            "val_queries": val.get("n_queries", ""),
            "val_gallery_size": val.get("gallery_size", ""),
            "val_R@1": val.get("recall@1", ""),
            "val_R@5": val.get("recall@5", ""),
            "val_R@10": val.get("recall@10", ""),
            "val_MRR": val.get("mrr", ""),
            "val_MedR": val.get("median_rank", ""),
            "is_best_epoch": int(epoch == result.get("best_epoch")),
            "result_path": str(result_path),
        })
    return output


def plot_batches(path, case, n, rows, rolling_width):
    step = np.array([r["global_step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    r1 = np.array([r["train_cross_patient_R@1"] for r in rows])
    r5 = np.array([r["train_cross_patient_R@5"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(step, loss, color="#4C78A8", alpha=.18, linewidth=.6, label="batch")
    smooth = rolling(loss, rolling_width)
    axes[0].plot(step[-len(smooth):], smooth, color="#1f4e79", linewidth=2, label=f"rolling {rolling_width}")
    axes[0].set_ylabel("Train loss ↓"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(step, r1, color="#F58518", alpha=.12, linewidth=.5)
    axes[1].plot(step, r5, color="#54A24B", alpha=.12, linewidth=.5)
    s1, s5 = rolling(r1, rolling_width), rolling(r5, rolling_width)
    axes[1].plot(step[-len(s1):], s1, color="#E45756", linewidth=2, label="Train batch R@1")
    axes[1].plot(step[-len(s5):], s5, color="#2E8B57", linewidth=2, label="Train batch R@5")
    axes[1].set_xlabel("Global training step"); axes[1].set_ylabel("Batch retrieval ↑")
    axes[1].legend(); axes[1].grid(alpha=.2)
    fig.suptitle(f"{case.upper()} n={n}: batch-level training dynamics")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_epochs(path, case, n, rows):
    epoch = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epoch, [r["train_loss_mean"] for r in rows], marker="o", ms=3)
    axes[0].set_title("Train batch loss (epoch mean)"); axes[0].set_ylabel("Loss ↓")
    axes[1].plot(epoch, [r["train_batch_R@1_mean"] for r in rows], label="Train batch R@1")
    axes[1].plot(epoch, [r["train_batch_R@5_mean"] for r in rows], label="Train batch R@5")
    axes[1].set_title("Train, gallery=batch"); axes[1].legend()
    axes[2].plot(epoch, [number(r["val_R@1"]) for r in rows], label="Val R@1")
    axes[2].plot(epoch, [number(r["val_R@5"]) for r in rows], label="Val R@5")
    axes[2].plot(epoch, [number(r["val_MRR"]) for r in rows], label="Val MRR")
    best = [r["epoch"] for r in rows if r["is_best_epoch"]]
    if best: axes[2].axvline(best[0], color="black", linestyle="--", alpha=.5, label=f"best E{best[0]}")
    axes[2].set_title("Validation, full gallery"); axes[2].legend()
    for ax in axes: ax.set_xlabel("Epoch"); ax.grid(alpha=.2)
    fig.suptitle(f"{case.upper()} n={n}: train vs validation")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, default=Path("./7.13/cross_patient_huge_batch"))
    p.add_argument("--output_dir", type=Path, default=Path("./7.13/batch_training_analysis"))
    p.add_argument("--rolling_width", type=int, default=50)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)

    for case, experiments in discover(a.results_root).items():
        case_dir = a.output_dir / case; case_dir.mkdir(parents=True, exist_ok=True)
        combined = []
        for n, result_path, result in experiments:
            dynamics = result_path.parent / "train_dynamics.csv"
            if not dynamics.is_file(): raise FileNotFoundError(dynamics)
            batches = read_batches(dynamics)
            rows = epoch_rows(case, n, result_path, result, batches); combined.extend(rows)
            plot_batches(case_dir / f"n{n}_batch_dynamics.png", case, n, batches, a.rolling_width)
            plot_epochs(case_dir / f"n{n}_train_validation.png", case, n, rows)
        if not combined: continue
        output = case_dir / "epoch_train_validation.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(combined[0]))
            writer.writeheader(); writer.writerows(combined)
        print(f"Wrote {output} and {len(experiments)*2} plots")

if __name__ == "__main__": main()
