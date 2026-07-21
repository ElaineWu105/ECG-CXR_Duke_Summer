#!/usr/bin/env python3
"""Plot one dashboard comparing all 21 cross-patient experiments."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CASES = {
    "case1": "Case 1: all ECGs",
    "case2": "Case 2: sequence mean-pool",
    "case3": "Case 3: nearest ECG",
}
COLORS = {"case1": "#2878B5", "case2": "#D95F02", "case3": "#2A9D55"}
MARKERS = {"case1": "o", "case2": "s", "case3": "^"}


def read_rows(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["n"] = int(row["n"])
        for key in (
            "best_val_MRR", "best_val_R@1", "train_R@1_gain_first_to_last",
            "val_MRR_drop_best_to_last",
        ):
            row[key] = float(row[key]) * 100.0
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary", type=Path,
        default=Path("./7.13/summary/batch_train_validation_summary.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("./7.13/summary/cross_patient_21_experiments.png"),
    )
    args = parser.parse_args()
    rows = read_rows(args.summary)

    panels = (
        ("best_val_MRR", "A. Best validation MRR", "MRR (%)", True),
        ("best_val_R@1", "B. Best validation R@1", "R@1 (%)", True),
        ("train_R@1_gain_first_to_last", "C. Train batch R@1 gain: first → last", "Gain (percentage points)", False),
        ("val_MRR_drop_best_to_last", "D. Validation MRR drop: best → last", "Drop (percentage points)", False),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, (key, title, ylabel, higher_better) in zip(axes.flat, panels):
        for case in CASES:
            selected = sorted((r for r in rows if r["case"] == case), key=lambda r: r["n"])
            ax.plot(
                [r["n"] for r in selected], [r[key] for r in selected],
                color=COLORS[case], marker=MARKERS[case], linewidth=2.2,
                markersize=7, label=CASES[case],
            )
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
        ax.grid(alpha=0.25)
        direction = "higher is better" if higher_better else (
            "larger = more training fit" if key.startswith("train") else "larger = more overfitting"
        )
        ax.text(0.99, 0.04, direction, transform=ax.transAxes, ha="right", va="bottom",
                color="#555555", fontsize=9)

    for ax in axes[1]:
        ax.set_xlabel("Window offset n (hours before CXR)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Cross-patient training vs validation: all 21 experiments", fontsize=17, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Train retrieval uses the current batch gallery; validation uses the full validation gallery. "
        "Compare trends, not their absolute scales.",
        ha="center", fontsize=10, color="#444444",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
