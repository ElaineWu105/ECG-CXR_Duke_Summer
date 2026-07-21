#!/usr/bin/env python3
"""Create one summary dashboard for the 21 six-label ECG probe experiments."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("./7.14")
SUMMARY = ROOT / "six_label_classification/all_results_summary.csv"
OUTPUT = ROOT / "six_label_classification_summary.png"

MODELS = {
    "single_mlp": ("Single ECG", "#2878B5", "o"),
    "sequence_attention": ("ECG sequence attention", "#D95F02", "s"),
    "latest_mlp": ("Latest ECG", "#2A9D55", "^"),
}
LABELS = [
    ("pneumonia", "Pneumonia"),
    ("consolidation", "Consolidation"),
    ("lung_opacity", "Lung opacity"),
    ("pneumothorax", "Pneumothorax"),
    ("pleural_effusion", "Pleural effusion"),
    ("pulmonary_edema", "Pulmonary edema"),
]
N_VALUES = [0, 2, 4, 6, 8, 10, 12]


def read_test_rows():
    with SUMMARY.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == "test"]


def pct(value):
    return 100.0 * float(value)


def main():
    rows = read_test_rows()
    by_key = {(row["model"], int(row["n_offset_hours"])): row for row in rows}
    best = by_key[("sequence_attention", 0)]

    fig = plt.figure(figsize=(20, 7.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 1.35], wspace=0.30)

    # A: Overall model comparison across temporal offsets.
    ax1 = fig.add_subplot(grid[0, 0])
    for model, (name, color, marker) in MODELS.items():
        values = [pct(by_key[(model, n)]["macro_auroc"]) for n in N_VALUES]
        ax1.plot(N_VALUES, values, color=color, marker=marker, linewidth=2.4,
                 markersize=7, label=name)
    ax1.axhline(50, color="#777777", linestyle="--", linewidth=1.2, label="Random AUROC")
    ax1.scatter([0], [pct(best["macro_auroc"])], s=180, facecolors="none",
                edgecolors="black", linewidths=2, zorder=5)
    ax1.annotate("Best: 70.8%", (0, pct(best["macro_auroc"])), xytext=(2.2, 72.5),
                 arrowprops={"arrowstyle": "->", "color": "#333333"}, fontsize=10)
    ax1.set_title("A. Overall six-label performance", loc="left", fontweight="bold")
    ax1.set_xlabel("Window offset n (hours before CXR)")
    ax1.set_ylabel("Test macro-AUROC (%)")
    ax1.set_xticks(N_VALUES)
    ax1.set_ylim(48, 75)
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")

    # B: Per-label temporal decay for the strongest model family.
    ax2 = fig.add_subplot(grid[0, 1])
    heat = np.array([
        [pct(by_key[("sequence_attention", n)][f"{key}_auroc"]) for n in N_VALUES]
        for key, _ in LABELS
    ])
    image = ax2.imshow(heat, cmap="YlOrRd", vmin=50, vmax=80, aspect="auto")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            color = "white" if heat[i, j] >= 68 else "#222222"
            ax2.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center",
                     fontsize=9, color=color)
    ax2.set_xticks(range(len(N_VALUES)), N_VALUES)
    ax2.set_yticks(range(len(LABELS)), [name for _, name in LABELS])
    ax2.set_xlabel("Window offset n (hours before CXR)")
    ax2.set_title("B. Sequence-attention AUROC by finding", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.03)
    colorbar.set_label("Test AUROC (%)")

    # C: Detailed metrics for the best experiment.
    ax3 = fig.add_subplot(grid[0, 2])
    names = [name for _, name in LABELS]
    auroc = [pct(best[f"{key}_auroc"]) for key, _ in LABELS]
    auprc = [pct(best[f"{key}_auprc"]) for key, _ in LABELS]
    prevalence = [pct(best[f"{key}_prevalence"]) for key, _ in LABELS]
    y = np.arange(len(names))
    height = 0.23
    ax3.barh(y - height, auroc, height, color="#4C78A8", label="AUROC")
    ax3.barh(y, auprc, height, color="#F58518", label="AUPRC")
    ax3.barh(y + height, prevalence, height, color="#BAB0AC", label="Positive prevalence")
    for values, offset in ((auroc, -height), (auprc, 0), (prevalence, height)):
        for yi, value in zip(y, values):
            ax3.text(value + 0.8, yi + offset, f"{value:.1f}", va="center", fontsize=8)
    ax3.set_yticks(y, names)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 86)
    ax3.set_xlabel("Percent (%)")
    ax3.set_title("C. Best setting: n=0 sequence attention", loc="left", fontweight="bold")
    ax3.grid(axis="x", alpha=0.22)
    ax3.legend(frameon=False, ncol=3, fontsize=9, loc="lower right")

    fig.suptitle("ECG embeddings predicting six CXR findings", fontsize=18, fontweight="bold", y=0.985)
    fig.text(0.5, 0.015,
             "AUROC: ranking/discrimination (50% ≈ random). AUPRC depends strongly on positive prevalence.",
             ha="center", fontsize=10, color="#444444")
    fig.subplots_adjust(top=0.90, bottom=0.12, left=0.06, right=0.98)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
