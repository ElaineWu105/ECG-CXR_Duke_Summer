#!/usr/bin/env python3
"""Create separate summary figures for the two CheXpert label policies."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("./7.14/label_policy_comparison")
INPUT = ROOT / "combined_results.csv"
FIGURES = ROOT / "figures"
N_VALUES = [0, 2, 4, 6, 8, 10, 12]
MODELS = [
    ("single_mlp", "Single ECG"),
    ("sequence_attention", "ECG sequence attention"),
    ("latest_mlp", "Latest ECG"),
]
POLICIES = [
    ("explicit_01", "Explicit 0/1 only", "#277DA1", "o"),
    ("all_nonpositive_negative", "1 positive; all others negative", "#F3722C", "s"),
]
LABELS = [
    ("pneumonia", "Pneumonia"),
    ("consolidation", "Consolidation"),
    ("lung_opacity", "Lung opacity"),
    ("pneumothorax", "Pneumothorax"),
    ("pleural_effusion", "Pleural effusion"),
    ("pulmonary_edema", "Pulmonary edema"),
]


def save(fig, filename):
    fig.savefig(FIGURES / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def line_figure(test, metric, ylabel, filename, random_line=False):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (model, title) in zip(axes, MODELS):
        subset = test[test["model"] == model]
        for policy, name, color, marker in POLICIES:
            rows = subset[subset["label_policy"] == policy].set_index("n_offset_hours")
            values = [100 * rows.loc[n, metric] for n in N_VALUES]
            ax.plot(N_VALUES, values, color=color, marker=marker, linewidth=2.3,
                    markersize=6, label=name)
        if random_line:
            ax.axhline(50, color="#777777", linestyle="--", linewidth=1)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Window offset n")
        ax.set_xticks(N_VALUES)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Test {ylabel}: label-policy comparison", fontsize=16,
                 fontweight="bold")
    fig.tight_layout()
    save(fig, filename)


def raw_heatmap(test, metric, title, filename, vmin, vmax, cmap):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True)
    for policy_row, (policy, policy_title, _, _) in enumerate(POLICIES):
        for model_column, (model, model_title) in enumerate(MODELS):
            ax = axes[policy_row, model_column]
            rows = test[
                (test["label_policy"] == policy) & (test["model"] == model)
            ].set_index("n_offset_hours")
            matrix = np.array([
                [100 * rows.loc[n, f"{label}_{metric}"] for n in N_VALUES]
                for label, _ in LABELS
            ])
            image = ax.imshow(matrix, aspect="auto", cmap=cmap,
                              vmin=vmin, vmax=vmax)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    value = matrix[row, column]
                    color = "white" if value > (vmin + vmax) / 2 else "#222222"
                    ax.text(column, row, f"{value:.1f}", ha="center", va="center",
                            fontsize=8, color=color)
            if policy_row == 0:
                ax.set_title(model_title, fontweight="bold")
            if model_column == 0:
                ax.set_ylabel(policy_title + "\n\nDisease")
            ax.set_xticks(range(len(N_VALUES)), N_VALUES)
            ax.set_yticks(range(len(LABELS)), [name for _, name in LABELS])
            if policy_row == 1:
                ax.set_xlabel("Window offset n")
    cbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Actual value (%)")
    fig.suptitle(title + " — actual test values", fontsize=16, fontweight="bold")
    fig.subplots_adjust(top=0.92, bottom=0.08, left=0.12, right=0.92,
                        hspace=0.15, wspace=0.08)
    save(fig, filename)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT)
    test = data[data["split"] == "test"].copy()
    line_figure(test, "macro_auroc", "Macro-AUROC (%)",
                "01_test_macro_auroc.png", random_line=True)
    line_figure(test, "macro_auprc", "Macro-AUPRC (%)",
                "02_test_macro_auprc.png")
    raw_heatmap(test, "auroc", "Six CXR classifications: AUROC",
                "03_six_classifications_auroc.png", 45, 80, "YlGnBu")
    raw_heatmap(test, "auprc", "Six CXR classifications: AUPRC",
                "04_six_classifications_auprc.png", 0, 100, "YlOrRd")
    raw_heatmap(test, "prevalence", "Six CXR classifications: prevalence",
                "05_six_classifications_prevalence.png", 0, 100, "Purples")
    print(f"Saved 5 figures to {FIGURES}")


if __name__ == "__main__":
    main()
