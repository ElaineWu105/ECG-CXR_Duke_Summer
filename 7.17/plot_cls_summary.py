#!/usr/bin/env python3
"""Plot a compact summary of the 7.17 CLS-pooling experiment."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent / "cross_patient_cls_pool_case2_case3"
OUT = Path(__file__).resolve().parent / "summary_cls_pool"


def load_rows():
    rows = []
    for path in ROOT.glob("*/results_table.csv"):
        with path.open(newline="") as handle:
            row = next(csv.DictReader(handle))
        row["n"] = int(re.search(r"_n(\d+)$", row["experiment_name"]).group(1))
        row["case"] = "Case 2: sequence CLS" if "case2_" in row["experiment_name"] else "Case 3: nearest ECG"
        rows.append(row)
    return sorted(rows, key=lambda r: (r["case"], r["n"]))


def main():
    rows = load_rows()
    if len(rows) != 8:
        raise RuntimeError(f"Expected 8 completed runs, found {len(rows)}")

    OUT.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT / "cls_pool_metrics.csv"
    columns = ["case", "n", "target_window", "cross_patient_recall@1",
               "cross_patient_recall@5", "cross_patient_recall@10",
               "cross_patient_mrr", "cross_patient_median_rank",
               "within_patient_temporal_recall@1",
               "within_patient_temporal_recall@5", "within_patient_temporal_mrr"]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in rows)

    panels = [
        ("cross_patient_recall@1", "Cross-patient R@1", "higher is better"),
        ("cross_patient_recall@5", "Cross-patient R@5", "higher is better"),
        ("cross_patient_recall@10", "Cross-patient R@10", "higher is better"),
        ("cross_patient_mrr", "Cross-patient MRR", "higher is better"),
        ("cross_patient_median_rank", "Cross-patient median rank", "lower is better"),
        ("within_patient_temporal_recall@1", "Within-patient temporal R@1", "higher is better"),
        ("within_patient_temporal_recall@5", "Within-patient temporal R@5", "higher is better"),
        ("within_patient_temporal_mrr", "Within-patient temporal MRR", "higher is better"),
    ]
    styles = {
        "Case 2: sequence CLS": ("#4C78A8", "o"),
        "Case 3: nearest ECG": ("#F58518", "s"),
    }
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for ax, (metric, title, note) in zip(axes.flat, panels):
        for case, (color, marker) in styles.items():
            selected = [row for row in rows if row["case"] == case]
            ax.plot([row["n"] for row in selected], [float(row[metric]) for row in selected],
                    color=color, marker=marker, linewidth=2.2, markersize=6, label=case)
            for row in selected:
                value = float(row[metric])
                label = f"{value:.4f}" if metric != "cross_patient_median_rank" else f"{value:.0f}"
                ax.annotate(label, (row["n"], value), xytext=(0, 7),
                            textcoords="offset points", ha="center", fontsize=7, color=color)
        ax.set_title(f"{title}\n{note}", fontsize=10.5)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xlabel("Window offset n")
        ax.grid(alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("7.17 CLS-pooling experiment summary (8 runs)",
                 fontsize=16, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_png = OUT / "cls_pool_summary.png"
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(output_png)
    print(summary_csv)


if __name__ == "__main__":
    main()
