#!/usr/bin/env python3
"""Plot epoch-level train and validation histories for all 7.17 CLS runs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parent
ROOT = BASE / "cross_patient_cls_pool_case2_case3"
OUT = BASE / "summary_cls_pool" / "train_validation_summary.png"


def main():
    runs = []
    for path in ROOT.glob("*/*/results.json"):
        data = json.loads(path.read_text())
        name = data["spec"]["name"]
        n = int(re.search(r"_n(\d+)$", name).group(1))
        case = 2 if name.startswith("case2") else 3
        runs.append((case, n, data))
    runs.sort(key=lambda item: (item[0], item[1]))
    if len(runs) != 8:
        raise RuntimeError(f"Expected 8 completed runs, found {len(runs)}")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=False)
    handles = None
    labels = None
    for ax, (case, n, data) in zip(axes.flat, runs):
        history = data["history"]
        epochs = [row["epoch"] for row in history]
        train_loss = [row["train"]["loss"] for row in history]
        train_r1 = [row["train"]["cross_patient_batch_top1"] for row in history]
        val_mrr = [row["val"]["cross_patient"]["mrr"] for row in history]
        best_epoch = int(data["best_epoch"])
        best_index = epochs.index(best_epoch)

        loss_line = ax.plot(epochs, train_loss, color="#666666", linewidth=2,
                            label="Train loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss", color="#555555")
        ax.tick_params(axis="y", labelcolor="#555555")
        ax.grid(alpha=0.22)

        metric_ax = ax.twinx()
        train_line = metric_ax.plot(epochs, train_r1, color="#4C78A8", linewidth=1.8,
                                    marker="o", markersize=3, label="Train batch R@1")
        val_line = metric_ax.plot(epochs, val_mrr, color="#F58518", linewidth=2.2,
                                  marker="s", markersize=3, label="Validation MRR")
        metric_ax.scatter([best_epoch], [val_mrr[best_index]], s=85, marker="*",
                          color="#E45756", zorder=5, label="Best epoch")
        metric_ax.axvline(best_epoch, color="#E45756", linestyle="--", alpha=0.55, linewidth=1)
        metric_ax.set_ylabel("Retrieval metric", color="#333333")
        metric_ax.set_ylim(bottom=0)
        label = "Case 2: sequence CLS" if case == 2 else "Case 3: nearest ECG"
        ax.set_title(f"{label}, n={n}\nbest epoch={best_epoch}, best val MRR={val_mrr[best_index]:.4f}",
                     fontsize=10.5)
        if handles is None:
            handles = loss_line + train_line + val_line + [metric_ax.collections[-1]]
            labels = [h.get_label() for h in handles]

    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.967))
    fig.suptitle("7.17 training and validation dynamics", fontsize=16,
                 fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
