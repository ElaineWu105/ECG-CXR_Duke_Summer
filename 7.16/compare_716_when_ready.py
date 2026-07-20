#!/usr/bin/env python3
"""Wait for 7.16 results, summarize them, and compare against prior runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

RUNS = {
    "original_713": ROOT / "7.13/summary/batch_train_validation_summary.csv",
    "dropout_713": ROOT / "7.13/summary_rerun_embdrop05/batch_train_validation_summary.csv",
    "antioverfit_715": ROOT / "7.15/summary_antioverfit_seqtokdrop/batch_train_validation_summary.csv",
}

FIELDS = [
    "case", "n", "target_window", "epochs_ran", "best_epoch",
    "first_train_loss", "best_epoch_train_loss", "last_train_loss",
    "first_train_batch_R@1", "best_epoch_train_batch_R@1", "last_train_batch_R@1",
    "first_train_batch_R@5", "best_epoch_train_batch_R@5", "last_train_batch_R@5",
    "first_val_R@1", "best_val_R@1", "last_val_R@1",
    "first_val_R@5", "best_val_R@5", "last_val_R@5",
    "first_val_R@10", "best_val_R@10", "last_val_R@10",
    "first_val_MRR", "best_val_MRR", "last_val_MRR",
    "best_val_MedR", "last_val_MedR",
    "val_MRR_gain_first_to_best", "val_MRR_drop_best_to_last",
    "train_R@1_gain_first_to_last", "result_path",
]

METRICS = [
    "best_val_MRR", "best_val_R@1", "best_val_R@5",
    "train_R@1_gain_first_to_last", "val_MRR_drop_best_to_last",
    "epochs_ran", "best_epoch",
]

CASE_LABELS = {
    "case1": "Case 1 all ECGs",
    "case2": "Case 2 sequence",
    "case3": "Case 3 nearest ECG",
}


def nested_get(row: dict, *keys, default=""):
    value = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def diff(a, b):
    try:
        return float(a) - float(b)
    except Exception:
        return ""


def number(value):
    try:
        out = float(value)
        return out if math.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


def mean(values):
    vals = [number(v) for v in values]
    vals = [v for v in vals if math.isfinite(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def wait_for_results(results_root: Path, expected: int, poll_seconds: int, timeout_minutes: float | None):
    start = time.time()
    while True:
        paths = sorted(results_root.glob("**/results.json"))
        count = len(paths)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] found {count}/{expected} results under {results_root}", flush=True)
        if count >= expected:
            return paths
        if timeout_minutes is not None and (time.time() - start) > timeout_minutes * 60:
            raise TimeoutError(f"Timed out waiting for {expected} results; found {count}")
        time.sleep(poll_seconds)


def summarize_results(results_root: Path, output_csv: Path):
    rows = []
    for path in sorted(results_root.glob("**/results.json")):
        with path.open() as handle:
            data = json.load(handle)
        spec = data.get("spec", {})
        name = spec.get("name", "")
        case_match = re.match(r"(case[123])_", name)
        n_match = re.search(r"_n(\d+)$", name)
        if not (case_match and n_match):
            continue
        history = data.get("history", [])
        if not history:
            continue
        best_epoch = int(data["best_epoch"])
        best = next((row for row in history if int(row["epoch"]) == best_epoch), None)
        if best is None:
            continue
        first, last = history[0], history[-1]
        val = lambda row, key: nested_get(row, "val", "cross_patient", key)
        train = lambda row, key: nested_get(row, "train", key)
        first_mrr, best_mrr, last_mrr = val(first, "mrr"), val(best, "mrr"), val(last, "mrr")
        first_train_r1 = train(first, "cross_patient_batch_top1")
        last_train_r1 = train(last, "cross_patient_batch_top1")
        rows.append({
            "case": case_match.group(1), "n": int(n_match.group(1)),
            "target_window": spec.get("target_window", ""),
            "epochs_ran": len(history), "best_epoch": best_epoch,
            "first_train_loss": train(first, "loss"),
            "best_epoch_train_loss": train(best, "loss"),
            "last_train_loss": train(last, "loss"),
            "first_train_batch_R@1": first_train_r1,
            "best_epoch_train_batch_R@1": train(best, "cross_patient_batch_top1"),
            "last_train_batch_R@1": last_train_r1,
            "first_train_batch_R@5": train(first, "cross_patient_batch_top5"),
            "best_epoch_train_batch_R@5": train(best, "cross_patient_batch_top5"),
            "last_train_batch_R@5": train(last, "cross_patient_batch_top5"),
            "first_val_R@1": val(first, "recall@1"), "best_val_R@1": val(best, "recall@1"),
            "last_val_R@1": val(last, "recall@1"),
            "first_val_R@5": val(first, "recall@5"), "best_val_R@5": val(best, "recall@5"),
            "last_val_R@5": val(last, "recall@5"),
            "first_val_R@10": val(first, "recall@10"), "best_val_R@10": val(best, "recall@10"),
            "last_val_R@10": val(last, "recall@10"),
            "first_val_MRR": first_mrr, "best_val_MRR": best_mrr, "last_val_MRR": last_mrr,
            "best_val_MedR": val(best, "median_rank"), "last_val_MedR": val(last, "median_rank"),
            "val_MRR_gain_first_to_best": diff(best_mrr, first_mrr),
            "val_MRR_drop_best_to_last": diff(best_mrr, last_mrr),
            "train_R@1_gain_first_to_last": diff(last_train_r1, first_train_r1),
            "result_path": str(path),
        })
    rows.sort(key=lambda row: (int(row["case"][-1]), int(row["n"])))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_csv} ({len(rows)} rows)")
    return rows


def read_summary(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_means(rows, group_key: str | None = None):
    if group_key is None:
        return {metric: mean(row[metric] for row in rows) for metric in METRICS}
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    return {key: {metric: mean(row[metric] for row in group) for metric in METRICS}
            for key, group in sorted(groups.items())}


def write_comparison_csv(run_rows: dict[str, list[dict]], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        fieldnames = ["run", "case"] + METRICS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, rows in run_rows.items():
            overall = grouped_means(rows)
            writer.writerow({"run": run_name, "case": "overall", **overall})
            for case, vals in grouped_means(rows, "case").items():
                writer.writerow({"run": run_name, "case": case, **vals})
    print(f"Wrote {output_csv}")


def write_delta_csv(base_rows, new_rows, output_csv: Path, base_name: str, new_name: str):
    base = {(r["case"], int(r["n"])): r for r in base_rows}
    new = {(r["case"], int(r["n"])): r for r in new_rows}
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        fieldnames = ["case", "n"] + [f"{m}_delta_{new_name}_minus_{base_name}" for m in METRICS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(base, key=lambda x: (int(x[0][-1]), x[1])):
            if key not in new:
                continue
            row = {"case": key[0], "n": key[1]}
            for metric in METRICS:
                row[f"{metric}_delta_{new_name}_minus_{base_name}"] = number(new[key][metric]) - number(base[key][metric])
            writer.writerow(row)
    print(f"Wrote {output_csv}")


def plot_overall(run_rows: dict[str, list[dict]], output_png: Path):
    run_names = list(run_rows)
    metrics = ["best_val_MRR", "best_val_R@1", "best_val_R@5", "train_R@1_gain_first_to_last", "val_MRR_drop_best_to_last"]
    titles = ["Best val MRR", "Best val R@1", "Best val R@5", "Train R@1 gain", "Val MRR drop"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4.6))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [grouped_means(run_rows[name])[metric] for name in run_names]
        ax.bar(range(len(run_names)), vals, color=["#666666", "#4C78A8", "#F58518", "#54A24B"][:len(run_names)])
        ax.set_title(title)
        ax.set_xticks(range(len(run_names)))
        ax.set_xticklabels(run_names, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Overall comparison across 21 experiments")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")


def plot_case2_by_n(run_rows: dict[str, list[dict]], output_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    plot_metrics = [("best_val_MRR", "Best val MRR"), ("train_R@1_gain_first_to_last", "Train R@1 gain"), ("val_MRR_drop_best_to_last", "Val MRR drop")]
    colors = ["#666666", "#4C78A8", "#F58518", "#54A24B"]
    for ax, (metric, title) in zip(axes, plot_metrics):
        for color, (run_name, rows) in zip(colors, run_rows.items()):
            selected = sorted((r for r in rows if r["case"] == "case2"), key=lambda r: int(r["n"]))
            ax.plot([int(r["n"]) for r in selected], [number(r[metric]) for r in selected], marker="o", linewidth=2, label=run_name, color=color)
        ax.set_title(title)
        ax.set_xlabel("n offset")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="best", fontsize=8)
    fig.suptitle("Case2 sequence comparison by n")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")



def plot_case_metric_grid(run_rows: dict[str, list[dict]], output_png: Path):
    metrics = [
        ("best_val_MRR", "Best validation MRR", "higher better"),
        ("best_val_R@1", "Best validation R@1", "higher better"),
        ("train_R@1_gain_first_to_last", "Train batch R@1 gain", "lower means less fitting"),
        ("val_MRR_drop_best_to_last", "Validation MRR drop", "lower means less overfit"),
    ]
    cases = ["case1", "case2", "case3"]
    colors = {
        "original_713": "#666666",
        "dropout_713": "#4C78A8",
        "antioverfit_715": "#F58518",
        "mild_716": "#54A24B",
    }
    markers = {"original_713": "o", "dropout_713": "s", "antioverfit_715": "^", "mild_716": "D"}
    fig, axes = plt.subplots(len(cases), len(metrics), figsize=(18, 11.5), sharex=True)
    for r, case in enumerate(cases):
        for c, (metric, title, note) in enumerate(metrics):
            ax = axes[r][c]
            for run_name, rows in run_rows.items():
                selected = sorted((row for row in rows if row["case"] == case), key=lambda row: int(row["n"]))
                ax.plot(
                    [int(row["n"]) for row in selected],
                    [number(row[metric]) for row in selected],
                    color=colors.get(run_name), marker=markers.get(run_name, "o"),
                    linewidth=2.0, markersize=5.5, label=run_name,
                )
            if r == 0:
                ax.set_title(f"{title}\n{note}", fontsize=11)
            if c == 0:
                ax.set_ylabel(CASE_LABELS.get(case, case), fontsize=11)
            ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
            ax.grid(alpha=0.25)
    for ax in axes[-1]:
        ax.set_xlabel("Window offset n")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Original vs dropout vs 7.15 vs 7.16: metric curves by case", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")


def plot_mrr_overfit_scatter(run_rows: dict[str, list[dict]], output_png: Path):
    colors = {
        "original_713": "#666666",
        "dropout_713": "#4C78A8",
        "antioverfit_715": "#F58518",
        "mild_716": "#54A24B",
    }
    markers = {"case1": "o", "case2": "s", "case3": "^"}
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for run_name, rows in run_rows.items():
        by_case = grouped_means(rows, "case")
        for case, vals in by_case.items():
            ax.scatter(
                vals["val_MRR_drop_best_to_last"], vals["best_val_MRR"],
                color=colors.get(run_name), marker=markers.get(case, "o"),
                s=95, edgecolor="white", linewidth=0.8,
            )
            ax.text(
                vals["val_MRR_drop_best_to_last"], vals["best_val_MRR"],
                f" {run_name.replace('_713','').replace('antioverfit_','715_').replace('mild_','')} {case[-1]}",
                fontsize=8, va="center",
            )
    ax.set_xlabel("Validation MRR drop best → last (lower is better)")
    ax.set_ylabel("Best validation MRR (higher is better)")
    ax.set_title("MRR vs overfitting tradeoff by case")
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, "Best region is upper-left: high MRR with low validation drop.", ha="center", color="#555555", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", type=Path, default=ROOT / "7.16/cross_patient_huge_batch_mild_antioverfit")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "7.16/summary_mild_antioverfit")
    parser.add_argument("--expected", type=int, default=21)
    parser.add_argument("--poll_seconds", type=int, default=300)
    parser.add_argument("--timeout_minutes", type=float, default=None)
    parser.add_argument("--no_wait", action="store_true")
    args = parser.parse_args()

    if args.no_wait:
        count = len(list(args.results_root.glob("**/results.json")))
        if count < args.expected:
            raise SystemExit(f"Only found {count}/{args.expected} results. Remove --no_wait or wait longer.")
    else:
        wait_for_results(args.results_root, args.expected, args.poll_seconds, args.timeout_minutes)

    summary_csv = args.output_dir / "batch_train_validation_summary.csv"
    rows_716 = summarize_results(args.results_root, summary_csv)
    if len(rows_716) != args.expected:
        raise SystemExit(f"Expected {args.expected} summarizable rows, got {len(rows_716)}")

    run_rows = {name: read_summary(path) for name, path in RUNS.items()}
    run_rows["mild_716"] = rows_716

    write_comparison_csv(run_rows, args.output_dir / "comparison_overall_and_by_case.csv")
    write_delta_csv(run_rows["dropout_713"], rows_716, args.output_dir / "delta_716_minus_dropout_713.csv", "dropout_713", "716")
    write_delta_csv(run_rows["antioverfit_715"], rows_716, args.output_dir / "delta_716_minus_715.csv", "715", "716")
    write_delta_csv(run_rows["original_713"], rows_716, args.output_dir / "delta_716_minus_original_713.csv", "original_713", "716")
    plot_overall(run_rows, args.output_dir / "overall_comparison.png")
    plot_case2_by_n(run_rows, args.output_dir / "case2_by_n_comparison.png")
    plot_case_metric_grid(run_rows, args.output_dir / "all_cases_metric_grid.png")
    plot_mrr_overfit_scatter(run_rows, args.output_dir / "mrr_vs_overfit_scatter.png")

    print("\nDone. Main image outputs:")
    print(f"  {args.output_dir / 'all_cases_metric_grid.png'}")
    print(f"  {args.output_dir / 'mrr_vs_overfit_scatter.png'}")
    print(f"  {args.output_dir / 'overall_comparison.png'}")
    print(f"  {args.output_dir / 'case2_by_n_comparison.png'}")
    print("\nCSV files are also saved in the same folder if you need exact values later.")


if __name__ == "__main__":
    main()
