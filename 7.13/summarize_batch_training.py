#!/usr/bin/env python3
"""Create one compact CSV summarizing train-batch and validation dynamics."""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path


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


def get(row, *keys, default=""):
    value = row
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def val(row, key):
    return get(row, "val", "cross_patient", key)


def train(row, key):
    return get(row, "train", key)


def difference(a, b):
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return ""


def make_row(case, path):
    data = json.load(path.open())
    spec = data.get("spec", {})
    match = re.search(r"_n(\d+)$", spec.get("name", ""))
    if not match:
        raise ValueError(f"Cannot parse n from {path}")
    history = data.get("history", [])
    if not history:
        raise ValueError(f"No history in {path}")
    first, last = history[0], history[-1]
    best_epoch = int(data["best_epoch"])
    best = next((row for row in history if int(row["epoch"]) == best_epoch), None)
    if best is None:
        raise ValueError(f"Best epoch {best_epoch} missing from history in {path}")

    first_mrr, best_mrr, last_mrr = val(first, "mrr"), val(best, "mrr"), val(last, "mrr")
    first_train_r1 = train(first, "cross_patient_batch_top1")
    last_train_r1 = train(last, "cross_patient_batch_top1")
    return {
        "case": case, "n": int(match.group(1)),
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
        "val_MRR_gain_first_to_best": difference(best_mrr, first_mrr),
        "val_MRR_drop_best_to_last": difference(best_mrr, last_mrr),
        "train_R@1_gain_first_to_last": difference(last_train_r1, first_train_r1),
        "result_path": str(path),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, default=Path("./7.13/cross_patient_huge_batch"))
    p.add_argument("--output", type=Path, default=Path("./7.13/summary/batch_train_validation_summary.csv"))
    a = p.parse_args(); rows = []
    for case in ("case1", "case2", "case3"):
        for path in (a.results_root / case).glob("*/*/results.json"):
            rows.append(make_row(case, path))
    rows.sort(key=lambda row: (int(row["case"][-1]), row["n"]))
    if len(rows) != 21:
        raise RuntimeError(f"Expected 21 results, found {len(rows)}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {a.output} ({len(rows)} rows)")


if __name__ == "__main__": main()
