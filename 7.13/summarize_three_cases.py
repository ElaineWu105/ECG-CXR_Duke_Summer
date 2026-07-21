#!/usr/bin/env python3
"""Write one CSV table per 7.13 cross-patient case."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FIELDS = [
    "n", "target_window", "experiment_name", "input_type", "loss_mode",
    "train_samples", "train_patients", "val_samples", "val_patients",
    "test_samples", "test_patients", "eligible_train_patients",
    "batch_n_patients", "k_intervals", "best_epoch", "best_val_mrr",
    "test_queries", "test_gallery_size", "test_recall_at_1",
    "test_recall_at_5", "test_recall_at_10", "test_mrr",
    "test_median_rank", "n_trainable_params", "result_path",
]


def nested(data, *keys, default=""):
    value = data
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def read_row(path: Path) -> dict:
    with path.open() as handle:
        data = json.load(handle)
    spec = data.get("spec", {})
    match = re.search(r"_n(\d+)$", str(spec.get("name", "")))
    if not match:
        raise ValueError(f"Cannot determine n from experiment name in {path}")
    split = data.get("split_temporal_negative_stats", {})
    sampler = data.get("sampler_config", {})
    test = nested(data, "test", "cross_patient", default={})
    return {
        "n": int(match.group(1)),
        "target_window": spec.get("target_window", ""),
        "experiment_name": spec.get("name", ""),
        "input_type": spec.get("ecg_mode", ""),
        "loss_mode": data.get("loss_mode", spec.get("loss_mode", "")),
        "train_samples": nested(split, "train", "n_queries"),
        "train_patients": nested(split, "train", "n_patients"),
        "val_samples": nested(split, "val", "n_queries"),
        "val_patients": nested(split, "val", "n_patients"),
        "test_samples": nested(split, "test", "n_queries"),
        "test_patients": nested(split, "test", "n_patients"),
        "eligible_train_patients": sampler.get("eligible_train_patients", ""),
        "batch_n_patients": sampler.get("n_patients", ""),
        "k_intervals": sampler.get("k_intervals", ""),
        "best_epoch": data.get("best_epoch", ""),
        "best_val_mrr": data.get("best_val_monitor", ""),
        "test_queries": test.get("n_queries", ""),
        "test_gallery_size": test.get("gallery_size", ""),
        "test_recall_at_1": test.get("recall@1", ""),
        "test_recall_at_5": test.get("recall@5", ""),
        "test_recall_at_10": test.get("recall@10", ""),
        "test_mrr": test.get("mrr", ""),
        "test_median_rank": test.get("median_rank", ""),
        "n_trainable_params": data.get("n_trainable_params", ""),
        "result_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_root", type=Path,
        default=Path("./7.13/cross_patient_huge_batch"),
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("./7.13/summary"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "case1": "case1_all_ecg_cross_n*",
        "case2": "case2_sequence_ecg_cross_n*",
        "case3": "case3_nearest_ecg_cross_n*",
    }
    for case, experiment_pattern in cases.items():
        paths = sorted(
            (args.results_root / case).glob(f"*/{experiment_pattern}/results.json")
        )
        rows = sorted((read_row(path) for path in paths), key=lambda row: row["n"])
        expected = {0, 2, 4, 6, 8, 10, 12}
        found = {row["n"] for row in rows}
        if found != expected:
            raise RuntimeError(f"{case}: expected n={sorted(expected)}, found n={sorted(found)}")
        output = args.output_dir / f"{case}_summary.csv"
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {output} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
