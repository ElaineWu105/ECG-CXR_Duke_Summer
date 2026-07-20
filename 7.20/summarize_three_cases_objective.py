#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.results_root.glob("case*/*/results.json"):
        match = re.search(r"case([123])_.*_n(\d+)$", path.parent.name)
        if not match:
            continue
        result = json.loads(path.read_text())
        cross = result["test"]["cross_patient"]
        temporal = result["test"].get("temporal", {})
        rows.append({
            "case": int(match.group(1)), "n": int(match.group(2)),
            "experiment": path.parent.name,
            "best_epoch": result["best_epoch"],
            "best_val_monitor": result["best_val_monitor"],
            "test_r1": cross["recall@1"], "test_r5": cross["recall@5"],
            "test_r10": cross["recall@10"], "test_mrr": cross["mrr"],
            "test_median_rank": cross["median_rank"],
            "temporal_r1": temporal.get("temporal_recall@1", ""),
            "temporal_mrr": temporal.get("temporal_mrr", ""),
        })
    rows.sort(key=lambda row: (row["case"], row["n"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"summarized {len(rows)} runs -> {args.output}")


if __name__ == "__main__":
    main()
