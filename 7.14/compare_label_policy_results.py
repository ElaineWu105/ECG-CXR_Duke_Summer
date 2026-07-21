#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


ROOT = Path("./7.14/label_policy_comparison")
KEY = ["n_offset_hours", "model", "split"]


def main() -> None:
    explicit = pd.read_csv(ROOT / "explicit_01" / "all_results_summary.csv")
    all_cohort = pd.read_csv(
        ROOT / "all_nonpositive_negative" / "all_results_summary.csv"
    )
    explicit.insert(0, "label_policy", "explicit_01")
    all_cohort.insert(0, "label_policy", "all_nonpositive_negative")
    pd.concat([explicit, all_cohort], ignore_index=True).to_csv(
        ROOT / "combined_results.csv", index=False
    )

    metrics = [
        column for column in explicit.columns
        if column.endswith(("_auroc", "_auprc", "_prevalence"))
    ]
    left = explicit[KEY + metrics].set_index(KEY)
    right = all_cohort[KEY + metrics].set_index(KEY)
    difference = right.subtract(left).add_suffix(
        "_difference_all_minus_explicit"
    ).reset_index()
    difference.to_csv(ROOT / "metric_differences.csv", index=False)


if __name__ == "__main__":
    main()
