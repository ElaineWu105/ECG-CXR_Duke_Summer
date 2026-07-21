#!/usr/bin/env python3
"""Count unique patients with six CXR findings in the full cohort and 7 windows."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(".")
CXR_TIMES = ROOT / "7.1/cxr_study_times.csv"
ECG_TIMES = ROOT / "7.1/ecg_record_times.csv"
PAIR_ROOT = ROOT / "7.13/pairs/seq"
LABEL_CSV = Path("/path/to/mimic_cxr/mimic_cxr_jpg/mimic-cxr-2.0.0-chexpert.csv.gz")
OUTPUT = ROOT / "7.14/six_label_patient_counts_full_and_7_windows.csv"
N_VALUES = (0, 2, 4, 6, 8, 10, 12)

LABELS = (
    ("pneumonia", "Pneumonia"),
    ("consolidation", "Consolidation"),
    ("lung_opacity", "Lung Opacity"),
    ("pneumothorax", "Pneumothorax"),
    ("pleural_effusion", "Pleural Effusion"),
    ("pulmonary_edema", "Edema"),
)


def load_labels() -> pd.DataFrame:
    columns = [source for _, source in LABELS]
    frame = pd.read_csv(LABEL_CSV, usecols=["subject_id", "study_id", *columns])
    return frame.drop_duplicates(["subject_id", "study_id"])


def full_intersection_studies() -> tuple[pd.DataFrame, set[int]]:
    """All CXR studies from patients having any ECG and any CXR; no time filter."""
    cxr = pd.read_csv(CXR_TIMES, usecols=["subject_id", "study_id"])
    ecg = pd.read_csv(ECG_TIMES, usecols=["subject_id"])
    cxr["subject_id"] = cxr["subject_id"].astype(int)
    ecg["subject_id"] = ecg["subject_id"].astype(int)
    common = set(cxr["subject_id"].unique()) & set(ecg["subject_id"].unique())
    if len(common) != 54_362:
        raise RuntimeError(f"Expected 54,362 ECG+CXR patients, found {len(common):,}")
    studies = cxr[cxr["subject_id"].isin(common)].drop_duplicates(["subject_id", "study_id"])
    return studies, common


def window_studies(n: int) -> tuple[pd.DataFrame, set[int]]:
    """CXR studies with at least one ECG in [t2-n-12h, t2-n]."""
    payload = json.loads((PAIR_ROOT / f"seq_n{n}.json").read_text())
    frame = pd.DataFrame({
        "subject_id": [int(row["patient_id"]) for row in payload["pairs"]],
        "study_id": [int(row["study_id"]) for row in payload["pairs"]],
    }).drop_duplicates(["subject_id", "study_id"])
    return frame, set(frame["subject_id"].unique())


def summarize(scope: str, n: int | None, studies: pd.DataFrame,
              cohort_patients: set[int], labels: pd.DataFrame) -> list[dict]:
    data = studies.merge(labels, on=["subject_id", "study_id"], how="left", validate="one_to_one")
    rows = []
    for short_name, source_name in LABELS:
        positive_studies = data[data[source_name] == 1.0]
        positive_patients = set(positive_studies["subject_id"].astype(int))
        explicit = data[data[source_name].isin([0.0, 1.0])]
        explicit_patients = set(explicit["subject_id"].astype(int))
        rows.append({
            "scope": scope,
            "n_offset_hours": "" if n is None else n,
            "window": "no temporal restriction" if n is None else f"[t2-{n}-12h, t2-{n}]",
            "label": short_name,
            "chexpert_column": source_name,
            "positive_patient_definition": "unique patient with >=1 CXR study labeled exactly 1",
            "positive_patients": len(positive_patients),
            "cohort_patients": len(cohort_patients),
            "positive_patient_percent_of_cohort": 100.0 * len(positive_patients) / len(cohort_patients),
            "patients_with_explicit_0_or_1": len(explicit_patients),
            "positive_percent_among_explicit_patients": (
                100.0 * len(positive_patients) / len(explicit_patients) if explicit_patients else ""
            ),
            "positive_cxr_studies": len(positive_studies),
            "cohort_cxr_studies": len(studies),
        })
    return rows


def main():
    labels = load_labels()
    rows = []
    studies, patients = full_intersection_studies()
    rows.extend(summarize("full_intersection", None, studies, patients, labels))
    for n in N_VALUES:
        studies, patients = window_studies(n)
        rows.extend(summarize(f"window_n{n}", n, studies, patients, labels))

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows: {OUTPUT}")


if __name__ == "__main__":
    main()
