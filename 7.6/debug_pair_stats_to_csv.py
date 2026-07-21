import json
import csv
import numpy as np
from pathlib import Path

PROJECT_DIR = Path.home() / "ECG+EHR+CXR/Waveform_CXR_EHR/ECGCXRPatientTemporal"

SEQ_PATH = PROJECT_DIR / "cache/seq_target_pairs.json"
EXP4_PATH = PROJECT_DIR / "cache/patient_temporal_pairs.json"
OUT_CSV = Path.home() / "7.6/pair_stats_summary.csv"


def add_row(rows, experiment, metric, value):
    rows.append({
        "experiment": experiment,
        "metric": metric,
        "value": value,
    })


def summarize_seq_target(path, rows):
    experiment = "Exp3_seq_target"

    data = json.load(open(path))
    pairs = data["pairs"]

    n_ecgs = np.array([len(p["ecg_ids"]) for p in pairs])
    delta_to_t2 = []
    close_gaps = []

    for p in pairs:
        t2 = p["t2_h"]
        ets = sorted(p["ecg_times_h"])
        delta_to_t2.extend([t2 - e for e in ets])

        if len(ets) >= 2:
            close_gaps.extend(np.diff(ets))

    delta_to_t2 = np.array(delta_to_t2)
    close_gaps = np.array(close_gaps) if close_gaps else np.array([])

    add_row(rows, experiment, "num_pairs", len(pairs))

    add_row(rows, experiment, "ecgs_per_sample_min", n_ecgs.min())
    add_row(rows, experiment, "ecgs_per_sample_median", np.median(n_ecgs))
    add_row(rows, experiment, "ecgs_per_sample_mean", n_ecgs.mean())
    add_row(rows, experiment, "ecgs_per_sample_max", n_ecgs.max())

    add_row(rows, experiment, "ecg_distance_to_cxr_t2_hours_min", delta_to_t2.min())
    add_row(rows, experiment, "ecg_distance_to_cxr_t2_hours_median", np.median(delta_to_t2))
    add_row(rows, experiment, "ecg_distance_to_cxr_t2_hours_mean", delta_to_t2.mean())
    add_row(rows, experiment, "ecg_distance_to_cxr_t2_hours_max", delta_to_t2.max())

    add_row(
        rows,
        experiment,
        "ecgs_within_12_24h_before_t2",
        np.mean((delta_to_t2 >= 12) & (delta_to_t2 <= 24)),
    )

    if len(close_gaps):
        add_row(rows, experiment, "adjacent_ecg_gap_hours_min", close_gaps.min())
        add_row(rows, experiment, "adjacent_ecg_gap_hours_median", np.median(close_gaps))
        add_row(rows, experiment, "adjacent_ecg_gap_hours_mean", close_gaps.mean())
        add_row(rows, experiment, "adjacent_ecg_gap_hours_max", close_gaps.max())

        add_row(rows, experiment, "adjacent_ecg_gap_lt_10min", np.mean(close_gaps < 10 / 60))
        add_row(rows, experiment, "adjacent_ecg_gap_lt_30min", np.mean(close_gaps < 30 / 60))
        add_row(rows, experiment, "adjacent_ecg_gap_lt_1hour", np.mean(close_gaps < 1))
    else:
        add_row(rows, experiment, "has_samples_with_multiple_ecgs", 0)


def summarize_exp4(path, rows):
    experiment = "Exp4_patient_temporal"

    data = json.load(open(path))
    pairs = data["pairs"]

    n_ecgs = np.array([len(p["ecg_ids"]) for p in pairs])
    dt = np.array([p["t2_h"] - p["t1_h"] for p in pairs])

    ecg_after_t1 = []
    ecg_before_t2 = []
    rel_pos = []
    close_gaps = []

    for p in pairs:
        t1, t2 = p["t1_h"], p["t2_h"]
        ets = sorted(p["ecg_times_h"])

        ecg_after_t1.extend([e - t1 for e in ets])
        ecg_before_t2.extend([t2 - e for e in ets])
        rel_pos.extend([(e - t1) / (t2 - t1) for e in ets])

        if len(ets) >= 2:
            close_gaps.extend(np.diff(ets))

    ecg_after_t1 = np.array(ecg_after_t1)
    ecg_before_t2 = np.array(ecg_before_t2)
    rel_pos = np.array(rel_pos)
    close_gaps = np.array(close_gaps) if close_gaps else np.array([])

    add_row(rows, experiment, "num_pairs", len(pairs))

    add_row(rows, experiment, "cxr_interval_t2_minus_t1_hours_min", dt.min())
    add_row(rows, experiment, "cxr_interval_t2_minus_t1_hours_median", np.median(dt))
    add_row(rows, experiment, "cxr_interval_t2_minus_t1_hours_mean", dt.mean())
    add_row(rows, experiment, "cxr_interval_t2_minus_t1_hours_max", dt.max())
    add_row(rows, experiment, "cxr_interval_within_3_48h", np.mean((dt >= 3) & (dt <= 48)))

    add_row(rows, experiment, "ecgs_per_sample_min", n_ecgs.min())
    add_row(rows, experiment, "ecgs_per_sample_median", np.median(n_ecgs))
    add_row(rows, experiment, "ecgs_per_sample_mean", n_ecgs.mean())
    add_row(rows, experiment, "ecgs_per_sample_max", n_ecgs.max())

    add_row(rows, experiment, "ecg_after_t1_hours_min", ecg_after_t1.min())
    add_row(rows, experiment, "ecg_after_t1_hours_median", np.median(ecg_after_t1))
    add_row(rows, experiment, "ecg_after_t1_hours_mean", ecg_after_t1.mean())
    add_row(rows, experiment, "ecg_after_t1_hours_max", ecg_after_t1.max())

    add_row(rows, experiment, "ecg_before_t2_hours_min", ecg_before_t2.min())
    add_row(rows, experiment, "ecg_before_t2_hours_median", np.median(ecg_before_t2))
    add_row(rows, experiment, "ecg_before_t2_hours_mean", ecg_before_t2.mean())
    add_row(rows, experiment, "ecg_before_t2_hours_max", ecg_before_t2.max())

    add_row(
        rows,
        experiment,
        "ecg_inside_t1_t2",
        np.mean((ecg_after_t1 > 0) & (ecg_before_t2 >= 0)),
    )

    add_row(rows, experiment, "ecg_relative_position_min", rel_pos.min())
    add_row(rows, experiment, "ecg_relative_position_median", np.median(rel_pos))
    add_row(rows, experiment, "ecg_relative_position_mean", rel_pos.mean())
    add_row(rows, experiment, "ecg_relative_position_max", rel_pos.max())

    if len(close_gaps):
        add_row(rows, experiment, "adjacent_ecg_gap_hours_min", close_gaps.min())
        add_row(rows, experiment, "adjacent_ecg_gap_hours_median", np.median(close_gaps))
        add_row(rows, experiment, "adjacent_ecg_gap_hours_mean", close_gaps.mean())
        add_row(rows, experiment, "adjacent_ecg_gap_hours_max", close_gaps.max())

        add_row(rows, experiment, "adjacent_ecg_gap_lt_10min", np.mean(close_gaps < 10 / 60))
        add_row(rows, experiment, "adjacent_ecg_gap_lt_30min", np.mean(close_gaps < 30 / 60))
        add_row(rows, experiment, "adjacent_ecg_gap_lt_1hour", np.mean(close_gaps < 1))
    else:
        add_row(rows, experiment, "has_samples_with_multiple_ecgs", 0)


def main():
    rows = []

    summarize_seq_target(SEQ_PATH, rows)
    summarize_exp4(EXP4_PATH, rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV: {OUT_CSV}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
