import json
import numpy as np
from pathlib import Path

PROJECT_DIR = Path.home() / "ECG+EHR+CXR/Waveform_CXR_EHR/ECGCXRPatientTemporal"

SEQ_PATH = PROJECT_DIR / "cache/seq_target_pairs.json"
EXP4_PATH = PROJECT_DIR / "cache/patient_temporal_pairs.json"


def summarize_seq_target(path):
    print("\n==============================")
    print("Exp3 seq_target:", path)
    data = json.load(open(path))
    pairs = data["pairs"]
    print("num pairs:", len(pairs))

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

    print("ECGs per sample:")
    print("  min/median/mean/max:",
          n_ecgs.min(), np.median(n_ecgs), n_ecgs.mean(), n_ecgs.max())

    print("ECG distance to CXR_t2, hours:")
    print("  min/median/mean/max:",
          delta_to_t2.min(), np.median(delta_to_t2), delta_to_t2.mean(), delta_to_t2.max())

    print("ECGs within 12-24h before t2:",
          np.mean((delta_to_t2 >= 12) & (delta_to_t2 <= 24)))

    if len(close_gaps):
        print("Adjacent ECG gaps within same sample, hours:")
        print("  min/median/mean/max:",
              close_gaps.min(), np.median(close_gaps), close_gaps.mean(), close_gaps.max())
        print("  gap < 10 min:", np.mean(close_gaps < 10 / 60))
        print("  gap < 30 min:", np.mean(close_gaps < 30 / 60))
        print("  gap < 1 hour:", np.mean(close_gaps < 1))
    else:
        print("No samples with >=2 ECGs.")


def summarize_exp4(path):
    print("\n==============================")
    print("Exp4 patient_temporal:", path)
    data = json.load(open(path))
    pairs = data["pairs"]
    print("num pairs:", len(pairs))

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

    print("CXR interval t2 - t1, hours:")
    print("  min/median/mean/max:",
          dt.min(), np.median(dt), dt.mean(), dt.max())
    print("  within 3-48h:",
          np.mean((dt >= 3) & (dt <= 48)))

    print("ECGs per sample:")
    print("  min/median/mean/max:",
          n_ecgs.min(), np.median(n_ecgs), n_ecgs.mean(), n_ecgs.max())

    print("ECG after t1, hours:")
    print("  min/median/mean/max:",
          ecg_after_t1.min(), np.median(ecg_after_t1), ecg_after_t1.mean(), ecg_after_t1.max())

    print("ECG before t2, hours:")
    print("  min/median/mean/max:",
          ecg_before_t2.min(), np.median(ecg_before_t2), ecg_before_t2.mean(), ecg_before_t2.max())

    print("ECG inside (t1, t2]:",
          np.mean((ecg_after_t1 > 0) & (ecg_before_t2 >= 0)))

    print("ECG relative position in interval:")
    print("  min/median/mean/max:",
          rel_pos.min(), np.median(rel_pos), rel_pos.mean(), rel_pos.max())

    if len(close_gaps):
        print("Adjacent ECG gaps within same sample, hours:")
        print("  min/median/mean/max:",
              close_gaps.min(), np.median(close_gaps), close_gaps.mean(), close_gaps.max())
        print("  gap < 10 min:", np.mean(close_gaps < 10 / 60))
        print("  gap < 30 min:", np.mean(close_gaps < 30 / 60))
        print("  gap < 1 hour:", np.mean(close_gaps < 1))
    else:
        print("No samples with >=2 ECGs.")


def main():
    summarize_seq_target(SEQ_PATH)
    summarize_exp4(EXP4_PATH)


if __name__ == "__main__":
    main()
