#!/usr/bin/env python3
"""Build lateral-only and same-study multi-view n=0..12 pair files.

Outputs are stored beside the established 7.13 pair family:

* ``7.13/pairs/lateral/``: one LATERAL-preferred (otherwise LL) target per study.
* ``7.13/pairs/multiview/``: the original PA-preferred (otherwise AP) target,
  plus every cached AP/PA/LATERAL/LL image from that study as positives.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

OFFSETS = tuple(range(0, 13, 2))
FRONTAL_RANK = {"PA": 0, "AP": 1}
LATERAL_RANK = {"LATERAL": 0, "LL": 1}
POSITIVE_VIEWS = set(FRONTAL_RANK) | set(LATERAL_RANK)


def hours(value):
    return value.toordinal() * 24 + value.hour + value.minute / 60 + value.second / 3600


def open_text(path):
    return gzip.open(path, "rt", newline="") if path.suffix == ".gz" else path.open(newline="")

def load_ids(path):
    return set(map(str, json.loads(path.read_text())))


def load_times(path):
    output = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            output[(row["subject_id"].strip(), row["study_id"].strip())] = \
                datetime.fromisoformat(row["cxr_time"])
    return output


def load_ecgs(path, valid):
    output = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ecg_id = row["record_id"].strip()
            if ecg_id in valid:
                output[row["subject_id"].strip()].append(
                    (datetime.fromisoformat(row["ecg_time"]), ecg_id))
    for rows in output.values():
        rows.sort()
    return output


def load_studies(metadata_path, times, valid):
    studies = defaultdict(list)
    with open_text(metadata_path) as handle:
        for row in csv.DictReader(handle):
            dicom_id = row["dicom_id"].strip()
            patient_id = row["subject_id"].strip()
            study_id = row["study_id"].strip()
            view = row.get("ViewPosition", "").strip().upper()
            timestamp = times.get((patient_id, study_id))
            if dicom_id in valid and view in POSITIVE_VIEWS and timestamp is not None:
                studies[(patient_id, study_id)].append((dicom_id, view, timestamp))
    return studies


def choose(rows, rank):
    candidates = [(rank[v], dicom, v, timestamp)
                  for dicom, v, timestamp in rows if v in rank]
    return min(candidates) if candidates else None


def select_window(rows, target_time, offset, width=12):
    times = [row[0] for row in rows]
    low = target_time - timedelta(hours=offset + width)
    high = target_time - timedelta(hours=offset)
    return rows[bisect_left(times, low):bisect_right(times, high)]


def append_three(targets, base, chosen, target_id, target_time_h, positive_ids=None,
                 positive_views=None):
    ecg_ids = [row[1] for row in chosen]
    ecg_times = [hours(row[0]) for row in chosen]
    extras = {}
    if positive_ids is not None:
        extras = {"cxr_positive_ids": positive_ids,
                  "cxr_positive_views": positive_views}
    targets["single"].extend({
        **base, **extras, "ecg_id": ecg_id, "ecg_time_h": ecg_time,
        "cxr_id": target_id, "cxr_time_h": target_time_h,
        "delta_h": target_time_h - ecg_time,
    } for ecg_id, ecg_time in zip(ecg_ids, ecg_times))
    targets["seq"].append({
        **base, **extras, "t2_h": target_time_h, "cxr_t2": target_id,
        "ecg_ids": ecg_ids, "ecg_times_h": ecg_times,
        "delta_h": target_time_h - ecg_times[-1],
    })
    targets["nearest"].append({
        **base, **extras, "ecg_id": ecg_ids[-1], "ecg_time_h": ecg_times[-1],
        "cxr_id": target_id, "cxr_time_h": target_time_h,
        "delta_h": target_time_h - ecg_times[-1],
    })


def main():
    root = Path(__file__).resolve().parent.parent
    public = Path(os.environ.get("MIMIC_CXR_ROOT", ROOT / "data" / "mimic-cxr-jpg"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--cxr_times", type=Path, default=root / "7.1/cxr_study_times.csv")
    parser.add_argument("--ecg_times", type=Path, default=root / "7.1/ecg_record_times.csv")
    parser.add_argument("--cxr_metadata", type=Path,
                        default=public / "mimic-cxr-2.0.0-metadata.csv.gz")
    parser.add_argument("--cxr_ids", type=Path,
                        default=root / "Waveform_CXR_EHR/ECGCXRPatientTemporal/cache/cxr_ids.json")
    parser.add_argument("--ecg_ids", type=Path,
                        default=root / "Waveform_CXR_EHR/ECGCXRPatientTemporal/cache/ecg_ids.json")
    parser.add_argument("--output_root", type=Path, default=root / "7.13/pairs")
    args = parser.parse_args()

    times = load_times(args.cxr_times)
    ecgs = load_ecgs(args.ecg_times, load_ids(args.ecg_ids))
    studies = load_studies(args.cxr_metadata, times, load_ids(args.cxr_ids))
    lateral_dir = args.output_root / "lateral"
    multiview_dir = args.output_root / "multiview"
    lateral_dir.mkdir(parents=True, exist_ok=True)
    multiview_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for offset in OFFSETS:
        lateral = defaultdict(list)
        multiview = defaultdict(list)
        positive_count = Counter()
        for (patient_id, study_id), images in studies.items():
            patient_ecgs = ecgs.get(patient_id, [])
            if not patient_ecgs:
                continue
            target_time = images[0][2]
            chosen = select_window(patient_ecgs, target_time, offset)
            if not chosen:
                continue
            target_time_h = hours(target_time)
            base = {
                "patient_id": int(patient_id), "study_id": study_id,
                "window_offset_h": offset,
                "window_start_h": target_time_h - offset - 12,
                "window_end_h": target_time_h - offset,
            }

            lateral_target = choose(images, LATERAL_RANK)
            if lateral_target is not None:
                _, dicom_id, view, _ = lateral_target
                append_three(lateral, {**base, "view": view}, chosen,
                             dicom_id, target_time_h)

            frontal_target = choose(images, FRONTAL_RANK)
            if frontal_target is not None:
                _, primary_id, primary_view, _ = frontal_target
                ordered = sorted({(dicom, view) for dicom, view, _ in images},
                                 key=lambda x: (0 if x[0] == primary_id else 1, x[1], x[0]))
                positive_ids = [row[0] for row in ordered]
                positive_views = [row[1] for row in ordered]
                positive_count[len(positive_ids)] += 1
                append_three(multiview, {**base, "view": primary_view}, chosen,
                             primary_id, target_time_h, positive_ids, positive_views)

        stats = {
            "n": offset,
            "lateral_sequence_pairs": len(lateral["seq"]),
            "multiview_sequence_pairs": len(multiview["seq"]),
            "multiview_positive_count_distribution": {
                str(k): v for k, v in sorted(positive_count.items())},
        }
        for kind in ("single", "seq", "nearest"):
            (lateral_dir / f"lateral_{kind}_n{offset}.json").write_text(
                json.dumps({"selection": "LATERAL preferred, otherwise LL", "stats": stats,
                            "pairs": lateral[kind]}))
            (multiview_dir / f"multiview_{kind}_n{offset}.json").write_text(
                json.dumps({"selection": "PA preferred otherwise AP; all same-study views positive",
                            "stats": stats, "pairs": multiview[kind]}))
        summaries.append(stats)
        print(json.dumps(stats), flush=True)

    (args.output_root / "lateral_multiview_summary.json").write_text(
        json.dumps({"stats": summaries}, indent=2))


if __name__ == "__main__":
    main()
