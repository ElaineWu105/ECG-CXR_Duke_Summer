#!/usr/bin/env python3
"""Pool multiview n=0..12 pairs into unique 0-24h target sequences."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=root / "7.13/pairs/multiview")
    parser.add_argument("--output", type=Path,
                        default=root / "7.20/pairs/seq_pooled_0_24h_multiview.json")
    args = parser.parse_args()
    groups = {}
    for n in range(0, 13, 2):
        rows = json.loads((args.input_dir / f"multiview_seq_n{n}.json").read_text())["pairs"]
        for row in rows:
            key = (int(row["patient_id"]), str(row["cxr_t2"]))
            group = groups.setdefault(key, {
                "patient_id": int(row["patient_id"]), "study_id": str(row["study_id"]),
                "view": row["view"], "window_offset_h": 0,
                "window_start_h": float(row["t2_h"]) - 24,
                "window_end_h": float(row["t2_h"]), "t2_h": float(row["t2_h"]),
                "cxr_t2": str(row["cxr_t2"]), "ecgs": {}, "positives": {},
                "source_n": set(),
            })
            group["source_n"].add(n)
            for ecg_id, ecg_time in zip(row["ecg_ids"], row["ecg_times_h"]):
                horizon = group["t2_h"] - float(ecg_time)
                if -1e-6 <= horizon <= 24 + 1e-6:
                    group["ecgs"][str(ecg_id)] = float(ecg_time)
            for dicom_id, view in zip(row["cxr_positive_ids"], row["cxr_positive_views"]):
                group["positives"][str(dicom_id)] = str(view)

    output = []
    for group in groups.values():
        ecgs = sorted(group.pop("ecgs").items(), key=lambda x: (x[1], x[0]))
        positives = group.pop("positives")
        primary = group["cxr_t2"]
        positive_rows = sorted(positives.items(),
                               key=lambda x: (0 if x[0] == primary else 1, x[1], x[0]))
        group["ecg_ids"] = [x[0] for x in ecgs]
        group["ecg_times_h"] = [x[1] for x in ecgs]
        group["delta_h"] = group["t2_h"] - ecgs[-1][1]
        group["cxr_positive_ids"] = [x[0] for x in positive_rows]
        group["cxr_positive_views"] = [x[1] for x in positive_rows]
        group["source_n"] = sorted(group["source_n"])
        output.append(group)
    output.sort(key=lambda x: (x["patient_id"], x["t2_h"], x["cxr_t2"]))
    counts = Counter(len(x["cxr_positive_ids"]) for x in output)
    metadata = {
        "definition": "unique [t2-24h,t2] ECG sequence; all cached same-study views positive",
        "n_pairs": len(output),
        "positive_count_distribution": {str(k): v for k, v in sorted(counts.items())},
        "fraction_with_extra_positive": sum(v for k, v in counts.items() if k > 1) / len(output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "pairs": output}))
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
