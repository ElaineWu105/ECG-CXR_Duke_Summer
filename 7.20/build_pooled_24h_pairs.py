#!/usr/bin/env python3
"""Merge n=0,2,...,12 sequence files into one unique 0-24h sequence per CXR."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=Path, default=root / "7.13/pairs/seq")
    ap.add_argument("--output", type=Path, default=root / "7.20/pairs/seq_pooled_0_24h.json")
    return ap.parse_args()


def main():
    args = parse_args()
    groups = {}
    source_counts = {}
    for n in range(0, 13, 2):
        path = args.input_dir / f"seq_n{n}.json"
        rows = json.loads(path.read_text())["pairs"]
        source_counts[str(n)] = len(rows)
        for row in rows:
            key = (int(row["patient_id"]), str(row["cxr_t2"]))
            if key not in groups:
                groups[key] = {
                    "patient_id": int(row["patient_id"]),
                    "study_id": str(row["study_id"]),
                    "view": row.get("view"),
                    "window_offset_h": 0,
                    "window_start_h": float(row["t2_h"]) - 24.0,
                    "window_end_h": float(row["t2_h"]),
                    "t2_h": float(row["t2_h"]),
                    "cxr_t2": str(row["cxr_t2"]),
                    "ecgs": {},
                    "source_n": set(),
                }
            group = groups[key]
            group["source_n"].add(n)
            for ecg_id, ecg_time in zip(row["ecg_ids"], row["ecg_times_h"]):
                horizon = group["t2_h"] - float(ecg_time)
                if -1e-6 <= horizon <= 24.0 + 1e-6:
                    group["ecgs"][str(ecg_id)] = float(ecg_time)

    output = []
    for group in groups.values():
        ordered = sorted(group.pop("ecgs").items(), key=lambda x: (x[1], x[0]))
        if not ordered:
            continue
        group["ecg_ids"] = [x[0] for x in ordered]
        group["ecg_times_h"] = [x[1] for x in ordered]
        group["delta_h"] = group["t2_h"] - ordered[-1][1]
        group["source_n"] = sorted(group["source_n"])
        output.append(group)
    output.sort(key=lambda x: (x["patient_id"], x["t2_h"], x["cxr_t2"]))

    lengths = Counter(len(x["ecg_ids"]) for x in output)
    summary = {
        "definition": "one unique target CXR with all ECGs in [t2-24h, t2]",
        "source_counts": source_counts,
        "n_unique_pairs": len(output),
        "n_unique_patients": len({x["patient_id"] for x in output}),
        "sequence_length_distribution": {str(k): v for k, v in sorted(lengths.items())},
        "fraction_multi_ecg": sum(v for k, v in lengths.items() if k > 1) / len(output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": summary, "pairs": output}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
