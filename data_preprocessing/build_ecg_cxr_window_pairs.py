"""Build CXR-anchored ECG-only window pairs.

For each target CXR at time t2, collect same-patient ECGs in windows

    [t2 - window_width_hours - n, t2 - n]

for n in a configurable offset list (default 0,2,4,6,8,10,12 hours).
The script writes both:

  * single_n{n}.json: one ECG -> CXR_t2 pairs
  * seq_n{n}.json   : ECG sequence in the window -> CXR_t2 pairs

These files intentionally contain no CXR_t1, so they test ECG-only alignment.
They are compatible with run_experiments.py via --single_pairs or
--seq_target_pairs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from build_pairs import load_cxr_nodes, load_ecg_nodes  # noqa: E402


def _parse_offsets(raw: str) -> list[float]:
    vals = []
    for tok in str(raw).replace(",", " ").split():
        vals.append(float(tok))
    if not vals:
        raise ValueError("--offset_hours must contain at least one value")
    return vals


def _fmt_offset(x: float) -> str:
    if float(x).is_integer():
        return str(int(x))
    return str(x).replace(".", "p")


def _summarize(name: str, pairs: list, key: str = "patient_id"):
    print(f"  {name}: {len(pairs):,} pairs")
    if not pairs:
        return
    per_pat = pd.Series([p[key] for p in pairs]).value_counts()
    print(
        f"    patients={per_pat.size:,} pairs/patient: "
        f"min={per_pat.min()} median={int(per_pat.median())} max={per_pat.max()}"
    )


def build_for_offset(cxr_nodes: dict, ecg_nodes: dict, offset_h: float, args):
    single_pairs = []
    seq_pairs = []
    cxr_meta: dict[str, dict] = {}
    ecg_meta: dict[str, dict] = {}
    n_targets = 0
    n_targets_with_ecg = 0
    n_targets_no_ecg = 0

    for sid, cseq in cxr_nodes.items():
        eseq = ecg_nodes.get(sid)
        if not eseq:
            continue
        e_times = np.array([e["t_h"] for e in eseq], dtype=np.float64)
        for cnode in cseq:
            n_targets += 1
            t2 = float(cnode["t_h"])
            start = t2 - float(args.window_width_hours) - float(offset_h)
            end = t2 - float(offset_h)
            lo = np.searchsorted(e_times, start, side="left")
            hi = np.searchsorted(e_times, end, side="right")
            idxs = list(range(lo, hi))
            if len(idxs) < args.min_ecgs_per_window:
                n_targets_no_ecg += 1
                continue
            if args.max_ecgs_per_window > 0 and len(idxs) > args.max_ecgs_per_window:
                idxs = idxs[-args.max_ecgs_per_window:]
            n_targets_with_ecg += 1

            cxr_id = cnode["dicom_id"]
            cxr_meta.setdefault(cxr_id, {"path": cnode["path"], "path_ok": cnode["path_ok"]})
            ecg_ids = []
            ecg_times = []
            for k in idxs:
                e = eseq[k]
                eid = e["ecg_id"]
                et = float(e["t_h"])
                ecg_ids.append(eid)
                ecg_times.append(et)
                ecg_meta.setdefault(eid, {"path": e["path"]})
                single_pairs.append({
                    "patient_id": int(sid),
                    "ecg_id": eid,
                    "ecg_time_h": et,
                    "cxr_id": cxr_id,
                    "cxr_time_h": t2,
                    "delta_h": float(t2 - et),
                    "window_offset_h": float(offset_h),
                    "window_start_h": float(start),
                    "window_end_h": float(end),
                })

            seq_pairs.append({
                "patient_id": int(sid),
                "t2_h": t2,
                "cxr_t2": cxr_id,
                "ecg_ids": ecg_ids,
                "ecg_times_h": ecg_times,
                "delta_h": float(offset_h),
                "window_offset_h": float(offset_h),
                "window_start_h": float(start),
                "window_end_h": float(end),
            })

    stats = {
        "window_offset_h": float(offset_h),
        "window_width_h": float(args.window_width_hours),
        "window_definition": "[t2 - window_width_h - offset_h, t2 - offset_h]",
        "n_targets_considered": int(n_targets),
        "n_targets_with_ecg": int(n_targets_with_ecg),
        "n_targets_no_ecg": int(n_targets_no_ecg),
        "n_single_pairs": int(len(single_pairs)),
        "n_sequence_pairs": int(len(seq_pairs)),
        "n_unique_cxr": int(len(cxr_meta)),
        "n_unique_ecg": int(len(ecg_meta)),
        "min_ecgs_per_window": int(args.min_ecgs_per_window),
        "max_ecgs_per_window": int(args.max_ecgs_per_window),
    }
    return single_pairs, seq_pairs, cxr_meta, ecg_meta, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cxr_csv", default=C.FULL_CXR_CATALOG_CSV)
    ap.add_argument("--ecg_csv", default=C.FULL_ECG_CATALOG_CSV)
    ap.add_argument("--metadata_path", default=C.CXR_METADATA_PATH)
    ap.add_argument("--cxr_root", default=C.CXR_ROOT)
    ap.add_argument("--out_dir", default=str(C.CACHE_ROOT / "ecg_cxr_windows"))
    ap.add_argument("--offset_hours", default="0,2,4,6,8,10,12")
    ap.add_argument("--window_width_hours", type=float, default=12.0)
    ap.add_argument("--min_ecgs_per_window", type=int, default=1)
    ap.add_argument("--max_ecgs_per_window", type=int, default=C.MAX_ECGS_PER_INTERVAL,
                    help="Cap ECGs per sequence window; <=0 keeps all ECGs.")
    ap.add_argument("--require_cxr_on_disk", action="store_true")
    ap.add_argument("--skip_cxr_path_check", action="store_true")
    args = ap.parse_args()

    offsets = _parse_offsets(args.offset_hours)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== build_ecg_cxr_window_pairs: CXR-anchored ECG-only windows ===")
    print(f"  window: [t2 - {args.window_width_hours:g}h - n, t2 - n]")
    print("  offsets n: " + ", ".join(f"{x:g}h" for x in offsets))

    cxr_nodes = load_cxr_nodes(
        args.cxr_csv, args.metadata_path, args.cxr_root,
        min_cxrs=1, check_paths=not args.skip_cxr_path_check)
    ecg_nodes = load_ecg_nodes(args.ecg_csv)

    all_stats = []
    for offset_h in offsets:
        label = _fmt_offset(offset_h)
        print(f"\n--- n={offset_h:g}h ---")
        single_pairs, seq_pairs, cxr_meta, ecg_meta, stats = build_for_offset(
            cxr_nodes, ecg_nodes, offset_h, args)
        if args.require_cxr_on_disk:
            before_single, before_seq = len(single_pairs), len(seq_pairs)
            single_pairs = [p for p in single_pairs if cxr_meta.get(p["cxr_id"], {}).get("path_ok")]
            seq_pairs = [p for p in seq_pairs if cxr_meta.get(p["cxr_t2"], {}).get("path_ok")]
            print(f"  require_cxr_on_disk: single {len(single_pairs):,}/{before_single:,}, "
                  f"seq {len(seq_pairs):,}/{before_seq:,}")
            stats["n_single_pairs"] = int(len(single_pairs))
            stats["n_sequence_pairs"] = int(len(seq_pairs))
        _summarize("single", single_pairs)
        _summarize("sequence", seq_pairs)

        single_path = out_dir / f"single_n{label}.json"
        seq_path = out_dir / f"seq_n{label}.json"
        payload_meta = {
            "stats": stats,
            "cxr_meta": cxr_meta,
            "ecg_meta": ecg_meta,
        }
        with open(single_path, "w") as f:
            json.dump({"pairs": single_pairs, **payload_meta}, f)
        with open(seq_path, "w") as f:
            json.dump({"pairs": seq_pairs, **payload_meta}, f)
        print(f"  wrote {single_path}")
        print(f"  wrote {seq_path}")
        stats = dict(stats)
        stats["single_path"] = str(single_path)
        stats["sequence_path"] = str(seq_path)
        all_stats.append(stats)

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"offsets": offsets, "stats": all_stats}, f, indent=2)
    print(f"\nWrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
