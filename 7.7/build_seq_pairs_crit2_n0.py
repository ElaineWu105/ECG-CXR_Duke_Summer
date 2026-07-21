"""Build Exp3 + Exp4 pairs with frontal CXR and Mehak crit2 n=0.

Exp 3 keeps the original seq_target logic:
    each CXR_t2 with ECGs in [t2 - lookback, t2 - min_horizon].

Exp 4 uses crit2 n=0:
    For each target CXR_t2:
      1) choose CXR_t1 from [t2 - 24h, t2)
      2) t1_strategy controls which t1 to choose:
           nearest  = closest CXR before t2
           earliest = earliest CXR within the 24h window
      3) ECG window:
           [max(t2 - 12h, t1), t2]

Sample format stays compatible with dataset.py:
    {patient_id, t1_h, t2_h, cxr_t1, cxr_t2, ecg_ids, ecg_times_h, delta_h}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Original experiment directory
EXP_DIR = Path("./Waveform_CXR_EHR/ECGCXRPatientTemporal")
sys.path.insert(0, str(EXP_DIR))

import config as C  # noqa: E402
from build_pairs import load_cxr_nodes, load_ecg_nodes  # noqa: E402


def _cap_recent(idxs: list, max_ecgs: int) -> list:
    return idxs[-max_ecgs:] if len(idxs) > max_ecgs else idxs


def _summarize_exp4_pairs(pairs: list[dict], out_path: Path):
    lines = []
    lines.append(f"Output: {out_path}")
    lines.append(f"Pairs: {len(pairs):,}")

    if not pairs:
        out_path.write_text("\n".join(lines) + "\n")
        return

    pat_counts = pd.Series([p["patient_id"] for p in pairs]).value_counts()
    n_ecg = np.array([len(p["ecg_ids"]) for p in pairs], dtype=np.float64)
    dt = np.array([p["delta_h"] for p in pairs], dtype=np.float64)

    # ECG window length = last - first if >=2 ECGs; otherwise 0
    ecg_span = []
    last_gap_to_t2 = []
    for p in pairs:
        ts = p["ecg_times_h"]
        if len(ts) >= 2:
            ecg_span.append(max(ts) - min(ts))
        else:
            ecg_span.append(0.0)
        last_gap_to_t2.append(p["t2_h"] - max(ts))

    ecg_span = np.array(ecg_span, dtype=np.float64)
    last_gap_to_t2 = np.array(last_gap_to_t2, dtype=np.float64)

    lines.append(f"Patients: {pat_counts.size:,}")
    lines.append(f"Pairs/patient median: {int(pat_counts.median())}")
    lines.append(f"Pairs/patient max: {int(pat_counts.max())}")
    lines.append("")
    lines.append("ECGs per sample:")
    lines.append(f"  min/median/mean/max: {int(n_ecg.min())} {np.median(n_ecg):.1f} {n_ecg.mean():.3f} {int(n_ecg.max())}")
    lines.append("")
    lines.append("CXR delta t2 - t1, hours:")
    lines.append(f"  min/median/mean/max: {dt.min():.3f} {np.median(dt):.3f} {dt.mean():.3f} {dt.max():.3f}")
    lines.append(f"  p25/p75/p90/p95/p99: {np.percentile(dt,25):.3f} {np.percentile(dt,75):.3f} {np.percentile(dt,90):.3f} {np.percentile(dt,95):.3f} {np.percentile(dt,99):.3f}")
    lines.append("")
    lines.append("ECG span within sample, hours:")
    lines.append(f"  min/median/mean/max: {ecg_span.min():.3f} {np.median(ecg_span):.3f} {ecg_span.mean():.3f} {ecg_span.max():.3f}")
    lines.append("")
    lines.append("Gap from most recent ECG to CXR_t2, hours:")
    lines.append(f"  min/median/mean/max: {last_gap_to_t2.min():.3f} {np.median(last_gap_to_t2):.3f} {last_gap_to_t2.mean():.3f} {last_gap_to_t2.max():.3f}")

    out_path.write_text("\n".join(lines) + "\n")


def build(cxr_nodes: dict, ecg_nodes: dict, args):
    target_pairs, t1_pairs = [], []
    cxr_meta: dict[str, dict] = {}
    ecg_meta: dict[str, dict] = {}

    def add_cxr(node):
        cxr_meta.setdefault(
            node["dicom_id"],
            {"path": node["path"], "path_ok": node["path_ok"]},
        )

    def add_ecgs(eseq, idxs):
        for k in idxs:
            ecg_meta.setdefault(eseq[k]["ecg_id"], {"path": eseq[k]["path"]})

    for sid, cseq in cxr_nodes.items():
        eseq = ecg_nodes.get(sid)
        if not eseq:
            continue

        e_times = np.array([e["t_h"] for e in eseq], dtype=np.float64)

        # ---- Exp 3: original seq_target logic ----
        for j in range(len(cseq)):
            t2 = cseq[j]["t_h"]

            lo = int(np.searchsorted(e_times, t2 - args.lookback_hours, side="left"))
            hi = int(np.searchsorted(e_times, t2 - args.min_horizon_hours, side="right"))

            idxs = _cap_recent(list(range(lo, hi)), args.max_ecgs)
            if len(idxs) < args.min_ecgs:
                continue

            add_cxr(cseq[j])
            add_ecgs(eseq, idxs)

            ecg_times = [eseq[k]["t_h"] for k in idxs]
            target_pairs.append({
                "patient_id": int(sid),
                "t2_h": float(t2),
                "cxr_t2": cseq[j]["dicom_id"],
                "ecg_ids": [eseq[k]["ecg_id"] for k in idxs],
                "ecg_times_h": ecg_times,
                "delta_h": float(t2 - max(ecg_times)),
            })

        # ---- Exp 4: Mehak crit2 n=0 ----
        #
        # For each fixed t2:
        #   candidate t1 in [t2 - 24h, t2)
        #   ECGs in [max(t2 - 12h, t1), t2]
        #   choose only one t1 by strategy.
        for j in range(len(cseq)):
            t2 = cseq[j]["t_h"]

            # Find candidate t1s from previous frontal CXRs within 24h.
            candidate_idxs = []
            for i in range(j):
                t1 = cseq[i]["t_h"]
                dt = t2 - t1
                if 0 < dt <= args.t1_lookback_hours:
                    candidate_idxs.append(i)

            if not candidate_idxs:
                continue

            if args.t1_strategy == "nearest":
                # closest previous t1 to t2 = largest t1
                i_sel = max(candidate_idxs, key=lambda i: cseq[i]["t_h"])
            elif args.t1_strategy == "earliest":
                # earliest t1 inside [t2-24h, t2)
                i_sel = min(candidate_idxs, key=lambda i: cseq[i]["t_h"])
            else:
                raise ValueError(f"Unknown t1_strategy: {args.t1_strategy}")

            t1 = cseq[i_sel]["t_h"]
            dt = t2 - t1

            # crit2 n=0 ECG window
            ecg_start = max(t2 - args.ecg_lookback_hours, t1)
            ecg_end = t2

            lo = int(np.searchsorted(e_times, ecg_start, side="left"))
            hi = int(np.searchsorted(e_times, ecg_end, side="right"))

            idxs = _cap_recent(list(range(lo, hi)), args.max_ecgs)
            if len(idxs) < args.min_ecgs:
                continue

            add_cxr(cseq[i_sel])
            add_cxr(cseq[j])
            add_ecgs(eseq, idxs)

            t1_pairs.append({
                "patient_id": int(sid),
                "t1_h": float(t1),
                "t2_h": float(t2),
                "cxr_t1": cseq[i_sel]["dicom_id"],
                "cxr_t2": cseq[j]["dicom_id"],
                "ecg_ids": [eseq[k]["ecg_id"] for k in idxs],
                "ecg_times_h": [eseq[k]["t_h"] for k in idxs],
                "delta_h": float(dt),

                # Extra fields for checking. dataset.py should ignore these.
                "t1_strategy": args.t1_strategy,
                "crit": "crit2_n0",
                "ecg_start_h": float(ecg_start),
                "ecg_end_h": float(ecg_end),
            })

    return target_pairs, t1_pairs, cxr_meta, ecg_meta


def _report(name, pairs):
    if not pairs:
        print(f"  [{name}] 0 pairs")
        return

    pat = pd.Series([p["patient_id"] for p in pairs]).value_counts()
    necg = np.array([len(p["ecg_ids"]) for p in pairs])

    print(
        f"  [{name}] pairs={len(pairs):,}  patients={pat.size:,}  "
        f"pairs/pt med={int(pat.median())} max={pat.max()}  "
        f"ecgs/sample med={int(np.median(necg))} max={necg.max()}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--cxr_csv", default=C.CXR_CATALOG_CSV)
    ap.add_argument("--ecg_csv", default=C.ECG_CATALOG_CSV)

    # IMPORTANT:
    # Use frontal_cxr_metadata.csv here.
    ap.add_argument(
        "--metadata_path",
        default="./7.7/frontal_cxr_metadata.csv",
    )

    ap.add_argument("--cxr_root", default=C.CXR_ROOT)

    ap.add_argument(
        "--target_out",
        default="./7.7/seq_target_pairs_frontal.json",
    )
    ap.add_argument(
        "--t1_out",
        default="./7.7/exp4_crit2_n0_pairs.json",
    )
    ap.add_argument(
        "--stats_out",
        default="./7.7/exp4_crit2_n0_stats.txt",
    )

    # Exp3 original controls
    ap.add_argument("--min_horizon_hours", type=float, default=C.SEQ_MIN_HORIZON_HOURS)
    ap.add_argument("--lookback_hours", type=float, default=C.SEQ_LOOKBACK_HOURS)

    # Exp4 crit2 n=0 controls
    ap.add_argument("--t1_lookback_hours", type=float, default=24.0)
    ap.add_argument("--ecg_lookback_hours", type=float, default=12.0)
    ap.add_argument("--t1_strategy", choices=["nearest", "earliest"], required=True)

    ap.add_argument("--min_ecgs", type=int, default=C.MIN_ECGS_PER_INTERVAL)
    ap.add_argument("--max_ecgs", type=int, default=C.MAX_ECGS_PER_INTERVAL)

    ap.add_argument("--require_cxr_on_disk", action="store_true")
    ap.add_argument(
        "--skip_cxr_path_check",
        action="store_true",
        help="Do not os.stat every CXR during pair building; assume constructed paths exist.",
    )

    args = ap.parse_args()

    print("=== build_seq_pairs_crit2_n0 ===")
    print(f"  metadata_path: {args.metadata_path}")
    print(f"  t1_strategy: {args.t1_strategy}")
    print(f"  t1 window: [t2 - {args.t1_lookback_hours}h, t2)")
    print(f"  ECG window: [max(t2 - {args.ecg_lookback_hours}h, t1), t2]")

    cxr_nodes = load_cxr_nodes(
        args.cxr_csv,
        args.metadata_path,
        args.cxr_root,
        min_cxrs=1,
        check_paths=not args.skip_cxr_path_check,
    )
    ecg_nodes = load_ecg_nodes(args.ecg_csv)

    target_pairs, t1_pairs, cxr_meta, ecg_meta = build(cxr_nodes, ecg_nodes, args)

    if args.require_cxr_on_disk:
        target_pairs = [
            p for p in target_pairs
            if cxr_meta.get(p["cxr_t2"], {}).get("path_ok")
        ]
        t1_pairs = [
            p for p in t1_pairs
            if cxr_meta.get(p["cxr_t1"], {}).get("path_ok")
            and cxr_meta.get(p["cxr_t2"], {}).get("path_ok")
        ]

    _report("Exp3 seq_target frontal", target_pairs)
    _report(f"Exp4 crit2_n0 {args.t1_strategy}", t1_pairs)
    print(f"  Unique CXR={len(cxr_meta):,}  Unique ECG={len(ecg_meta):,}")

    Path(args.target_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.t1_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.target_out, "w") as f:
        json.dump({"pairs": target_pairs, "cxr_meta": cxr_meta, "ecg_meta": ecg_meta}, f)
    print(f"  Wrote {args.target_out}  ({len(target_pairs):,} pairs)")

    with open(args.t1_out, "w") as f:
        json.dump({"pairs": t1_pairs, "cxr_meta": cxr_meta, "ecg_meta": ecg_meta}, f)
    print(f"  Wrote {args.t1_out}  ({len(t1_pairs):,} pairs)")

    _summarize_exp4_pairs(t1_pairs, Path(args.stats_out))
    print(f"  Wrote {args.stats_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
