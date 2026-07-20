#!/usr/bin/env python3
"""7.18-local wrapper selecting CLS pooling for one sequence experiment."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
sys.path.insert(0, str(ROOT / "7.15"))
sys.path.insert(1, str(EXP))
import run_experiments as runner

original_resolve = runner.resolve_specs


def resolve_cls(selected):
    specs = original_resolve(selected)
    if len(specs) != 1:
        raise ValueError("7.18 wrapper requires exactly one experiment")
    spec = specs[0]
    spec.name = os.environ["CASE_EXPERIMENT_NAME"]
    spec.description = os.environ["CASE_DESCRIPTION"]
    spec.target_window = os.environ["CASE_TARGET_WINDOW"]
    spec.temporal_min_horizon_hours = None
    spec.temporal_max_horizon_hours = None
    if spec.ecg_mode == "sequence":
        spec.ecg_pool = "cls"
        spec.use_future_query = False
    expected = os.environ.get("EXPECTED_ECG_POOL")
    actual = spec.ecg_pool if spec.ecg_mode == "sequence" else "none"
    if expected and actual != expected:
        raise ValueError(f"Expected pool={expected!r}, got {actual!r}")
    return specs


runner.resolve_specs = resolve_cls
runner.main()
