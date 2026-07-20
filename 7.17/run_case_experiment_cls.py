#!/usr/bin/env python3
"""Run one 7.17 experiment, using CLS pooling for sequence ECG models.

The model implementation is reused from 7.15 so regularization remains
identical to the mean-pooling run.  Only the sequence experiment's pooling
mode is changed from ``mean`` to ``cls``.
"""
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
MODEL_715 = ROOT / "7.15"

# Import staged_model.py from 7.15 and the remaining training package from the
# main experiment directory.
sys.path.insert(0, str(MODEL_715))
sys.path.insert(1, str(EXP))

import run_experiments as runner


original_resolve = runner.resolve_specs


def resolve_with_cls_pooling(selected):
    specs = original_resolve(selected)
    if len(specs) != 1:
        raise ValueError("7.17 wrapper requires exactly one selected experiment")

    spec = specs[0]
    spec.name = os.environ["CASE_EXPERIMENT_NAME"]
    spec.description = os.environ["CASE_DESCRIPTION"]
    spec.target_window = os.environ["CASE_TARGET_WINDOW"]
    spec.temporal_min_horizon_hours = None
    spec.temporal_max_horizon_hours = None

    if spec.ecg_mode == "sequence":
        spec.ecg_pool = "cls"
        spec.use_future_query = False

    expected_pool = os.environ.get("EXPECTED_ECG_POOL")
    actual_pool = spec.ecg_pool if spec.ecg_mode == "sequence" else "none"
    if expected_pool and actual_pool != expected_pool:
        raise ValueError(
            f"Expected effective ECG pool={expected_pool!r}, got {actual_pool!r}"
        )
    return specs


runner.resolve_specs = resolve_with_cls_pooling
runner.main()
