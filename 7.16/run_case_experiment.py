#!/usr/bin/env python3
"""Run one existing architecture with accurate 7.13 experiment metadata."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = Path(__file__).resolve().parents[1] / "Waveform_CXR_EHR" / "ECGCXRPatientTemporal"
# Put 7.16 first so engine imports this folder's staged_model.py, while all
# other modules still come from the main experiment package.
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(EXP))
import run_experiments as runner

original_resolve = runner.resolve_specs

def resolve_with_metadata(selected):
    specs = original_resolve(selected)
    if len(specs) != 1:
        raise ValueError("7.13 wrapper requires exactly one selected experiment")
    spec = specs[0]
    spec.name = os.environ["CASE_EXPERIMENT_NAME"]
    spec.description = os.environ["CASE_DESCRIPTION"]
    spec.target_window = os.environ["CASE_TARGET_WINDOW"]
    spec.temporal_min_horizon_hours = None
    spec.temporal_max_horizon_hours = None
    return specs

runner.resolve_specs = resolve_with_metadata
runner.main()
