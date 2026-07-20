#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE="$ROOT/Waveform_CXR_EHR/ECGCXRPatientTemporal"
OUT="$ROOT/7.19/label_multipositive_top10_lambda01_n2"
[[ -f "$EXP/setup_env.sh" ]] && source "$EXP/setup_env.sh"

export MULTIPOS_NAME="single_n2_label_multipos_top10_lambda01"
export TARGET_WINDOW="[t2-14h, t2-2h]"

PYTHONUNBUFFERED=1 python -u "$ROOT/7.19/train_label_multipositive.py" \
  --only exp1a_single_ecg_cross \
  --single_pairs "$ROOT/7.13/pairs/single/single_n2.json" \
  --cxr_emb "$BASE/cache/cxr_emb.npy" \
  --cxr_ids "$BASE/cache/cxr_ids.json" \
  --ecg_emb "$BASE/cache/ecg_emb.npy" \
  --ecg_ids "$BASE/cache/ecg_ids.json" \
  --output_dir "$OUT" \
  --lambda_label 0.1 \
  --label_top_k 10 \
  --n_patients 768 \
  --k_intervals 1 \
  --steps_per_epoch 100 \
  --epochs 60 \
  --eval_batch_size 512 \
  "$@"
