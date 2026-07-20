#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="$ROOT/Waveform_CXR_EHR/ECGCXRPatientTemporal"
EMB="$EXP/cache"
PAIRS="$ROOT/7.20/pairs/seq_pooled_0_24h_multiview.json"
OUT="$ROOT/7.20/latest_gated_history_multiview"
[[ -f "$EXP/setup_env.sh" ]] && source "$EXP/setup_env.sh"
mkdir -p "$OUT" "$ROOT/7.20/logs"
[[ -f "$PAIRS" ]] || python -u "$ROOT/7.20/build_pooled_24h_multiview_pairs.py"
PYTHONUNBUFFERED=1 python -u "$ROOT/7.20/train_latest_gated_history_multiview.py" \
  --only exp3a_seq_ecg_meanpool --seq_target_pairs "$PAIRS" \
  --cxr_emb "$EMB/cxr_emb.npy" --cxr_ids "$EMB/cxr_ids.json" \
  --ecg_emb "$EMB/ecg_emb.npy" --ecg_ids "$EMB/ecg_ids.json" \
  --loss_mode_override cross --output_dir "$OUT" \
  --n_patients 768 --k_intervals 1 --epochs 60 --steps_per_epoch 100 \
  --lr 1e-4 --weight_decay 3e-3 --early_stop_patience 10 \
  --eval_batch_size 512 --seed 42 --sample_unique_targets "$@"
