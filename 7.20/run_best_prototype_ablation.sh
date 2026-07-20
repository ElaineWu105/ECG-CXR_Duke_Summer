#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="$ROOT/Waveform_CXR_EHR/ECGCXRPatientTemporal"
EMB="$EXP/cache"
PAIRS="$ROOT/7.13/pairs/multiview"
GRID="$ROOT/7.20/three_cases_multiview_prototype"
SUMMARY="$GRID/summary.csv"
OUT="$ROOT/7.20/best_case_ablation_no_prototype"
LOGS="$ROOT/7.20/logs/best_case_ablation_no_prototype"

[[ -f "$EXP/setup_env.sh" ]] && source "$EXP/setup_env.sh"
mkdir -p "$OUT" "$LOGS"
[[ -f "$SUMMARY" ]] || { echo "Missing completed grid summary: $SUMMARY" >&2; exit 1; }

read -r case_no n < <(awk -F, 'NR>1 && $5+0>best {best=$5+0; c=$1; n=$2} END {print c, n}' "$SUMMARY")
case "$case_no" in
  1) architecture=exp1a_single_ecg_cross; pair_arg=--single_pairs; kind=single; label=all_ecg ;;
  2) architecture=exp3a_seq_ecg_meanpool; pair_arg=--seq_target_pairs; kind=seq; label=sequence_ecg ;;
  3) architecture=exp1a_single_ecg_cross; pair_arg=--single_pairs; kind=nearest; label=nearest_ecg ;;
  *) echo "Invalid best case: $case_no" >&2; exit 1 ;;
esac

pair_file="$PAIRS/multiview_${kind}_n${n}.json"
name="ablation_no_prototype_case${case_no}_${label}_n${n}"
case_out="$OUT/case${case_no}_${label}_n${n}"
log="$LOGS/$name.log"

echo "Best validation setting: case=$case_no n=$n"
echo "Ablation: lambda_multiview=0.02, lambda_prototype=0"
LAMBDA_MULTIVIEW=0.02 LAMBDA_PROTOTYPE=0 PAIR_JSON="$pair_file" \
EXPERIMENT_NAME="$name" TARGET_WINDOW="[t2-${n}-12h,t2-${n}]" \
EXPERIMENT_DESCRIPTION="Best-grid ablation: remove disease prototype" \
python -u "$ROOT/7.20/train_primary_multiview_prototype.py" \
  --only "$architecture" "$pair_arg" "$pair_file" \
  --cxr_emb "$EMB/cxr_emb.npy" --cxr_ids "$EMB/cxr_ids.json" \
  --ecg_emb "$EMB/ecg_emb.npy" --ecg_ids "$EMB/ecg_ids.json" \
  --loss_mode_override cross --output_dir "$case_out" \
  --n_patients 768 --k_intervals 1 --epochs 60 --steps_per_epoch 100 \
  --lr 1e-4 --weight_decay 3e-3 --early_stop_patience 10 \
  --eval_batch_size 512 --seed 42 --sample_unique_targets 2>&1 | tee "$log"

echo "Ablation complete: $case_out/$name/results.json"
