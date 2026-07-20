#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="$ROOT/Waveform_CXR_EHR/ECGCXRPatientTemporal"
EMB="$EXP/cache"
PAIRS="$ROOT/7.13/pairs/multiview"
OUT="$ROOT/7.20/three_cases_multiview_prototype"
LOGS="$ROOT/7.20/logs/three_cases_multiview_prototype"
TRAIN="$ROOT/7.20/train_primary_multiview_prototype.py"

[[ -f "$EXP/setup_env.sh" ]] && source "$EXP/setup_env.sh"
mkdir -p "$OUT" "$LOGS"

N_VALUES=(0 2 4 6 8 10 12)
N_PATIENTS="${N_PATIENTS:-768}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
LAMBDA_MULTIVIEW="${LAMBDA_MULTIVIEW:-0.02}"
LAMBDA_PROTOTYPE="${LAMBDA_PROTOTYPE:-0.1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

COMMON=(
  --cxr_emb "$EMB/cxr_emb.npy" --cxr_ids "$EMB/cxr_ids.json"
  --ecg_emb "$EMB/ecg_emb.npy" --ecg_ids "$EMB/ecg_ids.json"
  --loss_mode_override cross --n_patients "$N_PATIENTS" --k_intervals 1
  --epochs "$EPOCHS" --steps_per_epoch "$STEPS_PER_EPOCH"
  --lr 1e-4 --weight_decay 3e-3 --early_stop_patience "$EARLY_STOP_PATIENCE"
  --eval_batch_size 512 --seed 42 --sample_unique_targets
)

run_one() {
  local case_no="$1" n="$2" architecture="$3" pair_arg="$4" kind="$5" label="$6"
  local pair_file="$PAIRS/multiview_${kind}_n${n}.json"
  local name="case${case_no}_${label}_mv${LAMBDA_MULTIVIEW}_proto${LAMBDA_PROTOTYPE}_n${n}"
  local case_out="$OUT/case${case_no}_${label}_n${n}"
  local result="$case_out/$name/results.json"
  local log="$LOGS/$name.log"
  if [[ "$SKIP_EXISTING" == 1 && -f "$result" ]]; then
    echo "[SKIP] $result"
    return
  fi
  echo "===== START $name $(date --iso-8601=seconds) ====="
  LAMBDA_MULTIVIEW="$LAMBDA_MULTIVIEW" LAMBDA_PROTOTYPE="$LAMBDA_PROTOTYPE" \
  PAIR_JSON="$pair_file" EXPERIMENT_NAME="$name" \
  TARGET_WINDOW="[t2-${n}-12h,t2-${n}]" \
  EXPERIMENT_DESCRIPTION="Case $case_no $label; primary + ${LAMBDA_MULTIVIEW} multiview + ${LAMBDA_PROTOTYPE} prototype" \
  python -u "$TRAIN" --only "$architecture" "$pair_arg" "$pair_file" \
    --output_dir "$case_out" "${COMMON[@]}" 2>&1 | tee "$log"
  echo "===== END $name $(date --iso-8601=seconds) ====="
}

for n in "${N_VALUES[@]}"; do
  run_one 1 "$n" exp1a_single_ecg_cross --single_pairs single all_ecg
  run_one 2 "$n" exp3a_seq_ecg_meanpool --seq_target_pairs seq sequence_ecg
  run_one 3 "$n" exp1a_single_ecg_cross --single_pairs nearest nearest_ecg
done

python "$ROOT/7.20/summarize_three_cases_objective.py" --results-root "$OUT" \
  --output "$OUT/summary.csv"
echo "All 21 runs complete. Summary: $OUT/summary.csv"
