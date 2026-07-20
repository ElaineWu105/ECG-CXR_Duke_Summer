#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
PAIRS="${ROOT}/7.13/pairs/seq"
EMB="${EXP}/cache"
OUT="${ROOT}/7.18/temporal_supervision_cls_n024"
WRAPPER="${ROOT}/7.18/run_case_experiment_cls.py"

[[ -f "$EXP/setup_env.sh" ]] && source "$EXP/setup_env.sh"
mkdir -p "${OUT}" "${ROOT}/7.18/logs"

N_PATIENTS="${N_PATIENTS:-384}"
K_INTERVALS="${K_INTERVALS:-2}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-3}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
SEED="${SEED:-42}"
LAMBDA_TEMPORAL="${LAMBDA_TEMPORAL:-0.2}"

for N_VALUE in 0 2 4; do
  NAME="case2_cls_combined_n${N_VALUE}"
  RUN_ROOT="${OUT}/${NAME}"
  RESULT="${RUN_ROOT}/${NAME}/results.json"
  if [[ "${SKIP_EXISTING:-1}" == 1 && -f "${RESULT}" ]]; then
    echo "[SKIP] ${RESULT}"
    continue
  fi

  echo "===== 7.18 CLS combined temporal supervision n=${N_VALUE} ====="
  CASE_EXPERIMENT_NAME="${NAME}" \
  CASE_TARGET_WINDOW="[t2-$((N_VALUE + 12))h, t2-${N_VALUE}h]" \
  CASE_DESCRIPTION="CLS ECG sequence with cross-patient and within-patient temporal contrastive supervision." \
  EXPECTED_ECG_POOL="cls" \
  python3 -u "${WRAPPER}" \
    --only exp3a_seq_ecg_meanpool \
    --seq_target_pairs "${PAIRS}/seq_n${N_VALUE}.json" \
    --cxr_emb "${EMB}/cxr_emb.npy" --cxr_ids "${EMB}/cxr_ids.json" \
    --ecg_emb "${EMB}/ecg_emb.npy" --ecg_ids "${EMB}/ecg_ids.json" \
    --loss_mode_override combined \
    --lambda_temporal_override "${LAMBDA_TEMPORAL}" \
    --output_dir "${RUN_ROOT}" \
    --n_patients "${N_PATIENTS}" --k_intervals "${K_INTERVALS}" \
    --min_train_targets_per_patient 2 --sample_unique_targets \
    --epochs "${EPOCHS}" --steps_per_epoch "${STEPS_PER_EPOCH}" \
    --lr "${LR}" --weight_decay "${WEIGHT_DECAY}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" --seed "${SEED}"
done
