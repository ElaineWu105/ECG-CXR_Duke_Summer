#!/bin/bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
EMB="${EXP}/cache"
OUT="${ROOT}/7.18/sequence_change_cls_n024"
[[ -f "$EXP/setup_env.sh" ]] && source "$EXP/setup_env.sh"
mkdir -p "${OUT}" "${ROOT}/7.18/logs"

for MODEL_MODE in ${MODEL_MODES:-content delta}; do
  export MODEL_MODE
  for N_VALUE in 0 2 4; do
    export N_VALUE
    NAME="case2_no_time_${MODEL_MODE}_cls_n${N_VALUE}"
    [[ "${SKIP_EXISTING:-1}" == 1 && -f "${OUT}/${NAME}/results.json" ]] && continue
    echo "===== no-time ${MODEL_MODE} CLS n=${N_VALUE} ====="
    python3 -u "${ROOT}/7.18/train_sequence_change_cls.py" \
      --only exp3a_seq_ecg_meanpool \
      --seq_target_pairs "${ROOT}/7.13/pairs/seq/seq_n${N_VALUE}.json" \
      --cxr_emb "${EMB}/cxr_emb.npy" --cxr_ids "${EMB}/cxr_ids.json" \
      --ecg_emb "${EMB}/ecg_emb.npy" --ecg_ids "${EMB}/ecg_ids.json" \
      --loss_mode_override cross --output_dir "${OUT}" \
      --n_patients "${N_PATIENTS:-768}" --k_intervals 1 \
      --epochs "${EPOCHS:-60}" --steps_per_epoch "${STEPS_PER_EPOCH:-100}" \
      --lr "${LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-3e-3}" \
      --early_stop_patience "${EARLY_STOP_PATIENCE:-5}" \
      --eval_batch_size "${EVAL_BATCH_SIZE:-512}" --seed "${SEED:-42}" \
      --sample_unique_targets
  done
done
