#!/bin/bash
#SBATCH -J cls-diag-n2
#SBATCH -t 48:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o 7.17/logs/%x-%j.out
#SBATCH -e 7.17/logs/%x-%j.err

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
EMB="${EXP}/cache"
OUT="${ROOT}/7.17/cls_diagnostics_n2"
[[ -f "${EXP}/setup_env.sh" ]] && source "${EXP}/setup_env.sh"
mkdir -p "${OUT}" "${ROOT}/7.17/logs"

python3 -u "${ROOT}/7.17/train_cls_diagnostics.py" \
  --only exp3a_seq_ecg_meanpool \
  --seq_target_pairs "${ROOT}/7.13/pairs/seq/seq_n2.json" \
  --cxr_emb "${EMB}/cxr_emb.npy" --cxr_ids "${EMB}/cxr_ids.json" \
  --ecg_emb "${EMB}/ecg_emb.npy" --ecg_ids "${EMB}/ecg_ids.json" \
  --loss_mode_override cross --output_dir "${OUT}" \
  --n_patients "${N_PATIENTS:-768}" --k_intervals "${K_INTERVALS:-1}" \
  --epochs "${EPOCHS:-60}" --steps_per_epoch "${STEPS_PER_EPOCH:-100}" \
  --lr "${LR:-1e-4}" --weight_decay "${WEIGHT_DECAY:-3e-3}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-5}" \
  --eval_batch_size "${EVAL_BATCH_SIZE:-512}" --seed "${SEED:-42}" \
  --sample_unique_targets
