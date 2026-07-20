#!/bin/bash
#SBATCH -J cross-717-cls
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

# Case 2: ECG sequence -> Transformer self-attention -> trained CLS output -> future CXR.
# Case 3: nearest single ECG -> future CXR (no sequence, hence no CLS token).
# Settings otherwise match the 7.15 anti-overfitting run.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
PAIRS="${ROOT}/7.13/pairs"
EMB="${EXP}/cache"
TRAIN_CLS_ONLY="${TRAIN_CLS_ONLY:-0}"
if [[ "${TRAIN_CLS_ONLY}" == 1 ]]; then
  OUT="${ROOT}/7.17/cross_patient_cls_only_case2_case3"
else
  OUT="${ROOT}/7.17/cross_patient_cls_pool_case2_case3"
fi
WRAPPER="${ROOT}/7.17/run_case_experiment_cls.py"

[[ -f "${EXP}/setup_env.sh" ]] && source "${EXP}/setup_env.sh"
mkdir -p "${OUT}" "${ROOT}/7.17/logs"

N_PATIENTS="${N_PATIENTS:-768}"
K_INTERVALS="${K_INTERVALS:-1}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-3}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAMPLE_UNIQUE_TARGETS="${SAMPLE_UNIQUE_TARGETS:-1}"
N_VALUES=(0 2 4 6)

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }
}

for file in cxr_emb.npy cxr_ids.json ecg_emb.npy ecg_ids.json; do
  require_file "${EMB}/${file}"
done
for n in "${N_VALUES[@]}"; do
  require_file "${PAIRS}/seq/seq_n${n}.json"
  require_file "${PAIRS}/nearest/nearest_n${n}.json"
done

COMMON=(
  --cxr_emb "${EMB}/cxr_emb.npy"
  --cxr_ids "${EMB}/cxr_ids.json"
  --ecg_emb "${EMB}/ecg_emb.npy"
  --ecg_ids "${EMB}/ecg_ids.json"
  --n_patients "${N_PATIENTS}"
  --k_intervals "${K_INTERVALS}"
  --epochs "${EPOCHS}"
  --steps_per_epoch "${STEPS_PER_EPOCH}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --early_stop_patience "${EARLY_STOP_PATIENCE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --seed "${SEED}"
)

if [[ "${SAMPLE_UNIQUE_TARGETS}" == 1 ]]; then
  COMMON+=(--sample_unique_targets)
fi

run_one() {
  local case_no="$1"
  local n="$2"
  local architecture="$3"
  local pair_arg="$4"
  local pair_file="$5"
  local label="$6"
  local expected_pool="$7"
  local description="$8"
  local root="${OUT}/case${case_no}_${label}_n${n}"
  local name="case${case_no}_${label}_cross_n${n}"
  local result="${root}/${name}/results.json"
  local extra_args=()
  if [[ "${TRAIN_CLS_ONLY}" == 1 && "${case_no}" == 2 ]]; then
    extra_args+=(--train_cls_only)
  fi

  if [[ "${SKIP_EXISTING}" == 1 && -f "${result}" ]]; then
    echo "[SKIP] ${result}"
    return
  fi

  echo "===== 7.17 Case ${case_no}, n=${n}, pool=${expected_pool} ====="
  CASE_EXPERIMENT_NAME="${name}" \
  CASE_TARGET_WINDOW="[t2-${n}-12h, t2-${n}]" \
  CASE_DESCRIPTION="${description}" \
  EXPECTED_ECG_POOL="${expected_pool}" \
  python3 -u "${WRAPPER}" \
    --only "${architecture}" \
    "${pair_arg}" "${pair_file}" \
    --loss_mode_override cross \
    --output_dir "${root}" \
    "${extra_args[@]}" \
    "${COMMON[@]}"
}

echo "Running 7.17 Case 2 CLS attention-pooling and Case 3 comparator experiments."
echo "Output directory: ${OUT}"
echo "n values: ${N_VALUES[*]}"
echo "Sequence regularization inherited from 7.15: token_drop=0.20, pool_drop=0.30, emb_drop=0.50"
echo "Training: lr=${LR}, weight_decay=${WEIGHT_DECAY}, patience=${EARLY_STOP_PATIENCE}, unique_targets=${SAMPLE_UNIQUE_TARGETS}"
echo "CLS-only training for Case 2: ${TRAIN_CLS_ONLY}"

for n in "${N_VALUES[@]}"; do
  run_one 2 "${n}" exp3a_seq_ecg_meanpool \
    --seq_target_pairs "${PAIRS}/seq/seq_n${n}.json" sequence_ecg_cls cls \
    "All ECGs in the shifted 12-hour window form one sequence-CXR pair; trainable CLS token pools through Transformer self-attention; no mean pooling; cross-patient loss."

  run_one 3 "${n}" exp1a_single_ecg_cross \
    --single_pairs "${PAIRS}/nearest/nearest_n${n}.json" nearest_ecg none \
    "Only the ECG closest to t2 is paired with CXR; single-ECG comparator without CLS pooling."
done

echo "7.17 CLS-pooling experiment complete: ${OUT}"
