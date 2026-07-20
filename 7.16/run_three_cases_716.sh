#!/bin/bash
#SBATCH -J cross-716-mildreg
#SBATCH -t 72:00:00
#SBATCH -A kamaleswaranlab
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o 7.16/logs/%x-%j.out
#SBATCH -e 7.16/logs/%x-%j.err

# 7.16 milder anti-overfitting rerun of the 7.13 three-case grid.
# Uses 7.16/staged_model.py via 7.15/run_case_experiment.py.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
PAIRS="${ROOT}/7.13/pairs"
EMB="${EXP}/cache"
OUT="${ROOT}/7.16/cross_patient_huge_batch_mild_antioverfit"
WRAPPER="${ROOT}/7.16/run_case_experiment.py"

[[ -f "${EXP}/setup_env.sh" ]] && source "${EXP}/setup_env.sh"
mkdir -p "${OUT}" "${ROOT}/7.16/logs"

N_PATIENTS="${N_PATIENTS:-768}"
K_INTERVALS="${K_INTERVALS:-1}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-7}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SAMPLE_UNIQUE_TARGETS="${SAMPLE_UNIQUE_TARGETS:-1}"
N_VALUES=(0 2 4 6 8 10 12)

require_file() {
  [[ -f "$1" ]] || { echo "ERROR: missing $1" >&2; exit 1; }
}

for file in cxr_emb.npy cxr_ids.json ecg_emb.npy ecg_ids.json; do
  require_file "${EMB}/${file}"
done
for n in "${N_VALUES[@]}"; do
  for kind in single seq nearest; do
    require_file "${PAIRS}/${kind}/${kind}_n${n}.json"
  done
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
  local description="$7"
  local root="${OUT}/case${case_no}_${label}_n${n}"
  local name="case${case_no}_${label}_cross_n${n}"
  local result="${root}/${name}/results.json"

  if [[ "${SKIP_EXISTING}" == 1 && -f "${result}" ]]; then
    echo "[SKIP] ${result}"
    return
  fi

  echo "===== 7.16 Case ${case_no}, n=${n}, output=${root} ====="
  CASE_EXPERIMENT_NAME="${name}" \
  CASE_TARGET_WINDOW="[t2-${n}-12h, t2-${n}]" \
  CASE_DESCRIPTION="${description}" \
  python3 -u "${WRAPPER}" \
    --only "${architecture}" \
    "${pair_arg}" "${pair_file}" \
    --loss_mode_override cross \
    --output_dir "${root}" \
    "${COMMON[@]}"
}

echo "Running 7.16 milder anti-overfitting experiments."
echo "Output directory: ${OUT}"
echo "Model override: ${ROOT}/7.16/staged_model.py"
echo "Regularization: sequence_token_drop=0.10, seq_pool_dropout=0.15, emb_drop=0.50"
echo "Training knobs: lr=${LR}, weight_decay=${WEIGHT_DECAY}, patience=${EARLY_STOP_PATIENCE}, sample_unique_targets=${SAMPLE_UNIQUE_TARGETS}"

for n in "${N_VALUES[@]}"; do
  run_one 1 "${n}" exp1a_single_ecg_cross \
    --single_pairs "${PAIRS}/single/single_n${n}.json" all_ecg \
    "Every ECG in the shifted 12-hour window is a separate ECG-CXR pair; cross-patient only."

  run_one 2 "${n}" exp3a_seq_ecg_meanpool \
    --seq_target_pairs "${PAIRS}/seq/seq_n${n}.json" sequence_ecg \
    "All ECGs in the shifted 12-hour window form one sequence-CXR pair; cross-patient only; 7.16 milder sequence regularization enabled."

  run_one 3 "${n}" exp1a_single_ecg_cross \
    --single_pairs "${PAIRS}/nearest/nearest_n${n}.json" nearest_ecg \
    "Only the ECG closest to t2 in the shifted 12-hour window is paired with CXR; cross-patient only."
done

echo "7.16 milder anti-overfitting run complete: ${OUT}"
