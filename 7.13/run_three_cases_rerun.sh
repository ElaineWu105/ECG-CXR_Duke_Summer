#!/bin/bash
#SBATCH -J cross-713-rerun
#SBATCH -t 72:00:00
#SBATCH -A ACCOUNT_NAME
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -G 1
#SBATCH -o ./7.13/logs/%x-%j.out
#SBATCH -e ./7.13/logs/%x-%j.err

# Re-run the 21 experiments from 7.13 with the current model code.
# Results are written to a new directory so the original 7.13 results remain intact.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
PAIRS="${ROOT}/7.13/pairs"
EMB="${EXP}/cache"
RERUN_TAG="${RERUN_TAG:-rerun_embdrop05}"
OUT="${ROOT}/7.13/cross_patient_huge_batch_${RERUN_TAG}"
WRAPPER="${ROOT}/7.13/run_case_experiment.py"

source "${EXP}/setup_env.sh"
mkdir -p "${OUT}" "${ROOT}/7.13/logs"

N_PATIENTS="${N_PATIENTS:-768}"
K_INTERVALS="${K_INTERVALS:-1}"
EPOCHS="${EPOCHS:-60}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-100}"
LR="${LR:-1e-4}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
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
  --early_stop_patience "${EARLY_STOP_PATIENCE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --seed "${SEED}"
)

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

  echo "===== Case ${case_no}, n=${n}, output=${root} ====="
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

echo "Re-running 7.13 experiments with current source code."
echo "Output directory: ${OUT}"
echo "Current staged_model.py includes embedding Dropout(p=0.5)."

for n in "${N_VALUES[@]}"; do
  run_one 1 "${n}" exp1a_single_ecg_cross \
    --single_pairs "${PAIRS}/single/single_n${n}.json" all_ecg \
    "Every ECG in the shifted 12-hour window is a separate ECG-CXR pair; cross-patient only."

  run_one 2 "${n}" exp3a_seq_ecg_meanpool \
    --seq_target_pairs "${PAIRS}/seq/seq_n${n}.json" sequence_ecg \
    "All ECGs in the shifted 12-hour window form one sequence-CXR pair; cross-patient only."

  run_one 3 "${n}" exp1a_single_ecg_cross \
    --single_pairs "${PAIRS}/nearest/nearest_n${n}.json" nearest_ecg \
    "Only the ECG closest to t2 in the shifted 12-hour window is paired with CXR; cross-patient only."
done

echo "7.13 re-run complete: ${OUT}"
