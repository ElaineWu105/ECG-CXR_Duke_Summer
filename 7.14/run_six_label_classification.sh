#!/bin/bash
#SBATCH -J ecg-six-label
#SBATCH -t 24:00:00
#SBATCH -A ACCOUNT_NAME
#SBATCH -p gpu-common
#SBATCH -q normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -G 1
#SBATCH -o ./7.14/logs/%x-%j.out
#SBATCH -e ./7.14/logs/%x-%j.err

# ECG embeddings -> six CXR CheXpert findings.
# Runs 3 ECG input strategies x 7 temporal offsets = 21 experiments:
#   single_mlp        : every ECG in the window is an individual example
#   sequence_attention: all ECGs in the window are pooled with learned attention
#   latest_mlp        : only the latest ECG in the window is used

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
PAIR_SOURCE="${ROOT}/7.13/pairs"
PAIRS="${ROOT}/7.14/classification_pairs"
ECG_CACHE="${EXP}/cache"
OUT="${ROOT}/7.14/six_label_classification"
LOGS="${ROOT}/7.14/logs"

mkdir -p "${OUT}" "${LOGS}" "${PAIRS}"
source "${EXP}/setup_env.sh"

# Keep classification inputs together under 7.14 without copying large JSONs.
# The source stores pair kinds in subdirectories, but the Python program expects
# a flat directory, so create lightweight links with the expected names.
for n in 0 2 4 6 8 10 12; do
  ln -sfn "${PAIR_SOURCE}/single/single_n${n}.json" "${PAIRS}/single_n${n}.json"
  ln -sfn "${PAIR_SOURCE}/seq/seq_n${n}.json" "${PAIRS}/seq_n${n}.json"
done

for required in \
  "${ECG_CACHE}/ecg_emb.npy" \
  "${ECG_CACHE}/ecg_ids.json" \
  "${PAIRS}/single_n0.json" \
  "${PAIRS}/seq_n0.json"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: missing required file: ${required}" >&2
    exit 1
  fi
done

python -u "${EXP}/ecg_cxr_label_probe_v2.py" \
  --n all \
  --models single_mlp sequence_attention latest_mlp \
  --pairs_dir "${PAIRS}" \
  --ecg_emb "${ECG_CACHE}/ecg_emb.npy" \
  --ecg_ids "${ECG_CACHE}/ecg_ids.json" \
  --output_dir "${OUT}" \
  --split_json "${OUT}/patient_split.json" \
  --seed "${SEED:-42}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --dropout "${DROPOUT:-0.2}" \
  --lr "${LR:-3e-4}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --epochs "${EPOCHS:-100}" \
  --patience "${PATIENCE:-15}" \
  --batch_size "${BATCH_SIZE:-512}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}"

echo "Six-label classification complete."
echo "Summary: ${OUT}/all_results_summary.csv"
