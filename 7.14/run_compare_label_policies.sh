#!/bin/bash
#SBATCH -J ecg-label-policy
#SBATCH -t 48:00:00
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

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="${ROOT}/Waveform_CXR_EHR/ECGCXRPatientTemporal"
PAIRS="${ROOT}/7.14/classification_pairs"
ECG_CACHE="${EXP}/cache"
OUT_ROOT="${ROOT}/7.14/label_policy_comparison"
SPLIT="${OUT_ROOT}/patient_split.json"

mkdir -p "${OUT_ROOT}" "${ROOT}/7.14/logs"
source "${EXP}/setup_env.sh"

for policy in explicit_01 all_nonpositive_negative; do
  python -u "${EXP}/ecg_cxr_label_probe_v2.py" \
    --n all \
    --models single_mlp sequence_attention latest_mlp \
    --pairs_dir "${PAIRS}" \
    --ecg_emb "${ECG_CACHE}/ecg_emb.npy" \
    --ecg_ids "${ECG_CACHE}/ecg_ids.json" \
    --output_dir "${OUT_ROOT}/${policy}" \
    --split_json "${SPLIT}" \
    --label_policy "${policy}" \
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
done

python "${ROOT}/7.14/compare_label_policy_results.py"
