#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export LAMBDA_MULTIVIEW=0.02
export LAMBDA_PROTOTYPE=0.1
bash "$ROOT/7.20/run_three_cases_multiview_prototype.sh"
bash "$ROOT/7.20/run_best_prototype_ablation.sh"
