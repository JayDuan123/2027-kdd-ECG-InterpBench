#!/usr/bin/env bash
#SBATCH --job-name=csfm_erase_cd
#SBATCH --partition=commons
#SBATCH --output=logs/erase/csfm_qrs_cd_%j.out
#SBATCH --error=logs/erase/csfm_qrs_cd_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/erase

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_csfm_continuation_erase.py \
  --activation-index-dir results/activation_index/csfm_cu118_commons \
  --probe-features-dir results/probe_features/csfm_cu118_commons \
  --concept-id qrs_duration \
  --task-id ptbxl_cd \
  --layer 5 \
  --out-dir results/analysis/csfm_cu118_commons \
  --device cpu
