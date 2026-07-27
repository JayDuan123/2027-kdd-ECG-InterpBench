#!/usr/bin/env bash
#SBATCH --job-name=jepa_ers2
#SBATCH --partition=commons
#SBATCH --output=logs/erase/ecg_jepa_cont_%j.out
#SBATCH --error=logs/erase/ecg_jepa_cont_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/erase

/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/bin/python scripts/run_transformer_continuation_erase.py \
  --model ecg_jepa \
  --activation-index-dir results/activation_index/ecg_jepa_cu118_commons \
  --probe-features-dir results/probe_features/ecg_jepa_cu118_commons \
  --concept-id r_amp_precordial \
  --task-id ptbxl_hyp \
  --layer 7 \
  --out-dir results/analysis/ecg_jepa_cu118_commons \
  --device cpu
