#!/usr/bin/env bash
#SBATCH --job-name=ecgfm_feat
#SBATCH --partition=commons
#SBATCH --output=logs/probe_features/ecg_fm_%j.out
#SBATCH --error=logs/probe_features/ecg_fm_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/probe_features

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/build_probe_features.py \
  --index-dir results/activation_index/ecg_fm_cu118_commons \
  --out-dir results/probe_features/ecg_fm_cu118_commons \
  --layers all
