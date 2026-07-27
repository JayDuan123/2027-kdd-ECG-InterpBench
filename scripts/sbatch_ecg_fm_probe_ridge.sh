#!/usr/bin/env bash
#SBATCH --job-name=ecgfm_probe
#SBATCH --partition=commons
#SBATCH --output=logs/probe/ecg_fm_%j.out
#SBATCH --error=logs/probe/ecg_fm_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/probe

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/run_probe_ridge.py \
  --probe-features-dir results/probe_features/ecg_fm_cu118_commons \
  --out-dir results/probe/ecg_fm_cu118_commons \
  --alpha 10.0
