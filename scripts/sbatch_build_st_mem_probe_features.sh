#!/usr/bin/env bash
#SBATCH --job-name=stmem_feat
#SBATCH --partition=commons
#SBATCH --output=logs/probe_features/st_mem_%j.out
#SBATCH --error=logs/probe_features/st_mem_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/probe_features

/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python scripts/build_probe_features.py \
  --index-dir results/activation_index/st_mem_cu118_scavenge \
  --out-dir results/probe_features/st_mem_cu118_scavenge \
  --layers all
