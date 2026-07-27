#!/usr/bin/env bash
#SBATCH --job-name=stmem_probe
#SBATCH --partition=commons
#SBATCH --output=logs/probe/st_mem_%j.out
#SBATCH --error=logs/probe/st_mem_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/probe

/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python scripts/run_probe_ridge.py \
  --probe-features-dir results/probe_features/st_mem_cu118_scavenge \
  --out-dir results/probe/st_mem_cu118_scavenge \
  --alpha 10.0
