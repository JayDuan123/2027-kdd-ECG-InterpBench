#!/usr/bin/env bash
#SBATCH --job-name=stmem_ers2
#SBATCH --partition=commons
#SBATCH --output=logs/erase/st_mem_cont_%j.out
#SBATCH --error=logs/erase/st_mem_cont_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/erase

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_transformer_continuation_erase.py \
  --model st_mem \
  --activation-index-dir results/activation_index/st_mem_cu118_scavenge \
  --probe-features-dir results/probe_features/st_mem_cu118_scavenge \
  --concept-id hr_ventricular \
  --task-id ptbxl_sttc \
  --layer 5 \
  --out-dir results/analysis/st_mem_cu118_scavenge \
  --device cpu
