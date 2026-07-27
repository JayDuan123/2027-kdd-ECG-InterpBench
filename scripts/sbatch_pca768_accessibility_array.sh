#!/bin/bash
#SBATCH -J pca_access
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-29%10
#SBATCH -o logs/pca768_accessibility_%A_%a.out
#SBATCH -e logs/pca768_accessibility_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_pca768_accessibility_worker.py \
  --group-index "${SLURM_ARRAY_TASK_ID}"
