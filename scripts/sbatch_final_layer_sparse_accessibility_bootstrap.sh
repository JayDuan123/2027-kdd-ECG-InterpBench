#!/bin/bash
#SBATCH -J fl_sparse_boot
#SBATCH -p commons
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/final_layer_sparse_bootstrap_%A_%a.out
#SBATCH -e logs/final_layer_sparse_bootstrap_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/bootstrap_final_layer_sparse_accessibility.py \
  --model-index "${SLURM_ARRAY_TASK_ID}" \
  --bootstrap-draws 2000
