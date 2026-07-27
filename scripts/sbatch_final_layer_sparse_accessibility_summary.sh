#!/bin/bash
#SBATCH -J fl_sparse_sum
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/final_layer_sparse_summary_%j.out
#SBATCH -e logs/final_layer_sparse_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_final_layer_feature_yield.py

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_final_layer_sparse_accessibility.py
