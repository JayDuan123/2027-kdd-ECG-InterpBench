#!/bin/bash
#SBATCH -J five_scale_dense
#SBATCH -p commons
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/five_scale_dense_comparison_%j.out
#SBATCH -e logs/five_scale_dense_comparison_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_five_scale_dense_comparison.py \
  --bootstrap-replicates 20000
