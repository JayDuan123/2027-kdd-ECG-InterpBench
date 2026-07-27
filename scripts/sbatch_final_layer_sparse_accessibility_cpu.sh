#!/bin/bash
#SBATCH -J fl_sparse_cpu
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-11%12
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/final_layer_sparse_cpu_%A_%a.out
#SBATCH -e logs/final_layer_sparse_cpu_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if (( SLURM_ARRAY_TASK_ID < 6 )); then
  kind=dense
  index="${SLURM_ARRAY_TASK_ID}"
else
  kind=pca
  index="$((SLURM_ARRAY_TASK_ID - 6))"
fi

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_final_layer_sparse_accessibility_worker.py \
  --source-kind "${kind}" \
  --source-index "${index}" \
  --device cpu
