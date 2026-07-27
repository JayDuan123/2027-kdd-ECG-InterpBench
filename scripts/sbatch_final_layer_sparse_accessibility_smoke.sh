#!/bin/bash
#SBATCH -J fl_sparse_smoke
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH --array=0-3%4
#SBATCH -o logs/final_layer_sparse_smoke_%A_%a.out
#SBATCH -e logs/final_layer_sparse_smoke_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) kind=dense; index=0; device=cpu ;;
  1) kind=pca; index=0; device=cpu ;;
  2) kind=sae; index=0; device=cuda ;;
  3) kind=random; index=0; device=cuda ;;
esac

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_final_layer_sparse_accessibility_worker.py \
  --source-kind "${kind}" \
  --source-index "${index}" \
  --device "${device}" \
  --max-records-per-split 256 \
  --semantic-train-limit 256 \
  --budget-replicates 2 \
  --random-replicates 2 \
  --output-root results/final_layer_sparse_accessibility_e8_v2_smoke/workers
