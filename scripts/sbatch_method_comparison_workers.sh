#!/usr/bin/env bash
#SBATCH --job-name=method_cmp
#SBATCH --partition=commons
#SBATCH --array=1-71%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_method_comparison_v1/worker_%A_%a.out
#SBATCH --error=logs/benchmark_method_comparison_v1/worker_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_method_comparison_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/run_method_comparison_worker.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --rank 64 \
  --k 5 \
  --max-train 8192 \
  --max-validation 4096 \
  --sae-steps 2000 \
  --semi-nmf-iterations 80 \
  --semi-nmf-transform-iterations 50 \
  --ica-max-iter 1000 \
  --ica-tolerance 1e-3 \
  --n-random 20 \
  --bootstrap 500 \
  --device cuda
