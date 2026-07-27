#!/usr/bin/env bash
#SBATCH --job-name=ext_dose
#SBATCH --partition=commons
#SBATCH --array=0-9%10
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_extension_v1/dose_%A_%a.out
#SBATCH --error=logs/benchmark_extension_v1/dose_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/analyze_external_dose_direction.py \
  --bootstrap 2000 \
  --n-random 20 \
  --panel-index "${SLURM_ARRAY_TASK_ID}"
