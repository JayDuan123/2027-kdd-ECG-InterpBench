#!/usr/bin/env bash
#SBATCH --job-name=ext_base
#SBATCH --partition=commons
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/benchmark_extension_v1/baseline_%A_%a.out
#SBATCH --error=logs/benchmark_extension_v1/baseline_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
COHORTS=(chapman_f ningbo_f mimic_f)

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/run_external_baseline_controls.py \
  --cohort "${COHORTS[${SLURM_ARRAY_TASK_ID}]}" \
  --bootstrap 2000 \
  --n-random 20 \
  --pca-components 64
