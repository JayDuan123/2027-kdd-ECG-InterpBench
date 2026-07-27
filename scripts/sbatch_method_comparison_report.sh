#!/usr/bin/env bash
#SBATCH --job-name=method_cmp_report
#SBATCH --partition=commons
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/benchmark_method_comparison_v1/report_%j.out
#SBATCH --error=logs/benchmark_method_comparison_v1/report_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_method_comparison_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="/tmp/mplconfig-method-comparison-report-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/build_method_comparison_report.py
