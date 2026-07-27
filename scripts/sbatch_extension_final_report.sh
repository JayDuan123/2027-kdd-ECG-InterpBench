#!/usr/bin/env bash
#SBATCH --job-name=ext_final_report
#SBATCH --partition=commons
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/benchmark_extension_v1/final_report_%j.out
#SBATCH --error=logs/benchmark_extension_v1/final_report_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
export MPLCONFIGDIR="/tmp/mplconfig-extension-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}"
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/build_benchmark_extension_report.py
