#!/usr/bin/env bash
#SBATCH --job-name=method_cmp_wsum
#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/benchmark_method_comparison_v1/waveform_summary_%j.out
#SBATCH --error=logs/benchmark_method_comparison_v1/waveform_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_method_comparison_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/summarize_method_comparison_waveform.py \
  --bootstrap 2000 \
  --hierarchical-bootstrap 10000
