#!/usr/bin/env bash
#SBATCH --job-name=ext_base_sum
#SBATCH --partition=commons
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/benchmark_extension_v1/baseline_summary_%j.out
#SBATCH --error=logs/benchmark_extension_v1/baseline_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/summarize_external_baseline_controls.py
