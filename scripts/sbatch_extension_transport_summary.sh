#!/usr/bin/env bash
#SBATCH --job-name=ext_trans_sum
#SBATCH --partition=commons
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/benchmark_extension_v1/transport_summary_%j.out
#SBATCH --error=logs/benchmark_extension_v1/transport_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/summarize_transport_ladder.py
