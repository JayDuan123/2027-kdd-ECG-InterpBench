#!/usr/bin/env bash
#SBATCH --job-name=ext_final_audit
#SBATCH --partition=commons
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/benchmark_extension_v1/final_audit_%j.out
#SBATCH --error=logs/benchmark_extension_v1/final_audit_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/audit_benchmark_extension.py
