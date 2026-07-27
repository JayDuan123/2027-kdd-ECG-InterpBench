#!/usr/bin/env bash
#SBATCH --job-name=external_report
#SBATCH --partition=commons
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/external_benchmark/external_report_%j.out
#SBATCH --error=logs/external_benchmark/external_report_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/build_external_benchmark_final_report.py
/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/audit_30pair_completion.py \
  --out results/external_benchmark_v1/final/completion_audit.csv
