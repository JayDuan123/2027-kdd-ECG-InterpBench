#!/bin/bash
#SBATCH -J mimic_plan
#SBATCH -p scavenge
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/external_benchmark/mimic_plan_%j.out
#SBATCH -e logs/external_benchmark/mimic_plan_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/external_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/build_mimic_external_main_plan.py
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/build_full_external_benchmark_plan.py --cohorts ningbo
