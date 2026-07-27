#!/bin/bash
#SBATCH -p scavenge
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J mimic_icd_gate
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_icd_task_feasibility_%j.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_icd_task_feasibility_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs results/multicohort

python scripts/build_mimic_icd_task_feasibility.py \
  --out results/multicohort/external_task_feasibility.csv \
  --min-positive 50 \
  --min-negative 50
