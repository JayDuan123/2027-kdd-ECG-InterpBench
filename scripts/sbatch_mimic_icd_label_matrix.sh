#!/bin/bash
#SBATCH -p scavenge
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J mimic_lab
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_icd_label_matrix_%j.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_icd_label_matrix_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs results/multicohort

python scripts/build_mimic_icd_label_matrix.py \
  --out results/multicohort/mimic_icd_label_matrix.csv \
  --report-out results/multicohort/mimic_icd_label_matrix_report.md
