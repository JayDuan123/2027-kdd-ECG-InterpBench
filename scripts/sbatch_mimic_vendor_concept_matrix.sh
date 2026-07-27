#!/bin/bash
#SBATCH -p scavenge
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -J mimic_vmat
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_vendor_concept_matrix_%j.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_vendor_concept_matrix_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs results/multicohort

python scripts/build_mimic_vendor_concept_matrix.py \
  --out results/multicohort/mimic_vendor_concepts.csv \
  --summary-out results/multicohort/mimic_vendor_concept_summary.csv \
  --report-out results/multicohort/mimic_vendor_concept_report.md
