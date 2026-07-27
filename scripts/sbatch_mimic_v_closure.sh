#!/bin/bash
#SBATCH -p scavenge
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -J mimic_v_cl
#SBATCH -o /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_v_closure_%j.out
#SBATCH -e /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark/logs/mimic_v_closure_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs results/multicohort/mimic_v_closure

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_mimic_v_closure.py \
  --concepts results/multicohort/mimic_vendor_concepts.csv \
  --labels results/multicohort/mimic_icd_label_matrix.csv \
  --out-dir results/multicohort/mimic_v_closure

python scripts/summarize_mimic_v_closure.py \
  --closure-dir results/multicohort/mimic_v_closure
