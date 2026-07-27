#!/bin/bash
#SBATCH -J mme_summary
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/summary_%j.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_mimic_final_layer_matched_effect.py
