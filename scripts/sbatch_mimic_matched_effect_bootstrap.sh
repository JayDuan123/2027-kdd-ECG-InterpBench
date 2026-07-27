#!/bin/bash
#SBATCH -J mme_bootstrap
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH --array=0-5%6
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/bootstrap_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/bootstrap_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/bootstrap_mimic_final_layer_matched_effect.py \
  --model-index "${SLURM_ARRAY_TASK_ID}" \
  --bootstrap-draws 2000
