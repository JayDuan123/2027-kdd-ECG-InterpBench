#!/bin/bash
#SBATCH -J mme_prepare
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/mimic_final_layer_matched_effect_v1/prepare_%j.out
#SBATCH -e logs/mimic_final_layer_matched_effect_v1/prepare_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/prepare_mimic_final_layer_matched_effect.py
