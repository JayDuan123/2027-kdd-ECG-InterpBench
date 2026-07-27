#!/usr/bin/env bash
#SBATCH -J mme100k_xaudit
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/extract_smoke_audit_%j.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/extract_smoke_audit_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/audit_mimic_final_layer_100k_smoke.py
