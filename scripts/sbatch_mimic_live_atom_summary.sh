#!/usr/bin/env bash
#SBATCH -J mlive_summary
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o logs/mimic_final_layer_live_atom_matched_effect_100k_v1/summary_%j.out
#SBATCH -e logs/mimic_final_layer_live_atom_matched_effect_100k_v1/summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
MPLCONFIGDIR="/tmp/matplotlib-${USER}-${SLURM_JOB_ID}" \
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_mimic_final_layer_matched_effect.py \
  --protocol mimic_final_layer_live_atom_matched_effect_100k_v1 \
  --readout-protocol mimic_final_layer_matched_effect_100k_v1 \
  --quality-source worker \
  --manifest results/mimic_final_layer_matched_effect_100k_v1/training_manifest.csv \
  --readouts-root results/mimic_final_layer_matched_effect_100k_v1/readouts \
  --workers-root results/mimic_final_layer_live_atom_matched_effect_100k_v1/workers \
  --bootstrap-root results/mimic_final_layer_live_atom_matched_effect_100k_v1/bootstrap \
  --output-root results/mimic_final_layer_live_atom_matched_effect_100k_v1/summary
