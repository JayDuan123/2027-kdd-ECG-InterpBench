#!/usr/bin/env bash
#SBATCH -J mme100k_xsmoke
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-3%4
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/extract_smoke_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/extract_smoke_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/mimic_final_layer_matched_effect_100k_v1
COMMANDS_FILE="results/activations_external_full_v1/plan_mimic_final_layer_100k_v1/smoke_commands.txt" \
  bash scripts/sbatch_external_activation_extraction_scavenge.sh
