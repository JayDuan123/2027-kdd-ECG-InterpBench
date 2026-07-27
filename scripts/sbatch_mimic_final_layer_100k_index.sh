#!/usr/bin/env bash
#SBATCH -J mme100k_index
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --array=0-3%4
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/index_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/index_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
COMMANDS_FILE="results/activations_external_full_v1/plan_mimic_final_layer_100k_v1/index_commands.txt"
LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
CMD="$(sed -n "${LINE_NO}p" "${COMMANDS_FILE}")"
[[ -n "${CMD}" ]] || { echo "missing index command ${SLURM_ARRAY_TASK_ID}" >&2; exit 2; }
eval "${CMD}"
