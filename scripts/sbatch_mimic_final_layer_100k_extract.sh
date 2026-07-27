#!/usr/bin/env bash
#SBATCH -J mme100k_extract
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH --array=0-23%24
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_final_layer_matched_effect_100k_v1/extract_%A_%a.out
#SBATCH -e logs/mimic_final_layer_matched_effect_100k_v1/extract_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/mimic_final_layer_matched_effect_100k_v1
COMMANDS_FILE="results/activations_external_full_v1/plan_mimic_final_layer_100k_v1/all_commands.txt"
N_COMMANDS="$(wc -l < "${COMMANDS_FILE}")"
N_WORKERS="${SLURM_ARRAY_TASK_COUNT}"
WORKER_ID="${SLURM_ARRAY_TASK_ID}"
for ((COMMAND_INDEX = WORKER_ID; COMMAND_INDEX < N_COMMANDS; COMMAND_INDEX += N_WORKERS)); do
  SLURM_ARRAY_TASK_ID="${COMMAND_INDEX}" COMMANDS_FILE="${COMMANDS_FILE}" \
    bash scripts/sbatch_external_activation_extraction_scavenge.sh
done
