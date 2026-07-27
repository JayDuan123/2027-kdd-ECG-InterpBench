#!/usr/bin/env bash
#SBATCH -J mimic_src_extract
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --array=0-23%24
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_source_benchmark_100k_v1/extract_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/extract_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
COMMANDS_FILE="results/mimic_source_benchmark_100k_v1/activation_plan/all_commands.txt"
N_COMMANDS="$(wc -l < "${COMMANDS_FILE}")"
N_WORKERS="${SLURM_ARRAY_TASK_COUNT}"
for ((INDEX = SLURM_ARRAY_TASK_ID; INDEX < N_COMMANDS; INDEX += N_WORKERS)); do
  SLURM_ARRAY_TASK_ID="${INDEX}" COMMANDS_FILE="${COMMANDS_FILE}" \
    bash scripts/sbatch_external_activation_extraction_scavenge.sh
done
