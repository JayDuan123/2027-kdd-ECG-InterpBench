#!/usr/bin/env bash
#SBATCH --job-name=extact_worker
#SBATCH --partition=commons
#SBATCH --output=logs/external_activation_extraction/%A_%a.out
#SBATCH --error=logs/external_activation_extraction/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
: "${COMMANDS_FILE:?COMMANDS_FILE is required}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_activation_extraction

N_COMMANDS="$(wc -l < "${COMMANDS_FILE}")"
N_WORKERS="${SLURM_ARRAY_TASK_COUNT:?SLURM_ARRAY_TASK_COUNT is required}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

echo "worker=${WORKER_ID}/${N_WORKERS} commands=${N_COMMANDS} file=${COMMANDS_FILE}"

for ((COMMAND_INDEX = WORKER_ID; COMMAND_INDEX < N_COMMANDS; COMMAND_INDEX += N_WORKERS)); do
  echo "worker=${WORKER_ID} command_index=${COMMAND_INDEX}"
  SLURM_ARRAY_TASK_ID="${COMMAND_INDEX}" \
    COMMANDS_FILE="${COMMANDS_FILE}" \
    bash scripts/sbatch_external_activation_extraction_scavenge.sh
done

echo "worker=${WORKER_ID} complete"
