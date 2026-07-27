#!/usr/bin/env bash
#SBATCH --job-name=extsteer_worker
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --output=logs/external_benchmark/extsteer_worker_%A_%a.out
#SBATCH --error=logs/external_benchmark/extsteer_worker_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
: "${TASK_START:?TASK_START is required}"
: "${TASK_END:?TASK_END is required}"
: "${SAE_SOURCE:=source}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_benchmark

N_WORKERS="${SLURM_ARRAY_TASK_COUNT:-1}"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"
FIRST=$((TASK_START + WORKER_ID))

echo "worker=${WORKER_ID}/${N_WORKERS} range=${TASK_START}-${TASK_END} sae_source=${SAE_SOURCE}"
for ((TASK_INDEX = FIRST; TASK_INDEX <= TASK_END; TASK_INDEX += N_WORKERS)); do
  echo "steering_task=${TASK_INDEX}"
  SLURM_ARRAY_TASK_ID="${TASK_INDEX}" SAE_SOURCE="${SAE_SOURCE}" \
    bash scripts/sbatch_run_external_sae_steering.sh
done

echo "worker=${WORKER_ID} complete"
