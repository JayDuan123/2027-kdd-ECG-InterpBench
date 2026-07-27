#!/usr/bin/env bash
#SBATCH --job-name=extact_cursor
#SBATCH --partition=debug
#SBATCH --output=logs/external_activation_extraction/cursor_%j.out
#SBATCH --error=logs/external_activation_extraction/cursor_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
: "${COMMANDS_FILE:?COMMANDS_FILE is required}"
: "${STATE_ID:?STATE_ID is required}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_activation_extraction \
  results/activations_external_full_v1/requeue_state

STATE_FILE="${PROJECT_ROOT}/results/activations_external_full_v1/requeue_state/${STATE_ID}.cursor"
N_COMMANDS="$(wc -l < "${COMMANDS_FILE}")"
CURSOR=0
if [[ -s "${STATE_FILE}" ]]; then
  read -r CURSOR < "${STATE_FILE}"
fi

START_SECONDS="${SECONDS}"
REQUEUE_AFTER_SECONDS="${REQUEUE_AFTER_SECONDS:-210}"
MAX_SELF_REQUEUES="${MAX_SELF_REQUEUES:-1}"
echo "state_id=${STATE_ID} restart=${SLURM_RESTART_COUNT:-0} cursor=${CURSOR}/${N_COMMANDS}"

handoff_or_requeue() {
  local reason="$1"
  local restarts="${SLURM_RESTART_COUNT:-0}"
  if (( restarts >= MAX_SELF_REQUEUES )); then
    local successor
    successor="$(sbatch --parsable \
      --export=ALL,COMMANDS_FILE="${COMMANDS_FILE}",STATE_ID="${STATE_ID}",REQUEUE_AFTER_SECONDS="${REQUEUE_AFTER_SECONDS}",MAX_SELF_REQUEUES="${MAX_SELF_REQUEUES}" \
      "$(readlink -f "$0")")"
    echo "${reason}; handed off cursor=${CURSOR} to successor=${successor}"
    exit 0
  fi
  echo "${reason}; requeueing ${SLURM_JOB_ID} at cursor=${CURSOR}"
  scontrol requeue "${SLURM_JOB_ID}"
  sleep 10
  exit 0
}

while (( CURSOR < N_COMMANDS )); do
  if (( SECONDS - START_SECONDS >= REQUEUE_AFTER_SECONDS )); then
    handoff_or_requeue "time budget reached"
  fi

  echo "cursor=${CURSOR}/${N_COMMANDS}"
  set +e
  SLURM_ARRAY_TASK_ID="${CURSOR}" \
    COMMANDS_FILE="${COMMANDS_FILE}" \
    LOCK_WAIT_LOOPS=12 \
    bash scripts/sbatch_external_activation_extraction_scavenge.sh
  STATUS=$?
  set -e

  if (( STATUS == 75 )); then
    handoff_or_requeue "command ${CURSOR} is locked by another worker"
  fi
  if (( STATUS != 0 )); then
    echo "command ${CURSOR} failed with status=${STATUS}" >&2
    exit "${STATUS}"
  fi

  CURSOR=$((CURSOR + 1))
  TMP_STATE="${STATE_FILE}.tmp.${SLURM_JOB_ID}"
  printf '%s\n' "${CURSOR}" > "${TMP_STATE}"
  mv "${TMP_STATE}" "${STATE_FILE}"
done

echo "all ${N_COMMANDS} commands complete"
