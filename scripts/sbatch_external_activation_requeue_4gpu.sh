#!/usr/bin/env bash
#SBATCH --job-name=extact_4gpu
#SBATCH --partition=debug
#SBATCH --output=logs/external_activation_extraction/cursor4_%j.out
#SBATCH --error=logs/external_activation_extraction/cursor4_%j.err
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
: "${COMMANDS_FILE:?COMMANDS_FILE is required}"
: "${STATE_ID:?STATE_ID is required}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_activation_extraction \
  results/activations_external_full_v1/requeue_state

N_COMMANDS="$(wc -l < "${COMMANDS_FILE}")"
REQUEUE_AFTER_SECONDS="${REQUEUE_AFTER_SECONDS:-1500}"
MAX_SELF_REQUEUES="${MAX_SELF_REQUEUES:-1}"
START_EPOCH="$(date +%s)"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if (( ${#GPU_IDS[@]} < 4 )); then
  echo "Expected four allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 2
fi

run_lane() {
  local lane="$1" gpu_id="$2"
  local state_file="${PROJECT_ROOT}/results/activations_external_full_v1/requeue_state/${STATE_ID}.lane${lane}.cursor"
  local cursor="${lane}"
  if [[ -s "${state_file}" ]]; then
    read -r cursor < "${state_file}"
  fi
  echo "lane=${lane} gpu=${gpu_id} restart=${SLURM_RESTART_COUNT:-0} cursor=${cursor}/${N_COMMANDS}"

  while (( cursor < N_COMMANDS )); do
    if (( $(date +%s) - START_EPOCH >= REQUEUE_AFTER_SECONDS )); then
      echo "lane=${lane} time budget reached at cursor=${cursor}"
      break
    fi

    echo "lane=${lane} cursor=${cursor}/${N_COMMANDS}"
    set +e
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      SLURM_ARRAY_TASK_ID="${cursor}" \
      COMMANDS_FILE="${COMMANDS_FILE}" \
      LOCK_WAIT_LOOPS=12 \
      bash scripts/sbatch_external_activation_extraction_scavenge.sh
    local status=$?
    set -e

    if (( status == 75 )); then
      echo "lane=${lane} command=${cursor} is locked; retrying after requeue"
      break
    fi
    if (( status != 0 )); then
      echo "lane=${lane} command=${cursor} failed with status=${status}" >&2
      return "${status}"
    fi

    cursor=$((cursor + 4))
    local tmp_state="${state_file}.tmp.${SLURM_JOB_ID}"
    printf '%s\n' "${cursor}" > "${tmp_state}"
    mv "${tmp_state}" "${state_file}"
  done
}

pids=()
for lane in 0 1 2 3; do
  run_lane "${lane}" "${GPU_IDS[$lane]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if (( failed )); then
  echo "At least one lane failed" >&2
  exit 3
fi

complete=1
for lane in 0 1 2 3; do
  state_file="${PROJECT_ROOT}/results/activations_external_full_v1/requeue_state/${STATE_ID}.lane${lane}.cursor"
  cursor="${lane}"
  [[ -s "${state_file}" ]] && read -r cursor < "${state_file}"
  if (( cursor < N_COMMANDS )); then
    complete=0
  fi
done

if (( complete )); then
  echo "all ${N_COMMANDS} commands complete across four lanes"
  exit 0
fi

if (( ${SLURM_RESTART_COUNT:-0} >= MAX_SELF_REQUEUES )); then
  successor="$(sbatch --parsable \
    --export=ALL,COMMANDS_FILE="${COMMANDS_FILE}",STATE_ID="${STATE_ID}",REQUEUE_AFTER_SECONDS="${REQUEUE_AFTER_SECONDS}",MAX_SELF_REQUEUES="${MAX_SELF_REQUEUES}" \
    "$(readlink -f "$0")")"
  echo "four-lane time budget reached; handed off to successor=${successor}"
  exit 0
fi
echo "four-lane time budget reached; requeueing ${SLURM_JOB_ID}"
scontrol requeue "${SLURM_JOB_ID}"
sleep 10
