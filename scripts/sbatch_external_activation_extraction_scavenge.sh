#!/usr/bin/env bash
#SBATCH --job-name=extacts
#SBATCH --partition=scavenge
#SBATCH --output=logs/external_activation_extraction/%A_%a.out
#SBATCH --error=logs/external_activation_extraction/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:59:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
COMMANDS_FILE="${COMMANDS_FILE:-${PROJECT_ROOT}/results/activations_external/external_sae_plan/all_commands.txt}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_activation_extraction

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
CMD="$(sed -n "${LINE_NO}p" "${COMMANDS_FILE}")"

if [[ -z "${CMD}" ]]; then
  echo "No command found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

echo "Running external activation shard ${SLURM_ARRAY_TASK_ID}: ${CMD}"

OUT_DIR=""
SHARD_NAME=""
MODEL_SUFFIX=""
COHORT=""
read -r -a CMD_PARTS <<< "${CMD}"
for ((i = 0; i < ${#CMD_PARTS[@]}; i++)); do
  case "${CMD_PARTS[$i]}" in
    --out-dir)
      OUT_DIR="${CMD_PARTS[$((i + 1))]}"
      ;;
    --shard-name)
      SHARD_NAME="${CMD_PARTS[$((i + 1))]}"
      ;;
    --model)
      MODEL_SUFFIX="${CMD_PARTS[$((i + 1))]}"
      ;;
    --cohort)
      COHORT="${CMD_PARTS[$((i + 1))]}"
      ;;
  esac
done

if [[ -n "${OUT_DIR}" && -n "${SHARD_NAME}" && -n "${MODEL_SUFFIX}" && -n "${COHORT}" ]]; then
  if [[ "${OUT_DIR}" = /* ]]; then
    DONE_FILE="${OUT_DIR}/${MODEL_SUFFIX}/${COHORT}/${SHARD_NAME}/activation_metadata.json"
  else
    DONE_FILE="${PROJECT_ROOT}/${OUT_DIR}/${MODEL_SUFFIX}/${COHORT}/${SHARD_NAME}/activation_metadata.json"
  fi
  if [[ -s "${DONE_FILE}" ]]; then
    echo "Shard ${SLURM_ARRAY_TASK_ID} already complete: ${DONE_FILE}"
    exit 0
  fi
  mkdir -p "$(dirname "${DONE_FILE}")"
  LOCK_DIR="${DONE_FILE}.lock"
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "Shard ${SLURM_ARRAY_TASK_ID} is owned by another worker; waiting for completion."
    LOCK_WAIT_LOOPS="${LOCK_WAIT_LOOPS:-360}"
    for _ in $(seq 1 "${LOCK_WAIT_LOOPS}"); do
      if [[ -s "${DONE_FILE}" ]]; then
        echo "Concurrent worker completed shard ${SLURM_ARRAY_TASK_ID}: ${DONE_FILE}"
        exit 0
      fi
      sleep 5
    done
    echo "Timed out waiting for shard lock ${LOCK_DIR}" >&2
    exit 75
  fi
  trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT
  if [[ -s "${DONE_FILE}" ]]; then
    echo "Shard ${SLURM_ARRAY_TASK_ID} completed before lock acquisition: ${DONE_FILE}"
    exit 0
  fi
fi

eval "${CMD}"
