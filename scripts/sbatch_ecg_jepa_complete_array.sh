#!/usr/bin/env bash
#SBATCH --job-name=ecg_jepa_act
#SBATCH --output=logs/ecg_jepa_act/%A_%a.out
#SBATCH --error=logs/ecg_jepa_act/%A_%a.err
#SBATCH --array=0-681%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
COMMANDS_FILE="${PROJECT_ROOT}/results/activations/ecg_jepa_complete_plan/all_commands.txt"

cd "${PROJECT_ROOT}"
mkdir -p logs/ecg_jepa_act

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
CMD="$(sed -n "${LINE_NO}p" "${COMMANDS_FILE}")"

if [[ -z "${CMD}" ]]; then
  echo "No command found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

echo "Running shard ${SLURM_ARRAY_TASK_ID}: ${CMD}"

OUT_DIR=""
SHARD_NAME=""
read -r -a CMD_PARTS <<< "${CMD}"
for ((i = 0; i < ${#CMD_PARTS[@]}; i++)); do
  case "${CMD_PARTS[$i]}" in
    --out-dir)
      OUT_DIR="${CMD_PARTS[$((i + 1))]}"
      ;;
    --shard-name)
      SHARD_NAME="${CMD_PARTS[$((i + 1))]}"
      ;;
  esac
done

if [[ -n "${OUT_DIR}" && -n "${SHARD_NAME}" ]]; then
  DONE_FILE="${PROJECT_ROOT}/${OUT_DIR}/${SHARD_NAME}/activation_metadata.json"
  if [[ -s "${DONE_FILE}" ]]; then
    echo "Shard ${SLURM_ARRAY_TASK_ID} already complete: ${DONE_FILE}"
    exit 0
  fi
fi

eval "${CMD}"
