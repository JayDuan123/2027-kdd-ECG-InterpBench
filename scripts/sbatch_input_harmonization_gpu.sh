#!/usr/bin/env bash
#SBATCH --job-name=input_harm
#SBATCH --partition=commons
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/input_harmonization/%A_%a.out
#SBATCH --error=logs/input_harmonization/%A_%a.err

set -euo pipefail

ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
COMMANDS_FILE="${1:?commands file is required}"
GROUP_SIZE="${2:-1}"
cd "${ROOT}"
mkdir -p logs/input_harmonization

if (( GROUP_SIZE < 1 )); then
  echo "group size must be positive: ${GROUP_SIZE}" >&2
  exit 2
fi

START_LINE=$((SLURM_ARRAY_TASK_ID * GROUP_SIZE + 1))
END_LINE=$((START_LINE + GROUP_SIZE - 1))
mapfile -t COMMANDS < <(sed -n "${START_LINE},${END_LINE}p" "${COMMANDS_FILE}")
if (( ${#COMMANDS[@]} == 0 )); then
  echo "missing commands for task ${SLURM_ARRAY_TASK_ID}, lines ${START_LINE}-${END_LINE}" >&2
  exit 2
fi

echo "Running ${#COMMANDS[@]} commands from lines ${START_LINE}-${END_LINE}"
for COMMAND in "${COMMANDS[@]}"; do
  [[ -z "${COMMAND}" ]] && continue
  echo "${COMMAND}"
  eval "${COMMAND}"
done
