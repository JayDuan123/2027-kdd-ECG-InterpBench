#!/usr/bin/env bash
#SBATCH --job-name=resid_probe
#SBATCH --partition=commons
#SBATCH --output=logs/residual_probe/%A_%a.out
#SBATCH --error=logs/residual_probe/%A_%a.err
#SBATCH --array=0-0%1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
COMMANDS_FILE="${PROJECT_ROOT}/results/analysis/model_comparison/cleanup_audit/residual_probe_commands.txt"

cd "${PROJECT_ROOT}"
mkdir -p logs/residual_probe

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
CMD="$(sed -n "${LINE_NO}p" "${COMMANDS_FILE}")"
if [[ -z "${CMD}" ]]; then
  echo "No residual-probe command for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

echo "Running residual-probe task ${SLURM_ARRAY_TASK_ID}: ${CMD}"
eval "${CMD}"
