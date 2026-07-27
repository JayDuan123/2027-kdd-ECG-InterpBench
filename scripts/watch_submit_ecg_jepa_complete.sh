#!/usr/bin/env bash
set -u

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
LOG_DIR="${PROJECT_ROOT}/logs/ecg_jepa_act"
LOG_FILE="${LOG_DIR}/submit_watcher.log"
PID_FILE="${LOG_DIR}/submit_watcher.pid"
SBATCH_SCRIPT="${PROJECT_ROOT}/scripts/sbatch_ecg_jepa_complete_array.sh"

mkdir -p "${LOG_DIR}"
echo "$$" > "${PID_FILE}"

cd "${PROJECT_ROOT}" || exit 1

while true; do
  date >> "${LOG_FILE}"
  if sinfo >/dev/null 2>&1; then
    echo "Slurm reachable; submitting ECG-JEPA array" >> "${LOG_FILE}"
    sbatch "${SBATCH_SCRIPT}" >> "${LOG_FILE}" 2>&1
    status=$?
    echo "sbatch exit status: ${status}" >> "${LOG_FILE}"
    rm -f "${PID_FILE}"
    exit "${status}"
  fi
  echo "Slurm unreachable; retrying in 300s" >> "${LOG_FILE}"
  sleep 300
done
