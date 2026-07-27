#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
LOG_FILE="${PROJECT_ROOT}/logs/watch_submit_hubert_cardiac.log"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-72}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

cd "${PROJECT_ROOT}"
mkdir -p logs

echo "$(date): watcher started, max_attempts=${MAX_ATTEMPTS}, sleep=${SLEEP_SECONDS}s" >> "${LOG_FILE}"

for ((attempt = 1; attempt <= MAX_ATTEMPTS; attempt++)); do
  echo "$(date): attempt ${attempt}/${MAX_ATTEMPTS}" >> "${LOG_FILE}"
  if sinfo -s >> "${LOG_FILE}" 2>&1; then
    echo "$(date): Slurm reachable; submitting HuBERT-ECG and CARDIAC-FM" >> "${LOG_FILE}"
    sbatch scripts/sbatch_hubert_ecg_cu118_commons_array.sh >> "${LOG_FILE}" 2>&1
    sbatch scripts/sbatch_cardiac_fm_cu118_commons_array.sh >> "${LOG_FILE}" 2>&1
    echo "$(date): submissions attempted; watcher exiting" >> "${LOG_FILE}"
    exit 0
  fi
  sleep "${SLEEP_SECONDS}"
done

echo "$(date): watcher exhausted attempts without reachable Slurm" >> "${LOG_FILE}"
exit 1
