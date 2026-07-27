#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"
COMMANDS_FILE="${PROJECT_ROOT}/results/analysis/model_comparison/cleanup_audit/residual_probe_commands.txt"
PARTITION="${PARTITION:-scavenge}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
CONCURRENCY="${CONCURRENCY:-12}"

cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/make_v1_cleanup_audit.py

if [[ ! -s "${COMMANDS_FILE}" ]]; then
  echo "No missing residual-probe cells remain."
  exit 0
fi

N_COMMANDS="$(wc -l < "${COMMANDS_FILE}")"
LAST_INDEX=$((N_COMMANDS - 1))
echo "Submitting ${N_COMMANDS} missing residual-probe cells to ${PARTITION} with concurrency ${CONCURRENCY}."
sbatch \
  --partition="${PARTITION}" \
  --time="${TIME_LIMIT}" \
  --array="0-${LAST_INDEX}%${CONCURRENCY}" \
  scripts/sbatch_residual_probe_array.sh
