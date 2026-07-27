#!/usr/bin/env bash
#SBATCH --job-name=extactpost
#SBATCH --partition=scavenge
#SBATCH --output=logs/external_activation_extraction/post_%j.out
#SBATCH --error=logs/external_activation_extraction/post_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
INDEX_COMMANDS_FILE="${INDEX_COMMANDS_FILE:-${PROJECT_ROOT}/results/activations_external/external_sae_smoke_plan/index_commands.txt}"
WATCH_ACTIVATION_ROOT="${WATCH_ACTIVATION_ROOT:-}"
EXPECTED_ACTIVATION_SHARDS="${EXPECTED_ACTIVATION_SHARDS:-}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_activation_extraction

if [[ ! -s "${INDEX_COMMANDS_FILE}" ]]; then
  echo "Missing index commands file: ${INDEX_COMMANDS_FILE}" >&2
  exit 2
fi

PLAN_SUMMARY_FILE="$(dirname "${INDEX_COMMANDS_FILE}")/plan_summary.csv"
if [[ -z "${WATCH_ACTIVATION_ROOT}" && -s "${PLAN_SUMMARY_FILE}" ]]; then
  WATCH_ACTIVATION_ROOT=$(/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python - "${PLAN_SUMMARY_FILE}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as f:
    rows = list(csv.DictReader(f))
if rows and rows[0].get("activation_dir"):
    print(rows[0]["activation_dir"])
PY
)
fi
if [[ -z "${EXPECTED_ACTIVATION_SHARDS}" && -s "${PLAN_SUMMARY_FILE}" ]]; then
  EXPECTED_ACTIVATION_SHARDS=$(/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python - "${PLAN_SUMMARY_FILE}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as f:
    total = sum(int(float(row.get("shards") or 0)) for row in csv.DictReader(f))
print(total)
PY
)
fi

if [[ -n "${WATCH_ACTIVATION_ROOT}" && -n "${EXPECTED_ACTIVATION_SHARDS}" ]]; then
  ACTUAL_SHARDS=$(find "${WATCH_ACTIVATION_ROOT}" -mindepth 2 -maxdepth 2 -name activation_metadata.json | wc -l)
  echo "Activation shard preflight: ${ACTUAL_SHARDS}/${EXPECTED_ACTIVATION_SHARDS} under ${WATCH_ACTIVATION_ROOT}"
  if [[ "${ACTUAL_SHARDS}" -lt "${EXPECTED_ACTIVATION_SHARDS}" ]]; then
    echo "Refusing to index partial activation cache." >&2
    exit 3
  fi
fi

while IFS= read -r CMD; do
  [[ -z "${CMD}" ]] && continue
  echo "Running index command: ${CMD}"
  eval "${CMD}"
done < "${INDEX_COMMANDS_FILE}"

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python scripts/build_external_sae_recon_gate.py
