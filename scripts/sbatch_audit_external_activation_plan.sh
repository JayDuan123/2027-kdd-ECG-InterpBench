#!/usr/bin/env bash
#SBATCH --job-name=ext_audit
#SBATCH --partition=commons
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=logs/external_benchmark/ext_audit_%j.out
#SBATCH --error=logs/external_benchmark/ext_audit_%j.err

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
: "${COMMANDS_FILE:?COMMANDS_FILE is required}"
: "${AUDIT_OUT:?AUDIT_OUT is required}"

cd "${PROJECT_ROOT}"
mkdir -p logs/external_benchmark "$(dirname "${AUDIT_OUT}")"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/audit_external_activation_commands.py \
  --commands "${COMMANDS_FILE}" \
  --out "${AUDIT_OUT}" \
  --check-finite
