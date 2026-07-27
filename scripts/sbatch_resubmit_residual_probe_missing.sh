#!/usr/bin/env bash
#SBATCH --job-name=resid_retry
#SBATCH --partition=commons
#SBATCH --output=logs/residual_probe_retry/%j.out
#SBATCH --error=logs/residual_probe_retry/%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/residual_probe_retry

PARTITION="${PARTITION:-scavenge}" \
TIME_LIMIT="${TIME_LIMIT:-01:00:00}" \
CONCURRENCY="${CONCURRENCY:-12}" \
  scripts/resubmit_residual_probe_missing.sh
