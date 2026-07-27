#!/usr/bin/env bash
#SBATCH --job-name=requeue_smoke
#SBATCH --partition=commons
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00
#SBATCH --requeue
#SBATCH --output=logs/external_benchmark/requeue_smoke_%j.out
#SBATCH --error=logs/external_benchmark/requeue_smoke_%j.err

set -euo pipefail

STATE="/tmp/ecg_benchmark_requeue_${SLURM_JOB_ID}.state"
if [[ ! -f "${STATE}" ]]; then
  printf 'first_run host=%s time=%s\n' "$(hostname)" "$(date -Is)" > "${STATE}"
  echo "requesting requeue for ${SLURM_JOB_ID}"
  scontrol requeue "${SLURM_JOB_ID}"
  sleep 10
  exit 4
fi

cat "${STATE}"
echo "second_run_success host=$(hostname) time=$(date -Is)"
