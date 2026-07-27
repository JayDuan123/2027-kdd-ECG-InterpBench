#!/usr/bin/env bash
#SBATCH -J mimic_src_index
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --array=0-5%6
#SBATCH -o logs/mimic_source_benchmark_100k_v1/index_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/index_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
PLAN_KIND="${PLAN_KIND:-full}"
if [[ "${PLAN_KIND}" == "smoke" ]]; then
  FILE="results/mimic_source_benchmark_100k_v1/activation_plan/smoke_index_commands.txt"
else
  FILE="results/mimic_source_benchmark_100k_v1/activation_plan/index_commands.txt"
fi
CMD="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${FILE}")"
test -n "${CMD}"
eval "${CMD}"
