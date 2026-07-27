#!/usr/bin/env bash
#SBATCH --job-name=ext_pool_leace
#SBATCH --partition=commons
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/external_benchmark/ext_pool_leace_%A_%a.out
#SBATCH --error=logs/external_benchmark/ext_pool_leace_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/external_benchmark

read -r MODEL COHORT < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv("results/external_benchmark_v1/head_pair_manifest.csv").iloc[int(sys.argv[1])]; print(r.model_suffix,r.cohort)' \
  "${SLURM_ARRAY_TASK_ID}")

SUMMARY="results/external_benchmark_v1/${MODEL}/${COHORT}/pooled_leace/summary/pooled_leace_summary.json"
if [[ -s "${SUMMARY}" ]]; then
  echo "Pooled LEACE already complete for ${MODEL}/${COHORT}; skipping."
  exit 0
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_external_pooled_leace.py \
  --model-suffix "${MODEL}" --cohort "${COHORT}"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_external_pooled_leace.py \
  --model-suffix "${MODEL}" --cohort "${COHORT}" --bootstrap 2000
