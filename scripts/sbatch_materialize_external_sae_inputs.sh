#!/usr/bin/env bash
#SBATCH --job-name=ext_sae_inputs
#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:59:00
#SBATCH --output=logs/external_benchmark/ext_sae_inputs_%A_%a.out
#SBATCH --error=logs/external_benchmark/ext_sae_inputs_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
cd "${PROJECT_ROOT}"
mkdir -p logs/external_benchmark

read -r MODEL COHORT < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv("results/external_benchmark_v1/head_pair_manifest.csv").iloc[int(sys.argv[1])]; print(r.model_suffix,r.cohort)' \
  "${SLURM_ARRAY_TASK_ID}")

OUT="results/external_benchmark_v1/${MODEL}/${COHORT}/cohort_adapted_sae/materialization.json"
if [[ -s "${OUT}" ]]; then
  echo "Cohort-adapted SAE input already materialized for ${MODEL}/${COHORT}; skipping."
  exit 0
fi

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/materialize_external_sae_inputs.py \
  --model-suffix "${MODEL}" \
  --cohort "${COHORT}"
