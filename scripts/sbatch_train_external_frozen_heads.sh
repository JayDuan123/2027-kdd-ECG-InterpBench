#!/bin/bash
#SBATCH -J ext_heads
#SBATCH -p scavenge
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-11%12
#SBATCH -o logs/external_benchmark/ext_heads_%A_%a.out
#SBATCH -e logs/external_benchmark/ext_heads_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/external_benchmark
read -r MODEL COHORT < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv("results/external_benchmark_v1/head_pair_manifest.csv").iloc[int(sys.argv[1])]; print(r.model_suffix,r.cohort)' \
  "${SLURM_ARRAY_TASK_ID}")
OUT_DIR="results/external_benchmark_v1/${MODEL}/${COHORT}"
if [[ -s "${OUT_DIR}/frozen_heads.joblib" \
   && -s "${OUT_DIR}/frozen_heads_metrics.csv" \
   && -s "${OUT_DIR}/head_summary.json" ]]; then
  echo "Head outputs already complete for ${MODEL}/${COHORT}; skipping."
  exit 0
fi
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_external_frozen_heads.py \
  --model-suffix "${MODEL}" --cohort "${COHORT}"
