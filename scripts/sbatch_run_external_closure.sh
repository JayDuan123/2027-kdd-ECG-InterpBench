#!/usr/bin/env bash
#SBATCH --job-name=ext_closure
#SBATCH --partition=commons
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:59:00
#SBATCH --output=logs/external_benchmark/ext_closure_%A_%a.out
#SBATCH --error=logs/external_benchmark/ext_closure_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/external_benchmark
read -r MODEL COHORT < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
 'import pandas as pd,sys; r=pd.read_csv("results/external_benchmark_v1/head_pair_manifest.csv").iloc[int(sys.argv[1])]; print(r.model_suffix,r.cohort)' \
 "${SLURM_ARRAY_TASK_ID}")
OUT="results/external_benchmark_v1/${MODEL}/${COHORT}/closure/closure_summary.json"
if [[ "${FORCE:-0}" != "1" && -s "${OUT}" ]]; then echo "Closure already complete for ${MODEL}/${COHORT}"; exit 0; fi
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_external_closure.py --model-suffix "${MODEL}" --cohort "${COHORT}"
