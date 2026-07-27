#!/bin/bash
#SBATCH -J ext_steer
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --array=0-107%24
#SBATCH --requeue
#SBATCH -o logs/external_benchmark/ext_steer_%A_%a.out
#SBATCH -e logs/external_benchmark/ext_steer_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
: "${SAE_SOURCE:=source}"
read -r MODEL COHORT TARGET SEED < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
 'import pandas as pd,sys; r=pd.read_csv("results/external_benchmark_v1/steering_manifest.csv").iloc[int(sys.argv[1])]; print(r.model_suffix,r.cohort,r.target,int(r.seed))' \
 "${SLURM_ARRAY_TASK_ID}")
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_external_sae_steering_task.py \
 --model-suffix "${MODEL}" --cohort "${COHORT}" --target "${TARGET}" --seed "${SEED}" \
 --sae-source "${SAE_SOURCE}" --device cuda
