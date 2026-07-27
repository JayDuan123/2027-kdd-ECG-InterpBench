#!/bin/bash
#SBATCH -J mm_heads
#SBATCH -p scavenge
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/mm_heads_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mm_heads_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
MODELS=(CSFM CARDIAC-FM ECG-FM ECG-JEPA HuBERT-ECG ST-MEM)
SUFFIXES=(csfm_cu118_commons cardiac_fm_cu118_commons ecg_fm_cu118_commons ecg_jepa_cu118_commons hubert_ecg_cu118_commons st_mem_cu118_commons)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}; SUFFIX=${SUFFIXES[$SLURM_ARRAY_TASK_ID]}
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/fit_multimodel_steering_heads.py \
  --model "${MODEL}" --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --out "results/sae_reconciliation/steering_benchmark_multimodel_v1/models/${SAFE}/frozen_heads.joblib"
