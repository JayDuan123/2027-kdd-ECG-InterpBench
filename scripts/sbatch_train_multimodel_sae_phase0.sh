#!/bin/bash
#SBATCH -J mm_sae_p0
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-5%6
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/mm_sae_p0_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mm_sae_p0_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
MODELS=(CSFM CARDIAC-FM ECG-FM ECG-JEPA HuBERT-ECG ST-MEM)
SUFFIXES=(csfm_cu118_commons cardiac_fm_cu118_commons ecg_fm_cu118_commons ecg_jepa_cu118_commons hubert_ecg_cu118_commons st_mem_cu118_commons)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}; SUFFIX=${SUFFIXES[$SLURM_ARRAY_TASK_ID]}
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}
OUT="results/sae_reconciliation/steering_benchmark_multimodel_v1/models/${SAFE}/checkpoints/seed4311/batchtopk_N8192_k128.pt"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out "${OUT}" --seed 4311 --n-features 8192 --k 128 --steps 8000 \
  --batch-size 256 --checkpoint-every 100 --device cuda
