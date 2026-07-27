#!/bin/bash
#SBATCH -J mm_sae_fix3
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-3%4
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/mm_sae_fix3_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mm_sae_fix3_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
MODELS=(CARDIAC-FM CARDIAC-FM HuBERT-ECG HuBERT-ECG)
SUFFIXES=(cardiac_fm_cu118_commons cardiac_fm_cu118_commons hubert_ecg_cu118_commons hubert_ecg_cu118_commons)
KS=(64 32 64 32); I=${SLURM_ARRAY_TASK_ID}; MODEL=${MODELS[$I]}; SUFFIX=${SUFFIXES[$I]}; K=${KS[$I]}; N=2048
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}
OUT="results/sae_reconciliation/steering_benchmark_multimodel_v1/models/${SAFE}/checkpoints/seed4311/batchtopk_N${N}_k${K}.pt"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out "${OUT}" --seed 4311 --n-features "${N}" --k "${K}" --steps 8000 \
  --batch-size 256 --checkpoint-every 100 --device cuda
