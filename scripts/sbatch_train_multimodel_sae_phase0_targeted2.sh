#!/bin/bash
#SBATCH -J mm_sae_fix2
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-4%5
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/mm_sae_fix2_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mm_sae_fix2_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
MODELS=(CARDIAC-FM CARDIAC-FM HuBERT-ECG HuBERT-ECG ST-MEM)
SUFFIXES=(cardiac_fm_cu118_commons cardiac_fm_cu118_commons hubert_ecg_cu118_commons hubert_ecg_cu118_commons st_mem_cu118_commons)
NS=(2048 1024 2048 1024 8192)
KS=(128 128 128 128 512)
I=${SLURM_ARRAY_TASK_ID}; MODEL=${MODELS[$I]}; SUFFIX=${SUFFIXES[$I]}; N=${NS[$I]}; K=${KS[$I]}
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}
OUT="results/sae_reconciliation/steering_benchmark_multimodel_v1/models/${SAFE}/checkpoints/seed4311/batchtopk_N${N}_k${K}.pt"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out "${OUT}" --seed 4311 --n-features "${N}" --k "${K}" --steps 8000 \
  --batch-size 256 --checkpoint-every 100 --device cuda
