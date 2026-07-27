#!/bin/bash
#SBATCH -J hubert_k28
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/hubert_k28_%j.out
#SBATCH -e logs/sae_reconciliation/hubert_k28_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts results/probe_features/hubert_ecg_cu118_commons/pooled.npy \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out results/sae_reconciliation/steering_benchmark_multimodel_v1/models/hubert_ecg/checkpoints/seed4311/batchtopk_N2048_k28.pt \
  --seed 4311 --n-features 2048 --k 28 --steps 8000 --batch-size 256 --checkpoint-every 100 --device cuda
