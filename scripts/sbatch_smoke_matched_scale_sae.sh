#!/bin/bash
#SBATCH -J msae_smoke
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:15:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/msae_smoke_%j.out
#SBATCH -e logs/sae_reconciliation/msae_smoke_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts results/probe_features/csfm_cu118_commons/pooled.npy \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out results/sae_reconciliation/matched_scale_v1/smoke/csfm_seed4311_N6144_k96_steps100.pt \
  --seed 4311 --n-features 6144 --k 96 --steps 100 --batch-size 256 \
  --checkpoint-every 25 --device cuda
