#!/bin/bash
#SBATCH -J mm_sae_seed
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-11%12
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/mm_sae_seed_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mm_sae_seed_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / 2)); SEED_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEEDS=(4312 4313); SEED=${SEEDS[$SEED_IDX]}
read -r MODEL SUFFIX N K < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv("results/sae_reconciliation/steering_benchmark_multimodel_v1/selected_operating_points.csv").iloc[int(sys.argv[1])]; print(r.model,r.feature_suffix,int(r.N),int(r.k))' "${MODEL_IDX}")
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}
OUT="results/sae_reconciliation/steering_benchmark_multimodel_v1/models/${SAFE}/checkpoints/seed${SEED}/batchtopk_N${N}_k${K}.pt"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out "${OUT}" --seed "${SEED}" --n-features "${N}" --k "${K}" --steps 8000 \
  --batch-size 256 --checkpoint-every 100 --device cuda
