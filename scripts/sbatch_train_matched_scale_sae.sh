#!/bin/bash
#SBATCH -J msae_train
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-17%18
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/msae_train_%A_%a.out
#SBATCH -e logs/sae_reconciliation/msae_train_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
MANIFEST=results/sae_reconciliation/matched_scale_v1/training_manifest.csv
read -r SUFFIX N K SEED STEPS BATCH LR OUT < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv(sys.argv[1]).iloc[int(sys.argv[2])]; print(r.feature_suffix,int(r.N),int(r.k),int(r.seed),int(r.steps),int(r.batch_size),r.learning_rate,r.checkpoint)' \
  "${MANIFEST}" "${SLURM_ARRAY_TASK_ID}")
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv \
  --out "${OUT}" --seed "${SEED}" --n-features "${N}" --k "${K}" \
  --steps "${STEPS}" --batch-size "${BATCH}" --lr "${LR}" \
  --checkpoint-every 100 --device cuda
