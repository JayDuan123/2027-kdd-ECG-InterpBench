#!/bin/bash
#SBATCH -J pheno_sae
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:15:00
#SBATCH --array=0-1
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/pheno_sae_%A_%a.out
#SBATCH -e logs/sae_reconciliation/pheno_sae_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
SEEDS=(4312 4313)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT=results/sae_reconciliation/phenotype_steering/checkpoints/seed${SEED}/batchtopk_8192_k128.pt
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --manifest results/sae_reconciliation/phenotype_steering/manifest.csv \
  --out "${OUT}" --seed "${SEED}" --n-features 8192 --k 128 --steps 8000 \
  --batch-size 256 --checkpoint-every 100 --device cuda
