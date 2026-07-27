#!/bin/bash
#SBATCH -J pheno_steer
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:15:00
#SBATCH --array=0-14
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/pheno_steer_%A_%a.out
#SBATCH -e logs/sae_reconciliation/pheno_steer_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
TARGETS=(lbbb rbbb pvc avb1 lafb)
SEEDS=(4311 4312 4313)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 5))
TARGET_IDX=$((SLURM_ARRAY_TASK_ID % 5))
SEED=${SEEDS[$SEED_IDX]}
TARGET=${TARGETS[$TARGET_IDX]}
if [[ "${SEED}" == "4311" ]]; then
  CKPT=results/sae_reconciliation/lbbb_fig6/checkpoints/batchtopk_8192_k128.pt
else
  CKPT=results/sae_reconciliation/phenotype_steering/checkpoints/seed${SEED}/batchtopk_8192_k128.pt
fi
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_phenotype_steering_task.py \
  --target "${TARGET}" --seed "${SEED}" --checkpoint "${CKPT}" --device cuda
