#!/bin/bash
#SBATCH -J joint_st
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH --array=0-0%24
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/joint_st_%A_%a.out
#SBATCH -e logs/sae_reconciliation/joint_st_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
GROUP_INDEX=$((SLURM_ARRAY_TASK_ID / 3)); SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3)); SEEDS=(4311 4312 4313)
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_joint_steering_task.py \
  --base results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded \
  --group-index "${GROUP_INDEX}" --seed "${SEEDS[$SEED_INDEX]}" --n-random 20
