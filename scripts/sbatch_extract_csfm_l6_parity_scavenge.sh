#!/bin/bash
#SBATCH -J csfm_l6par
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:59:00
#SBATCH --array=0-340%32
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/extract_%A_%a.out
#SBATCH -e logs/sae_reconciliation/extract_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/extract_csfm_l6_parity_activations.py \
  --shard-id "${SLURM_ARRAY_TASK_ID}" --shard-size 64 --micro-batch 1 --device cuda
