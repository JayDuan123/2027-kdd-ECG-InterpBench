#!/bin/bash
#SBATCH -J l6par_dbgfill
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH --array=0-3
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/debug_fill_%A_%a.out
#SBATCH -e logs/sae_reconciliation/debug_fill_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
PY=/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python

# Shard 0 was validated by the smoke test. Four workers cover 1..340 without
# creating hundreds of independently queued debug jobs.
for ((SHARD=SLURM_ARRAY_TASK_ID+1; SHARD<=340; SHARD+=4)); do
  "${PY}" scripts/extract_csfm_l6_parity_activations.py \
    --shard-id "${SHARD}" --shard-size 64 --micro-batch 1 --device cuda
done
