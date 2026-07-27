#!/bin/bash
#SBATCH -J exp_range
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:29:00
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/exp_range_%j.out
#SBATCH -e logs/sae_reconciliation/exp_range_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
START=${1:?start task id required}; END=${2:?end task id required}
for ((TASK_ID=START; TASK_ID<=END; TASK_ID++)); do
  export SLURM_ARRAY_TASK_ID=${TASK_ID}
  bash scripts/sbatch_run_expanded_steering_array.sh
done
