#!/bin/bash
#SBATCH -J joint_pipe
#SBATCH -p commons
#SBATCH -c 24
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/sae_reconciliation/joint_pipe_%j.out
#SBATCH -e logs/sae_reconciliation/joint_pipe_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
export OMP_NUM_THREADS=2
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_joint_steering_batch.py \
  --base results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded --workers 6
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_joint_steering.py \
  --base results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded --bootstrap 2000
