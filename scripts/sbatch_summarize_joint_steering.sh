#!/bin/bash
#SBATCH -J joint_sum
#SBATCH -p commons
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 02:00:00
#SBATCH -o logs/sae_reconciliation/joint_sum_%j.out
#SBATCH -e logs/sae_reconciliation/joint_sum_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_joint_steering.py \
  --base results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded --bootstrap 2000
