#!/bin/bash
#SBATCH -J steer_heads
#SBATCH -p scavenge
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/steer_heads_%j.out
#SBATCH -e logs/sae_reconciliation/steer_heads_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/fit_steering_benchmark_heads.py
