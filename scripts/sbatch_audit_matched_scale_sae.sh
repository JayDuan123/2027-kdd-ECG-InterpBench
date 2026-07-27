#!/bin/bash
#SBATCH -J msae_audit
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 00:59:00
#SBATCH -o logs/sae_reconciliation/msae_audit_%j.out
#SBATCH -e logs/sae_reconciliation/msae_audit_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/audit_matched_scale_sae.py
