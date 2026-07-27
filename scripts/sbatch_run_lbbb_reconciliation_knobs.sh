#!/bin/bash
#SBATCH -J lbbb_knobs
#SBATCH -p debug
#SBATCH --gres=gpu:1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 00:20:00
#SBATCH -o logs/sae_reconciliation/knobs_%j.out
#SBATCH -e logs/sae_reconciliation/knobs_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_lbbb_reconciliation_knobs.py --device cuda
