#!/bin/bash
#SBATCH -J exp_art
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o logs/sae_reconciliation/exp_art_%j.out
#SBATCH -e logs/sae_reconciliation/exp_art_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/make_expanded_steering_artifacts.py
