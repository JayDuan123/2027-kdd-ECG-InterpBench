#!/bin/bash
#SBATCH -J mm_artifacts
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o logs/sae_reconciliation/mm_artifacts_%j.out
#SBATCH -e logs/sae_reconciliation/mm_artifacts_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/make_multimodel_steering_artifacts.py
