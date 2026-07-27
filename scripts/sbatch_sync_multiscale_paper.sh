#!/bin/bash
#SBATCH -J mssae_paper_sync
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o logs/multiscale_sae_v1/paper_sync_%j.out
#SBATCH -e logs/multiscale_sae_v1/paper_sync_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/sync_multiscale_paper_artifacts.py
