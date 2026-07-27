#!/bin/bash
#SBATCH -J mssae_select
#SBATCH -p commons
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o logs/multiscale_sae_v1/select_%j.out
#SBATCH -e logs/multiscale_sae_v1/select_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/select_multiscale_sae_operating_points.py \
  --root results/multiscale_sae_v1
