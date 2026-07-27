#!/bin/bash
#SBATCH -J mssae_pboot_sum
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/multiscale_sae_v1/pboot_summary_%j.out
#SBATCH -e logs/multiscale_sae_v1/pboot_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_multiscale_sae_patient_bootstrap.py \
  --root results/multiscale_sae_v1 \
  --split test
