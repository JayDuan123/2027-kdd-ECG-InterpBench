#!/bin/bash
#SBATCH -J mssae_pboot_smoke
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o logs/multiscale_sae_v1/pboot_smoke_%j.out
#SBATCH -e logs/multiscale_sae_v1/pboot_smoke_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_multiscale_sae_patient_bootstrap.py \
  --task-index 0 \
  --bootstrap-samples 100 \
  --output-root results/multiscale_sae_v1_patient_bootstrap_smoke \
  --device cuda
