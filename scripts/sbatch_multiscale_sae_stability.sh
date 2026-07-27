#!/bin/bash
#SBATCH -J mssae_stability
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/multiscale_sae_v1/stability_%j.out
#SBATCH -e logs/multiscale_sae_v1/stability_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/analyze_multiscale_sae_stability.py \
  --root results/multiscale_sae_v1 \
  --top-features 256 \
  --random-permutations 100
