#!/bin/bash
#SBATCH -J mssae_kn_audit
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/multiscale_sae_fixed_k_over_n_middepth_v1/audit_%j.out
#SBATCH -e logs/multiscale_sae_fixed_k_over_n_middepth_v1/audit_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/audit_multiscale_sae.py \
  --root results/multiscale_sae_fixed_k_over_n_middepth_v1
