#!/bin/bash
#SBATCH -J mssae_figures
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/multiscale_sae_v1/figures_%j.out
#SBATCH -e logs/multiscale_sae_v1/figures_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/make_multiscale_sae_figures.py \
  --root results/multiscale_sae_v1
