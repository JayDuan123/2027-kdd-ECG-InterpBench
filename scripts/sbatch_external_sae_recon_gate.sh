#!/usr/bin/env bash
#SBATCH --job-name=extsaerg
#SBATCH --partition=scavenge
#SBATCH --output=logs/external_sae_recon_gate_%j.out
#SBATCH --error=logs/external_sae_recon_gate_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs

"${PYTHON}" scripts/build_external_sae_recon_gate.py "$@"
