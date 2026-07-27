#!/bin/bash
#SBATCH -J ext_sum
#SBATCH -p scavenge
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 00:59:00
#SBATCH -o logs/external_benchmark/ext_sum_%j.out
#SBATCH -e logs/external_benchmark/ext_sum_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_external_sae_steering.py
