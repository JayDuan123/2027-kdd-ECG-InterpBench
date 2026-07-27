#!/bin/bash
#SBATCH -J sae_topk_sum
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:15:00
#SBATCH -o logs/sae_extension/topk_v2_summary_%j.out
#SBATCH -e logs/sae_extension/topk_v2_summary_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/summarize_sae_topk_v2.py
