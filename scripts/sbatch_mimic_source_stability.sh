#!/usr/bin/env bash
#SBATCH -J mimic_src_stab
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/stability_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/stability_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/analyze_multiscale_sae_stability.py \
  --root results/mimic_source_benchmark_100k_v1 --top-features 256 --random-permutations 100
