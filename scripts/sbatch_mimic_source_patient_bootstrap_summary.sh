#!/usr/bin/env bash
#SBATCH -J mimic_src_pbsum
#SBATCH -p commons
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/pboot_summary_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/pboot_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_multiscale_sae_patient_bootstrap.py \
  --root results/mimic_source_benchmark_100k_v1 --split test
