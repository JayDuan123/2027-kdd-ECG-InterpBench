#!/usr/bin/env bash
#SBATCH -J mimic_src_dictsum
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/dictionary_summary_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/dictionary_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_dictionary_accessibility.py \
  --workers-root results/mimic_source_benchmark_100k_v1/dictionary/workers \
  --output-root results/mimic_source_benchmark_100k_v1/dictionary/summary \
  --expected-groups 30 --expected-feature-rows 208 --expected-target-rows 1248 \
  --skip-calibration-reproduction
