#!/usr/bin/env bash
#SBATCH -J mimic_src_accsum
#SBATCH -p commons
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/access_summary_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/access_summary_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_accessibility_calibration.py \
  --workers-root results/mimic_source_benchmark_100k_v1/accessibility/workers \
  --output-root results/mimic_source_benchmark_100k_v1/accessibility/summary \
  --expected-cells 90 --expected-concepts 7
