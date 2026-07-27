#!/usr/bin/env bash
#SBATCH -J mimic_src_smoke_x
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH --array=0-5%6
#SBATCH -o logs/mimic_source_benchmark_100k_v1/smoke_extract_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/smoke_extract_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/mimic_source_benchmark_100k_v1
COMMANDS_FILE="results/mimic_source_benchmark_100k_v1/activation_plan/smoke_commands.txt" \
  bash scripts/sbatch_external_activation_extraction_scavenge.sh
