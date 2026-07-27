#!/bin/bash
#SBATCH -J dict_access_smoke
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o logs/dictionary_accessibility_smoke_%j.out
#SBATCH -e logs/dictionary_accessibility_smoke_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_dictionary_accessibility_worker.py \
  --group-index 5 \
  --expansion 8 \
  --batch-size 256 \
  --semantic-train-limit 4096 \
  --random-replicates 2 \
  --random-seed-base 930000 \
  --matched-budget 768 \
  --budget-replicates 2 \
  --budget-seed-base 940000 \
  --device cuda \
  --output-root results/dictionary_accessibility_e8_v1_smoke/workers
