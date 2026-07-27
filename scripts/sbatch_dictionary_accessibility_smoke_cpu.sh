#!/bin/bash
#SBATCH -J dict_access_smoke_cpu
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/dictionary_accessibility_smoke_cpu_%j.out
#SBATCH -e logs/dictionary_accessibility_smoke_cpu_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_dictionary_accessibility_worker.py \
  --group-index 5 \
  --expansion 8 \
  --batch-size 128 \
  --semantic-train-limit 4096 \
  --random-replicates 2 \
  --random-seed-base 930000 \
  --matched-budget 768 \
  --budget-replicates 2 \
  --budget-seed-base 940000 \
  --device cpu \
  --output-root results/dictionary_accessibility_e8_v1_smoke_cpu/workers
