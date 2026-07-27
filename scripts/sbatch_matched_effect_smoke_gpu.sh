#!/bin/bash
#SBATCH -J me_smoke_gpu
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o logs/matched_effect_smoke_gpu_%j.out
#SBATCH -e logs/matched_effect_smoke_gpu_%j.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_final_layer_matched_effect_worker.py \
  --source-index 0 \
  --device cuda \
  --max-records-per-split 512 \
  --semantic-train-limit 512 \
  --budget-replicates 2 \
  --readout-root results/final_layer_matched_effect_v1_smoke/readouts \
  --output-root results/final_layer_matched_effect_v1_smoke/workers
