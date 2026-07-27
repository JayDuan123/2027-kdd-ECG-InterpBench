#!/bin/bash
#SBATCH -J me_intervene
#SBATCH -p scavenge
#SBATCH --qos=nots_scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 6
#SBATCH --mem=56G
#SBATCH -t 00:59:00
#SBATCH --array=0-17%12
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o logs/matched_effect_gpu_%A_%a.out
#SBATCH -e logs/matched_effect_gpu_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/run_final_layer_matched_effect_worker.py \
  --source-index "${SLURM_ARRAY_TASK_ID}" \
  --device cuda
