#!/usr/bin/env bash
#SBATCH --job-name=ext2_triangle
#SBATCH --partition=scavenge
#SBATCH --array=0-11%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --requeue
#SBATCH --output=logs/benchmark_extension_v2/triangle_%A_%a.out
#SBATCH --error=logs/benchmark_extension_v2/triangle_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="/tmp/mplconfig-extension-v2-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/lib/python3.10/site-packages:/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/lib/python3.10/site-packages:/rhf/allocations/wq8/yd68/fairseq-signals:${PYTHONPATH:-}"
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/run_waveform_triangle_worker.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --model-batch "${MODEL_BATCH:-32}" \
  --device cuda
