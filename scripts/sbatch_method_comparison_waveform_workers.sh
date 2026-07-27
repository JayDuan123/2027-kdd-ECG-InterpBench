#!/usr/bin/env bash
#SBATCH --job-name=method_cmp_wave
#SBATCH --partition=commons
#SBATCH --array=0-11%6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/benchmark_method_comparison_v1/waveform_%A_%a.out
#SBATCH --error=logs/benchmark_method_comparison_v1/waveform_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_method_comparison_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="/tmp/mplconfig-method-comparison-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/lib/python3.10/site-packages:/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/lib/python3.10/site-packages:/rhf/allocations/wq8/yd68/fairseq-signals:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "${MPLCONFIGDIR}"

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/run_method_comparison_waveform_worker.py \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --model-batch "${MODEL_BATCH:-32}" \
  --device cuda
