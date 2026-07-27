#!/usr/bin/env bash
#SBATCH --job-name=ext_wave
#SBATCH --partition=commons
#SBATCH --array=0-11%8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/benchmark_extension_v1/waveform_%A_%a.out
#SBATCH --error=logs/benchmark_extension_v1/waveform_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="/tmp/mplconfig-extension-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export PYTHONPATH="/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/lib/python3.10/site-packages:/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/lib/python3.10/site-packages:/rhf/allocations/wq8/yd68/fairseq-signals:${PYTHONPATH:-}"
mkdir -p "${MPLCONFIGDIR}"

ARGS=(
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --sample-size "${SAMPLE_SIZE:-256}"
  --model-batch "${MODEL_BATCH:-32}"
  --device cuda
)
if [[ -n "${OUT_DIR:-}" ]]; then ARGS+=(--out "${OUT_DIR}"); fi
if [[ "${FORCE:-0}" == "1" ]]; then ARGS+=(--force); fi

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/run_waveform_intervention_worker.py "${ARGS[@]}"
