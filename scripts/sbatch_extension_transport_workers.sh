#!/usr/bin/env bash
#SBATCH --job-name=ext_transport
#SBATCH --partition=scavenge
#SBATCH --array=0-71%12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --requeue
#SBATCH --output=logs/benchmark_extension_v1/transport_%A_%a.out
#SBATCH --error=logs/benchmark_extension_v1/transport_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/benchmark_extension_v1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="/tmp/mplconfig-extension-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}"

ARGS=(
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --device cuda
  --steps "${STEPS:-2000}"
  --batch-size "${BATCH_SIZE:-256}"
  --fewshot-sizes "${FEWSHOT_SIZES:-128,512,2048}"
  --max-train-eval "${MAX_TRAIN_EVAL:-8192}"
  --max-val-eval "${MAX_VAL_EVAL:-4096}"
  --max-test-eval "${MAX_TEST_EVAL:-8192}"
  --covariance-sample "${COVARIANCE_SAMPLE:-8192}"
  --n-random "${N_RANDOM:-20}"
)
if [[ -n "${OUT_DIR:-}" ]]; then ARGS+=(--out "${OUT_DIR}"); fi
if [[ "${FORCE:-0}" == "1" ]]; then ARGS+=(--force); fi

/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python \
  scripts/run_transport_ladder_worker.py "${ARGS[@]}"
