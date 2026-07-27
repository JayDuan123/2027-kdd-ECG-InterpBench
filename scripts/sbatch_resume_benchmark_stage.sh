#!/usr/bin/env bash
#SBATCH --job-name=fm_bench
#SBATCH --partition=commons
#SBATCH --output=logs/benchmark_stage/%x_%A_%a.out
#SBATCH --error=logs/benchmark_stage/%x_%A_%a.err
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"

cd "${PROJECT_ROOT}"
mkdir -p logs/benchmark_stage

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    MODEL="hubert_ecg"
    ACTIVATION_DIR="results/activations/hubert_ecg_cu118_commons"
    OUT_SUFFIX="hubert_ecg_cu118_commons"
    ;;
  1)
    MODEL="ecg_fm"
    ACTIVATION_DIR="results/activations/ecg_fm_cu118_commons"
    OUT_SUFFIX="ecg_fm_cu118_commons"
    ;;
  2)
    MODEL="ecg_jepa"
    ACTIVATION_DIR="results/activations/ecg_jepa_cu118_commons"
    OUT_SUFFIX="ecg_jepa_cu118_commons"
    ;;
  3)
    MODEL="st_mem"
    ACTIVATION_DIR="results/activations/st_mem_cu118_commons"
    OUT_SUFFIX="st_mem_cu118_commons"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

INDEX_DIR="results/activation_index/${OUT_SUFFIX}"
FEATURE_DIR="results/probe_features/${OUT_SUFFIX}"
PROBE_DIR="results/probe/${OUT_SUFFIX}"
ANALYSIS_DIR="results/analysis/${OUT_SUFFIX}"

echo "Benchmark resume stage for ${MODEL} (${OUT_SUFFIX})"
echo "Activation dir: ${ACTIVATION_DIR}"

if [[ ! -d "${ACTIVATION_DIR}" ]]; then
  echo "Missing activation dir: ${ACTIVATION_DIR}" >&2
  exit 3
fi

if [[ ! -s "${INDEX_DIR}/index_report.json" ]]; then
  "${PYTHON}" scripts/build_activation_index.py \
    --model "${MODEL}" \
    --activation-dir "${ACTIVATION_DIR}" \
    --out-dir "${INDEX_DIR}"
else
  echo "Skip activation index: ${INDEX_DIR}/index_report.json exists"
fi

if [[ ! -s "${FEATURE_DIR}/probe_features_report.json" ]]; then
  "${PYTHON}" scripts/build_probe_features.py \
    --index-dir "${INDEX_DIR}" \
    --out-dir "${FEATURE_DIR}" \
    --layers all
else
  echo "Skip probe features: ${FEATURE_DIR}/probe_features_report.json exists"
fi

if [[ ! -s "${PROBE_DIR}/probe_report.json" ]]; then
  "${PYTHON}" scripts/run_probe_ridge.py \
    --probe-features-dir "${FEATURE_DIR}" \
    --out-dir "${PROBE_DIR}" \
    --alpha 10.0
else
  echo "Skip ridge probes: ${PROBE_DIR}/probe_report.json exists"
fi

if [[ ! -s "${ANALYSIS_DIR}/probe_atlas_report.json" ]]; then
  mkdir -p "${ANALYSIS_DIR}"
  "${PYTHON}" scripts/summarize_probe_atlas.py \
    --probe-scores "${PROBE_DIR}/probe_scores.csv" \
    --out-dir "${ANALYSIS_DIR}" \
    --encoded-threshold 0.1
else
  echo "Skip probe atlas: ${ANALYSIS_DIR}/probe_atlas_report.json exists"
fi

if [[ ! -s "${ANALYSIS_DIR}/linear_task_report.json" ]]; then
  "${PYTHON}" scripts/run_linear_task_benchmarks.py \
    --probe-features-dir "${FEATURE_DIR}" \
    --probe-atlas-dir "${ANALYSIS_DIR}" \
    --out-dir "${ANALYSIS_DIR}" \
    --alpha 1.0
else
  echo "Skip closure/task benchmarks: ${ANALYSIS_DIR}/linear_task_report.json exists"
fi

if [[ ! -s "${ANALYSIS_DIR}/linear_erasure_screen_report.json" ]]; then
  "${PYTHON}" scripts/run_linear_erasure_screen.py \
    --probe-features-dir "${FEATURE_DIR}" \
    --probe-atlas-dir "${ANALYSIS_DIR}" \
    --out-dir "${ANALYSIS_DIR}" \
    --task-alpha 1.0 \
    --concept-alpha 10.0
else
  echo "Skip linear erasure screen: ${ANALYSIS_DIR}/linear_erasure_screen_report.json exists"
fi

echo "Done benchmark resume stage for ${MODEL}"
