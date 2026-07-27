#!/usr/bin/env bash
set -euo pipefail

MODEL_KEY="$1"
SUFFIX="$2"
CONCEPT_ID="$3"
TASK_ID="$4"
LAYER="$5"

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
case "${MODEL_KEY}" in
  csfm|st_mem)
    PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
    ;;
  ecg_jepa)
    PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/bin/python"
    ;;
  ecg_fm|hubert_ecg|cardiac_fm)
    PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"
    ;;
  *)
    echo "Unknown model key: ${MODEL_KEY}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
mkdir -p "results/analysis/${SUFFIX}"

OUT_JSON="results/analysis/${SUFFIX}/continuation_erase_${CONCEPT_ID}_to_${TASK_ID}_layer$(printf '%02d' "${LAYER}").json"
if [[ -s "${OUT_JSON}" ]]; then
  if "${PYTHON}" - "${OUT_JSON}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
required = {"delta_auroc_minus_random_ci_low", "residual_probe_r2", "eraser_effective_flag"}
raise SystemExit(0 if required.issubset(data) and data.get("eraser_method") == "leace" else 1)
PY
  then
    echo "Skip existing bootstrapped continuation result with residual probe: ${OUT_JSON}"
    exit 0
  fi
  echo "Existing result lacks bootstrap CI or residual probe; recomputing: ${OUT_JSON}"
fi

if [[ "${MODEL_KEY}" == "csfm" ]]; then
  "${PYTHON}" scripts/run_csfm_continuation_erase.py \
    --activation-index-dir "results/activation_index/${SUFFIX}" \
    --probe-features-dir "results/probe_features/${SUFFIX}" \
    --concept-id "${CONCEPT_ID}" \
    --task-id "${TASK_ID}" \
    --layer "${LAYER}" \
    --out-dir "results/analysis/${SUFFIX}" \
    --device cuda \
    --bootstrap-samples 1000
else
  "${PYTHON}" scripts/run_transformer_continuation_erase.py \
    --model "${MODEL_KEY}" \
    --activation-index-dir "results/activation_index/${SUFFIX}" \
    --probe-features-dir "results/probe_features/${SUFFIX}" \
    --concept-id "${CONCEPT_ID}" \
    --task-id "${TASK_ID}" \
    --layer "${LAYER}" \
    --out-dir "results/analysis/${SUFFIX}" \
    --device cuda \
    --bootstrap-samples 1000
fi
