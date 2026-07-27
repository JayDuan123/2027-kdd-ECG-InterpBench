#!/bin/bash
#SBATCH -J sae_topk_v2
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH -t 00:59:00
#SBATCH --array=0-93%32
#SBATCH --requeue
#SBATCH -o logs/sae_extension/topk_v2_%A_%a.out
#SBATCH -e logs/sae_extension/topk_v2_%A_%a.err

set -euo pipefail

cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_extension results/sae_extension/six_model_sae_audit/topk_group_steering_v2/runs

PYTHON="/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"
MANIFEST="results/sae_extension/six_model_sae_audit/topk_group_steering_v2/topk_v2_manifest.csv"

"${PYTHON}" scripts/run_sae_topk_v2_task.py \
  --manifest "${MANIFEST}" \
  --task-index "${SLURM_ARRAY_TASK_ID}" \
  --out-root results/sae_extension/six_model_sae_audit/topk_group_steering_v2/runs \
  --python "${PYTHON}" \
  --n-random 20 \
  --bootstrap-samples 1000 \
  --device cuda
