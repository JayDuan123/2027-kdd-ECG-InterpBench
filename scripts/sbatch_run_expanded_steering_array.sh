#!/bin/bash
#SBATCH -J exp_steer
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --array=0-0%24
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/exp_steer_%A_%a.out
#SBATCH -e logs/sae_reconciliation/exp_steer_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
CELL_IDX=$((SLURM_ARRAY_TASK_ID / 3)); SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3)); SEEDS=(4311 4312 4313); SEED=${SEEDS[$SEED_IDX]}
BASE=results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded
read -r MODEL TARGET < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv("results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded/eligible_steering_cells.csv").iloc[int(sys.argv[1])]; print(r.model,r.target)' "${CELL_IDX}")
read -r SUFFIX N K < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; d=pd.read_csv("results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded/selected_operating_points.csv"); r=d[d.model==sys.argv[1]].iloc[0]; print(r.feature_suffix,int(r.N),int(r.k))' "${MODEL}")
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}; SAEBASE=results/sae_reconciliation/steering_benchmark_multimodel_v1/models/${SAFE}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_steering_benchmark_task.py \
  --model "${MODEL}" --target "${TARGET}" --seed "${SEED}" \
  --checkpoint "${SAEBASE}/checkpoints/seed${SEED}/batchtopk_N${N}_k${K}.pt" \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest "${BASE}/manifest.csv" --registry "${BASE}/candidate_target_registry.csv" \
  --heads "${BASE}/models/${SAFE}/frozen_heads.joblib" --out-dir "${BASE}/models/${SAFE}/tasks" --device cuda
