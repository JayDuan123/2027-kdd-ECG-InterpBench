#!/bin/bash
#SBATCH -J mm_steer
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --array=0-251%24
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/mm_steer_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mm_steer_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
TARGETS=(lbbb rbbb pvc avb1 lafb afib hr_ventricular qrs_duration qtc_fridericia st_amp_global qrst_angle age sex baseline_drift_present)
SEEDS=(4311 4312 4313)
PER_MODEL=42; MODEL_IDX=$((SLURM_ARRAY_TASK_ID / PER_MODEL)); LOCAL=$((SLURM_ARRAY_TASK_ID % PER_MODEL))
SEED_IDX=$((LOCAL / 14)); TARGET_IDX=$((LOCAL % 14)); SEED=${SEEDS[$SEED_IDX]}; TARGET=${TARGETS[$TARGET_IDX]}
read -r MODEL SUFFIX N K < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv("results/sae_reconciliation/steering_benchmark_multimodel_v1/selected_operating_points.csv").iloc[int(sys.argv[1])]; print(r.model,r.feature_suffix,int(r.N),int(r.k))' "${MODEL_IDX}")
SAFE=${MODEL//-/_}; SAFE=${SAFE,,}
BASE=results/sae_reconciliation/steering_benchmark_multimodel_v1
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_steering_benchmark_task.py \
  --model "${MODEL}" --target "${TARGET}" --seed "${SEED}" \
  --checkpoint "${BASE}/models/${SAFE}/checkpoints/seed${SEED}/batchtopk_N${N}_k${K}.pt" \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest "${BASE}/manifest.csv" --registry "${BASE}/target_registry.csv" \
  --heads "${BASE}/models/${SAFE}/frozen_heads.joblib" \
  --out-dir "${BASE}/models/${SAFE}/tasks" --device cuda
