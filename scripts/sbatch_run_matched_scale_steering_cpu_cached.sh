#!/bin/bash
#SBATCH -J mscale_cpu
#SBATCH -p scavenge
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH --array=153-317%64
#SBATCH --requeue
#SBATCH -o logs/sae_reconciliation/mscale_cpu_%A_%a.out
#SBATCH -e logs/sae_reconciliation/mscale_cpu_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
BASE=results/sae_reconciliation/matched_scale_v1/steering
SCALE=results/sae_reconciliation/matched_scale_v1
CELL_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEEDS=(4311 4312 4313)
SEED=${SEEDS[$SEED_IDX]}
read -r MODEL TARGET < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; r=pd.read_csv(sys.argv[1]).iloc[int(sys.argv[2])]; print(r.model,r.target)' \
  "${BASE}/eligible_steering_cells.csv" "${CELL_IDX}")
read -r SUFFIX N K < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
  'import pandas as pd,sys; d=pd.read_csv(sys.argv[1]); r=d[d.model==sys.argv[2]].iloc[0]; print(r.feature_suffix,int(r.N),int(r.k))' \
  "${BASE}/selected_operating_points.csv" "${MODEL}")
SAFE=${MODEL//-/_}
SAFE=${SAFE,,}
CACHE="${BASE}/models/${SAFE}/shared_cache/seed${SEED}_N${N}_k${K}/complete.json"
[[ -f "${CACHE}" ]] || { echo "missing prebuilt cache: ${CACHE}" >&2; exit 2; }
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_steering_benchmark_task.py \
  --model "${MODEL}" --target "${TARGET}" --seed "${SEED}" \
  --checkpoint "${SCALE}/models/${SAFE}/checkpoints/seed${SEED}/batchtopk_N${N}_k${K}.pt" \
  --acts "results/probe_features/${SUFFIX}/pooled.npy" \
  --manifest "${BASE}/manifest.csv" --registry "${BASE}/candidate_target_registry.csv" \
  --heads "${BASE}/models/${SAFE}/frozen_heads.joblib" \
  --out-dir "${BASE}/models/${SAFE}/tasks" --device cpu
