#!/bin/bash
#SBATCH -J ext_local_sae
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/external_benchmark/local_sae_%A_%a.out
#SBATCH -e logs/external_benchmark/local_sae_%A_%a.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
ROW=${SLURM_ARRAY_TASK_ID}
read -r ACTS MANIFEST SEED N K OUT < <(/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python -c \
 'import pandas as pd,sys; r=pd.read_csv("results/external_benchmark_v1/cohort_adapted_sae_manifest.csv").iloc[int(sys.argv[1])]; print(r.acts,r.manifest,int(r.seed),int(r.N),int(r.k),r.checkpoint)' "${ROW}")
METRICS="${OUT%.pt}.metrics.json"
if [[ -s "${OUT}" && -s "${METRICS}" ]]; then
  echo "Cohort-adapted SAE already complete: ${OUT}"
  exit 0
fi
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
 --acts "${ACTS}" --manifest "${MANIFEST}" --out "${OUT}" --n-features "${N}" --k "${K}" \
 --steps 8000 --batch-size 256 --lr 3e-4 --checkpoint-every 250 --seed "${SEED}" --device cuda
