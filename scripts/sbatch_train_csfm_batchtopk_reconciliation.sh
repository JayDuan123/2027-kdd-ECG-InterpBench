#!/bin/bash
#SBATCH -J csfm_btk8192
#SBATCH -p scavenge
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 00:59:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH -o logs/sae_reconciliation/train_%j.out
#SBATCH -e logs/sae_reconciliation/train_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/train_csfm_batchtopk_reconciliation.py \
  --n-features 8192 --k 128 --steps 8000 --batch-size 256 --checkpoint-every 100 --device cuda
