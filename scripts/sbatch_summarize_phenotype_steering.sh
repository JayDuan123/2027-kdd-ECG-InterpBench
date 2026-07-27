#!/bin/bash
#SBATCH -J pheno_summary
#SBATCH -p scavenge
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 00:59:00
#SBATCH -o logs/sae_reconciliation/pheno_summary_%j.out
#SBATCH -e logs/sae_reconciliation/pheno_summary_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
mkdir -p logs/sae_reconciliation
export OMP_NUM_THREADS=1
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/summarize_phenotype_steering.py --bootstrap 2000 --jobs 15
