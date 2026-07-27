#!/bin/bash
#SBATCH -J pheno_repair
#SBATCH -p scavenge
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 00:59:00
#SBATCH -o logs/sae_reconciliation/pheno_repair_%j.out
#SBATCH -e logs/sae_reconciliation/pheno_repair_%j.err
set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
export OMP_NUM_THREADS=1
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/repair_nonconverged_phenotype_heads.py
