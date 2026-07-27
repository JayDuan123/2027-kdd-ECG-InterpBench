#!/bin/bash
#SBATCH -J mssae_paper_build
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o logs/multiscale_sae_v1/paper_build_%j.out
#SBATCH -e logs/multiscale_sae_v1/paper_build_%j.err

set -euo pipefail

PAPER=/rhf/allocations/wq8/yd68/overleaf_paper_benchmark
BENCHMARK=/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark

module load GCC/13.2.0 texlive/20230313
cd "$PAPER"
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null

cd "$BENCHMARK"
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python \
  scripts/audit_multiscale_paper_build.py
