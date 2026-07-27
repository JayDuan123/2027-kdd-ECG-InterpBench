#!/usr/bin/env bash
#SBATCH -J mimic_src_mat
#SBATCH -p commons
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/mimic_source_benchmark_100k_v1/materialize_%j.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/materialize_%j.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
PYTHON=/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python
if [[ "${PLAN_KIND:-full}" == "smoke" ]]; then
  "${PYTHON}" scripts/materialize_mimic_source_atlas.py \
    --activation-root results/activations_external_full_v1/mimic_source_100k_v1_smoke \
    --out results/mimic_source_benchmark_100k_v1_smoke --max-records 128
  "${PYTHON}" scripts/build_mimic_source_sae_manifest.py \
    --root results/mimic_source_benchmark_100k_v1_smoke \
    --depths 1 --expansions 8 --seeds 4311 --steps 10 --batch-size 32
else
  "${PYTHON}" scripts/materialize_mimic_source_atlas.py
  "${PYTHON}" scripts/build_mimic_source_sae_manifest.py
fi
