#!/usr/bin/env bash
#SBATCH -J mimic_src_sae
#SBATCH -p commons
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 06:00:00
#SBATCH --array=0-449%36
#SBATCH --requeue
#SBATCH --signal=B:USR1@90
#SBATCH --open-mode=append
#SBATCH -o logs/mimic_source_benchmark_100k_v1/sae_%A_%a.out
#SBATCH -e logs/mimic_source_benchmark_100k_v1/sae_%A_%a.err

set -euo pipefail
cd /rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark
/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python scripts/run_multiscale_sae_task.py \
  --manifest results/mimic_source_benchmark_100k_v1/training_manifest.csv \
  --task-index "${SLURM_ARRAY_TASK_ID}" --device cuda --checkpoint-every 100 \
  --semantic-train-limit 4096 \
  --concepts results/mimic_source_benchmark_100k_v1/derived/concepts_matrix.csv \
  --concept-registry results/mimic_source_benchmark_100k_v1/derived/concept_registry.csv \
  --preserve-missing-concepts --complete-case-evaluation
