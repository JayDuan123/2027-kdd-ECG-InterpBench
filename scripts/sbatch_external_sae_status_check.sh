#!/usr/bin/env bash
#SBATCH --job-name=extsaechk
#SBATCH --partition=scavenge
#SBATCH --output=logs/external_sae_status_check_%j.out
#SBATCH --error=logs/external_sae_status_check_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -euo pipefail

PROJECT_ROOT="/rhf/allocations/wq8/yd68/ecg_fm_interpretability_benchmark"
PYTHON="/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"
WATCH_JOB_ID="${WATCH_JOB_ID:-}"
WATCH_ACTIVATION_ROOT="${WATCH_ACTIVATION_ROOT:-}"
WATCH_EXPECTED_SHARDS="${WATCH_EXPECTED_SHARDS:-}"
STATUS_LOG="${STATUS_LOG:-${PROJECT_ROOT}/results/multicohort/external_sae/hourly_status.log}"

cd "${PROJECT_ROOT}"
mkdir -p logs "$(dirname "${STATUS_LOG}")"

{
  echo "===== $(date -Is) ====="
  if [[ -n "${WATCH_JOB_ID}" ]]; then
    echo "[squeue]"
    squeue -j "${WATCH_JOB_ID}" || true
    echo "[sacct]"
    sacct -j "${WATCH_JOB_ID}" --format=JobID,JobName%20,State,ExitCode,Elapsed -P || true
  fi
  if [[ -n "${WATCH_ACTIVATION_ROOT}" ]]; then
    echo "[activation_progress]"
    WATCH_ACTIVATION_ROOT="${WATCH_ACTIVATION_ROOT}" WATCH_EXPECTED_SHARDS="${WATCH_EXPECTED_SHARDS}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["WATCH_ACTIVATION_ROOT"])
expected = os.environ.get("WATCH_EXPECTED_SHARDS", "")
metadata_count = sum(1 for _ in root.glob("*/activation_metadata.json")) if root.exists() else 0
records = root / "records.csv"
shards = root / "shards.csv"
print("root", root)
print("metadata_shards", metadata_count)
if expected:
    print("expected_shards", expected)
print("records_exists", records.exists())
print("shards_exists", shards.exists())
if records.exists():
    print("records_lines", max(sum(1 for _ in records.open()) - 1, 0))
if shards.exists():
    print("shards_lines", max(sum(1 for _ in shards.open()) - 1, 0))
PY
  fi
  echo "[external_sae_recon_gate]"
  "${PYTHON}" - <<'PY'
import csv
from collections import Counter
from pathlib import Path
p=Path("results/multicohort/external_sae/external_sae_recon_gate.csv")
if not p.exists():
    print("missing external_sae_recon_gate.csv")
else:
    rows=list(csv.DictReader(p.open()))
    print("rows", len(rows))
    print("status", dict(Counter(r["recon_gate_status"] for r in rows)))
    print("passes", sum(r["recon_gate_pass"] == "true" for r in rows))
    print("available_activation_rows", sum(r["external_activation_status"] == "available" for r in rows))
PY
  echo
} >> "${STATUS_LOG}"

echo "wrote ${STATUS_LOG}"
