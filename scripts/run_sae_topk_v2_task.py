#!/usr/bin/env python
"""Execute one top-k SAE group-clamp task from the frozen v2 manifest."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd


MODEL_PYTHON = {
    "CSFM": "/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python",
    "ECG-JEPA": "/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/bin/python",
}
DEFAULT_TRANSFORMER_PYTHON = "/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    match = manifest[manifest["task_index"] == args.task_index]
    if len(match) != 1:
        raise ValueError(f"task_index={args.task_index} matched {len(match)} rows")
    row = match.iloc[0]
    if row["group_status"] != "eligible" or not bool(row["checkpoint_exists"]):
        raise RuntimeError(f"task is not runnable: {row.to_dict()}")

    model_slug = str(row["model"]).lower().replace("-", "_")
    out_dir = args.out_root / (
        f"task_{args.task_index:03d}_{model_slug}_cell_{int(row['cell_index'])}_k{int(row['group_k']):02d}"
    )
    output_csv = out_dir / "sae_layer_per_cell.csv"
    if output_csv.exists() and output_csv.stat().st_size > 0:
        print(f"skip existing {output_csv}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    inner_python = MODEL_PYTHON.get(str(row["model"]), DEFAULT_TRANSFORMER_PYTHON)

    cmd = [
        inner_python,
        "-m",
        "benchmark_v1.sae_extension.run_sae_layer",
        "--environment",
        str(row["environment"]),
        "--cells",
        "results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv",
        "--coupling",
        "results/analysis/model_comparison/leace_coupling_risk_summary.csv",
        "--artifacts",
        "results/sae_artifacts_six_model",
        "--out",
        str(out_dir),
        "--cell-index",
        str(int(row["cell_index"])),
        "--selection-mode",
        "recon_band",
        "--recon-target",
        "0.90",
        "--recon-band-width",
        "0.02",
        "--relaxed-band-width",
        "0.04",
        "--E-grid",
        "1",
        "--n-features-grid",
        str(int(row["N_capacity"])),
        "--l0-grid",
        str(int(row["l0_target"])),
        "--require-matched-tier",
        "in_band",
        "--feature-ranking",
        "concept",
        "--n-features",
        str(int(row["group_k"])),
        "--n-random",
        str(args.n_random),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--selectivity-mode",
        "endpoint",
        "--steps",
        "4000",
        "--checkpoint-dir",
        str(row["checkpoint_dir"]),
        "--checkpoint-every",
        "250",
        "--skip-existing",
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
