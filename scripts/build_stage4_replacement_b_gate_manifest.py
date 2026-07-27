#!/usr/bin/env python3
"""Build a targeted manifest for Stage IV replacement-control B-gate audits.

These jobs are not part of the recon-band primary SAE aggregate. They are a
small anchor audit for source-clean replacement concepts that currently lack
SAE l0-clamp evidence in the Stage IV orthogonal-control screen.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "results/analysis/model_comparison/orthogonal_concepts/"
    / "stage4_replacement_b_gate_manifest.csv"
)


TARGETS = [
    {
        "cell_index": 25,
        "source_csv": (
            "results/sae_extension/six_model_sae_audit/"
            "phase0_recon_grid_sixart_highN_targeted/cell_25/N128/L032/"
            "sae_recon_curve.csv"
        ),
        "reason": "pr_interval_missing_B_gate",
    },
    {
        "cell_index": 32,
        "source_csv": (
            "results/sae_extension/six_model_sae_audit/"
            "phase0_recon_grid_sixart_highN_targeted/cell_32/N128/L032/"
            "sae_recon_curve.csv"
        ),
        "reason": "p_area_limb_missing_B_gate",
    },
    {
        "cell_index": 33,
        "source_csv": (
            "results/sae_extension/six_model_sae_audit/"
            "phase0_recon_grid_sixart_highN_targeted/cell_33/N128/L032/"
            "sae_recon_curve.csv"
        ),
        "reason": "t_duration_global_missing_B_gate_cardiac_fm_anchor",
    },
    {
        "cell_index": 29,
        "source_csv": (
            "results/sae_extension/six_model_sae_audit/"
            "phase0_recon_grid_tinyN_low_sixart/cell_29/N8/L01/"
            "sae_recon_curve.csv"
        ),
        "reason": "t_duration_global_missing_B_gate_st_mem_anchor",
    },
    {
        "cell_index": 37,
        "source_csv": (
            "results/sae_extension/six_model_sae_audit/"
            "phase0_recon_grid_tinyN_low_sixart/cell_37/N8/L01/"
            "sae_recon_curve.csv"
        ),
        "reason": "p_area_precordial_missing_B_gate",
    },
]


FIELDNAMES = [
    "cell_index",
    "model",
    "concept",
    "task",
    "layer",
    "N_capacity",
    "l0_target",
    "clamp_n_features",
    "recon_R2",
    "source_csv",
    "checkpoint_dir",
    "recon_r2_floor",
    "audit_mode",
    "reason",
]


def read_one(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def main() -> None:
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        rel = Path(str(target["source_csv"]))
        row = read_one(ROOT / rel)
        n_capacity = int(float(row["N_capacity"]))
        l0_target = int(float(row["l0_target"]))
        rows.append(
            {
                "cell_index": target["cell_index"],
                "model": row["model"],
                "concept": row["concept"],
                "task": row["task"],
                "layer": int(float(row["layer"])),
                "N_capacity": n_capacity,
                "l0_target": l0_target,
                "clamp_n_features": l0_target,
                "recon_R2": row["recon_R2"],
                "source_csv": str(rel),
                "checkpoint_dir": str(rel.parent / "checkpoints"),
                "recon_r2_floor": "0.85",
                "audit_mode": "stage4_B_gate_only_floor",
                "reason": target["reason"],
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.name + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(OUT)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
