#!/usr/bin/env python
"""Verify fixed operating-point fidelity and dictionary health across SAE seeds."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"


def main() -> None:
    selected = pd.read_csv(BASE / "selected_operating_points.csv")
    rows = []
    for op in selected.itertuples(index=False):
        safe = op.model.lower().replace("-", "_")
        for seed in (4311, 4312, 4313):
            path = BASE / "models" / safe / "checkpoints" / f"seed{seed}" / f"batchtopk_N{int(op.N)}_k{int(op.k)}.metrics.json"
            if not path.exists():
                rows.append({"model": op.model, "seed": seed, "N": op.N, "k": op.k, "status": "missing"}); continue
            metric = json.loads(path.read_text()); recon = float(metric["explained_variance"]); dead = float(metric["dead_fraction"])
            rows.append({"model": op.model, "seed": seed, "N": int(op.N), "k": int(op.k), "recon_R2": recon,
                         "dead_fraction": dead, "recon_in_band": 0.95 <= recon <= 0.98,
                         "dead_pass": dead < 0.20, "status": "pass" if 0.95 <= recon <= 0.98 and dead < .20 else "quality_warning"})
    out = pd.DataFrame(rows); out.to_csv(BASE / "sae_seed_fidelity_audit.csv", index=False); print(out.to_string(index=False))
    if (out.status == "missing").any(): raise SystemExit("Missing selected SAE seed")


if __name__ == "__main__": main()
