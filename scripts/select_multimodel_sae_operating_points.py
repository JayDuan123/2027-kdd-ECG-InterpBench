#!/usr/bin/env python
"""Select one frozen SAE operating point per model from Phase 0 metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"


def main() -> None:
    gate = pd.read_csv(BASE / "model_gate.csv")
    rows = []
    for model_row in gate.itertuples(index=False):
        safe = model_row.model.lower().replace("-", "_")
        candidates = []
        for path in sorted((BASE / "models" / safe / "checkpoints" / "seed4311").glob("*.metrics.json")):
            metric = json.loads(path.read_text())
            checkpoint = path.with_name(path.name.removesuffix(".metrics.json"))
            candidates.append({"model": model_row.model, "feature_suffix": model_row.feature_suffix,
                               "d_hidden": model_row.d_hidden, "N": int(metric["N"]), "k": int(metric["k"]),
                               "recon_R2": float(metric["explained_variance"]), "dead_fraction": float(metric["dead_fraction"]),
                               "checkpoint_seed4311": str(checkpoint),
                               "in_band": 0.95 <= float(metric["explained_variance"]) <= 0.98 and float(metric["dead_fraction"]) < 0.20})
        eligible = [c for c in candidates if c["in_band"]]
        if eligible:
            eligible.sort(key=lambda c: (c["N"], c["dead_fraction"], abs(c["recon_R2"] - 0.965)))
            selected = eligible[0]; selected["status"] = "in_band"
        elif candidates:
            candidates.sort(key=lambda c: (abs(c["recon_R2"] - 0.965), c["dead_fraction"]))
            selected = candidates[0]; selected["status"] = "no_matched_point"
        else:
            selected = {"model": model_row.model, "feature_suffix": model_row.feature_suffix, "d_hidden": model_row.d_hidden,
                        "N": "", "k": "", "recon_R2": "", "dead_fraction": "", "checkpoint_seed4311": "",
                        "in_band": False, "status": "missing_phase0"}
        rows.append(selected)
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "selected_operating_points.csv", index=False)
    print(out.to_string(index=False))
    if not out.in_band.all():
        raise SystemExit("At least one model has no in-band operating point")


if __name__ == "__main__":
    main()
