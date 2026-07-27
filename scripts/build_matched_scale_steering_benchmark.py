#!/usr/bin/env python
"""Create an isolated steering benchmark rooted at matched-scale SAE checkpoints."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
SCALE = ROOT / "results/sae_reconciliation/matched_scale_v1"
OUT = SCALE / "steering"


def replace_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(source)


def main() -> None:
    profile = pd.read_csv(SCALE / "matched_scale_model_profile.csv")
    eligible_models = profile.loc[profile.matched_scale_primary_eligible, "model"].tolist()
    if not eligible_models:
        raise RuntimeError("No model passed the matched-scale fidelity audit")

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.csv", "target_registry.csv", "candidate_target_registry.csv"):
        shutil.copy2(SOURCE / name, OUT / name)

    cells = pd.read_csv(SOURCE / "eligible_steering_cells.csv")
    cells = cells[cells.model.isin(eligible_models)].copy().reset_index(drop=True)
    cells["task_index"] = range(len(cells))
    cells.to_csv(OUT / "eligible_steering_cells.csv", index=False)

    gate = pd.read_csv(SOURCE / "model_target_gate.csv")
    gate[gate.model.isin(eligible_models)].to_csv(OUT / "model_target_gate.csv", index=False)

    training = pd.read_csv(SCALE / "training_manifest.csv")
    audit = pd.read_csv(SCALE / "matched_scale_fidelity_audit.csv")
    seed4311 = training[training.seed.eq(4311)].set_index("model")
    rows = []
    for model in eligible_models:
        train = seed4311.loc[model]
        model_audit = audit[audit.model.eq(model)]
        rows.append(
            {
                "model": model,
                "feature_suffix": train.feature_suffix,
                "d_hidden": int(train.d_hidden),
                "N": int(train.N),
                "k": int(train.k),
                "recon_R2": float(model_audit.recon_R2.mean()),
                "dead_fraction": float(model_audit.dead_fraction.mean()),
                "checkpoint_seed4311": str(train.checkpoint),
                # Legacy summary field: True means the frozen fidelity gate passed.
                "in_band": True,
                "status": "matched_scale_primary",
            }
        )
        safe = model.lower().replace("-", "_")
        source_model = SOURCE / "models" / safe
        out_model = OUT / "models" / safe
        replace_symlink(source_model / "frozen_heads.joblib", out_model / "frozen_heads.joblib")
        replace_symlink(source_model / "frozen_heads.metrics.json", out_model / "frozen_heads.metrics.json")

    pd.DataFrame(rows).to_csv(OUT / "selected_operating_points.csv", index=False)
    (OUT / "README.md").write_text(
        "# Matched-scale SAE steering benchmark\n\n"
        "Primary models pass all three fidelity gates in every seed: reconstruction "
        "R2 >= 0.90, dead fraction < 0.20, and median frozen-readout retention >= 0.95.\n\n"
        "All SAEs use E=N/d=8, k/d=1/8, and k/N=1/64. Results are independent "
        "from the earlier reconstruction-matched benchmark.\n"
    )
    print(f"models={eligible_models}")
    print(f"eligible_cells={len(cells)} array_tasks={len(cells) * 3}")


if __name__ == "__main__":
    main()
