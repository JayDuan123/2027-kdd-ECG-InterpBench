#!/usr/bin/env python
"""Freeze the six-model matched-capacity and matched-sparsity SAE protocol."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
OUT = ROOT / "results/sae_reconciliation/matched_scale_v1"
EXPANSION = 8
K_OVER_D = 1 / 8
SEEDS = (4311, 4312, 4313)


def main() -> None:
    selected = pd.read_csv(SOURCE / "selected_operating_points.csv")
    rows = []
    for model_index, row in selected.reset_index(drop=True).iterrows():
        d = int(row.d_hidden)
        n_features = int(EXPANSION * d)
        k = int(round(K_OVER_D * d))
        safe = str(row.model).lower().replace("-", "_")
        for seed in SEEDS:
            checkpoint = (
                OUT
                / "models"
                / safe
                / "checkpoints"
                / f"seed{seed}"
                / f"batchtopk_N{n_features}_k{k}.pt"
            )
            rows.append(
                {
                    "task_index": len(rows),
                    "model_index": model_index,
                    "model": row.model,
                    "model_safe": safe,
                    "feature_suffix": row.feature_suffix,
                    "d_hidden": d,
                    "expansion_E": EXPANSION,
                    "N": n_features,
                    "k_over_d": K_OVER_D,
                    "k_over_N": k / n_features,
                    "k": k,
                    "seed": seed,
                    "steps": 8000,
                    "batch_size": 256,
                    "learning_rate": 3e-4,
                    "checkpoint": str(checkpoint),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 18:
        raise RuntimeError(f"Expected 18 matched-scale cells, got {len(frame)}")
    if not frame.groupby("d_hidden").expansion_E.nunique().eq(1).all():
        raise RuntimeError("Expansion ratio is not frozen")
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "training_manifest.csv", index=False)
    protocol = [
        "# Matched-Scale SAE Protocol v1",
        "",
        "- Anchor activations: PTB-XL train split pooled representations.",
        "- Expansion ratio: `N/d = 8` for every model.",
        "- Relative sparsity: `k/d = 1/8` for every model.",
        "- Active dictionary fraction: `k/N = 1/64` for every model.",
        "- Seeds: 4311, 4312, 4313.",
        "- Training: 8,000 steps, batch size 256, Adam lr 3e-4.",
        "- Normalization: PTB-XL train-only per-dimension mean/std, frozen externally.",
        "- Fidelity gates: recon R2 >= 0.90, dead fraction < 0.20, median readout retention >= 0.95.",
        "- No model-specific retuning is allowed in the primary matched-scale arm.",
    ]
    (OUT / "protocol.md").write_text("\n".join(protocol) + "\n")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
