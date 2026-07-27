#!/usr/bin/env python
"""Paired audit of reconstruction-matched versus scale-matched SAE steering."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
NEW = ROOT / "results/sae_reconciliation/matched_scale_v1/steering"
OUT = NEW / "summary/scale_comparison"


def markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        vals = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.4f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    old = pd.read_csv(OLD / "summary/multimodel_steering_cells.csv")
    new = pd.read_csv(NEW / "summary/multimodel_steering_cells.csv")
    keys = ["model", "target", "seed"]
    metrics = [
        "sae_readout_retention",
        "tier1_excess_attribution",
        "excess_selectivity",
        "wbi_improvement",
        "wrong_atom_margin",
        "behavior_excess",
    ]
    passes = ["tier1_sparse_attribution", "tier2_selective_steering", "tier3_behavior_changing"]
    keep = keys + metrics + passes
    paired = old[keep].merge(new[keep], on=keys, suffixes=("_recon_matched", "_scale_matched"), validate="one_to_one")
    expected = len(new)
    if len(paired) != expected:
        raise RuntimeError(f"Only {len(paired)}/{expected} scale-matched cells have an old paired result")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"{metric}_scale_matched"] - paired[f"{metric}_recon_matched"]
    OUT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUT / "paired_seed_cells.csv", index=False)

    rows = []
    for model, group in paired.groupby("model"):
        row = {"model": model, "paired_cells": len(group)}
        for name in passes:
            old_pass = group[f"{name}_recon_matched"].astype(bool)
            new_pass = group[f"{name}_scale_matched"].astype(bool)
            short = name.split("_")[0]
            row[f"{short}_old"] = int(old_pass.sum())
            row[f"{short}_new"] = int(new_pass.sum())
            row[f"{short}_retained"] = int((old_pass & new_pass).sum())
            row[f"{short}_lost"] = int((old_pass & ~new_pass).sum())
            row[f"{short}_gained"] = int((~old_pass & new_pass).sum())
        row["retention_delta_median"] = float(group.delta_sae_readout_retention.median())
        row["selectivity_delta_median"] = float(group.delta_excess_selectivity.median())
        rows.append(row)
    model = pd.DataFrame(rows)
    model.to_csv(OUT / "paired_model_profile.csv", index=False)

    old_profile = pd.read_csv(OLD / "summary/multimodel_target_profile.csv")
    new_profile = pd.read_csv(NEW / "summary/multimodel_target_profile.csv")
    robust = old_profile[["model", "target", "tier2_pass"]].merge(
        new_profile[["model", "target", "tier2_pass"]], on=["model", "target"],
        suffixes=("_recon_matched", "_scale_matched"), validate="one_to_one"
    )
    robust["robust_recon_matched"] = robust.tier2_pass_recon_matched.eq(3)
    robust["robust_scale_matched"] = robust.tier2_pass_scale_matched.eq(3)
    robust.to_csv(OUT / "paired_target_robustness.csv", index=False)

    old_ops = pd.read_csv(OLD / "selected_operating_points.csv")
    new_ops = pd.read_csv(NEW / "selected_operating_points.csv")
    scale = old_ops[["model", "d_hidden", "N", "k", "recon_R2", "dead_fraction"]].merge(
        new_ops[["model", "d_hidden", "N", "k", "recon_R2", "dead_fraction"]],
        on=["model", "d_hidden"], suffixes=("_recon_matched", "_scale_matched"), validate="one_to_one"
    )
    scale.to_csv(OUT / "operating_point_comparison.csv", index=False)

    robust_counts = robust.groupby("model").agg(
        targets=("target", "size"),
        robust_recon_matched=("robust_recon_matched", "sum"),
        robust_scale_matched=("robust_scale_matched", "sum"),
    ).reset_index()
    lines = [
        "# Matched-Scale SAE Steering Sensitivity",
        "",
        f"- Paired unit: identical `(model, target, seed)`; paired cells: {len(paired)}.",
        "- This is a sensitivity audit, not a model leaderboard.",
        "- The scale-matched arm fixes `N/d=8`, `k/d=1/8`, and `k/N=1/64`.",
        "- Primary interpretation depends on effects and pass/fail conclusions that persist across both SAE operating-point protocols.",
        "",
        "## Operating points",
        "",
        markdown(scale),
        "",
        "## Seed-level pass transitions",
        "",
        markdown(model),
        "",
        "## Robust 3-of-3 Tier-2 targets",
        "",
        markdown(robust_counts),
        "",
        "A result present only under one capacity protocol is operating-point-sensitive and must not be presented as a stable model property.",
    ]
    (OUT / "matched_scale_sensitivity_report.md").write_text("\n".join(lines) + "\n")
    print(model.to_string(index=False))
    print(robust_counts.to_string(index=False))


if __name__ == "__main__":
    main()
