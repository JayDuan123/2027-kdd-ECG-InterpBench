#!/usr/bin/env python
"""Build data-driven family and cross-family joint steering groups."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--max-family-members", type=int, default=5)
    parser.add_argument("--max-cross-family-members", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cells = pd.read_csv(args.base / "summary/multimodel_steering_cells.csv")
    required = cells[
        cells.analysis_role.eq("main")
        & cells.headline_eligible.astype(bool)
        & ~cells.readout_quality_warning.astype(bool)
    ].copy()
    profile = required.groupby(["model", "target", "family"], as_index=False).agg(
        seeds=("seed", "nunique"),
        tier1_pass=("tier1_sparse_attribution", "sum"),
        tier2_pass=("tier2_selective_steering", "sum"),
        excess_selectivity_mean=("excess_selectivity", "mean"),
        wbi_improvement_mean=("wbi_improvement", "mean"),
    )
    robust = profile[(profile.seeds.eq(3)) & (profile.tier2_pass.eq(3))].copy()
    rows = []
    for (model, family), group in robust.groupby(["model", "family"]):
        group = group.sort_values(
            ["excess_selectivity_mean", "wbi_improvement_mean"], ascending=False
        ).head(args.max_family_members)
        if len(group) < 2:
            continue
        members = group.target.tolist()
        rows.append(
            {
                "model": model,
                "group_id": f"family__{family}",
                "group_type": "family_joint",
                "family_scope": family,
                "members_json": json.dumps(members),
                "member_count": len(members),
                "selection_rule": "main_headline_quality_qualified_tier2_3of3",
            }
        )
    for model, group in robust.groupby("model"):
        representatives = (
            group.sort_values(
                ["family", "excess_selectivity_mean", "wbi_improvement_mean"],
                ascending=[True, False, False],
            )
            .groupby("family", as_index=False)
            .head(1)
            .sort_values(["excess_selectivity_mean", "wbi_improvement_mean"], ascending=False)
            .head(args.max_cross_family_members)
        )
        if len(representatives) < 3:
            continue
        members = representatives.target.tolist()
        rows.append(
            {
                "model": model,
                "group_id": "cross_family__top_robust",
                "group_type": "cross_family_joint",
                "family_scope": "+".join(representatives.family.tolist()),
                "members_json": json.dumps(members),
                "member_count": len(members),
                "selection_rule": "one_best_tier2_3of3_target_per_family",
            }
        )
    groups = pd.DataFrame(rows)
    if groups.empty:
        raise RuntimeError("No eligible robust joint groups were found")
    groups = groups.sort_values(["model", "group_type", "group_id"]).reset_index(drop=True)
    groups.insert(0, "group_index", range(len(groups)))
    out = args.base / "joint_steering"
    out.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out / "joint_steering_manifest.csv", index=False)
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
