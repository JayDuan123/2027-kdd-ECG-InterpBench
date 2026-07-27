#!/usr/bin/env python
"""Select validation-only Pareto operating points for stage-two evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (4311, 4312, 4313)
PRIMARY_SPARSITY_ARM = "fixed_k_over_d"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty selection table")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def pareto_frontier(candidates: pd.DataFrame) -> pd.DataFrame:
    keep = []
    rows = list(candidates.itertuples(index=False))
    for index, point in enumerate(rows):
        dominated = False
        for other_index, other in enumerate(rows):
            if index == other_index:
                continue
            weakly_better = (
                other.validation_recon_R2_mean >= point.validation_recon_R2_mean
                and other.validation_semantic_alignment_mean >= point.validation_semantic_alignment_mean
                and other.validation_dead_fraction_mean <= point.validation_dead_fraction_mean
                and other.expansion_E <= point.expansion_E
            )
            strictly_better = (
                other.validation_recon_R2_mean > point.validation_recon_R2_mean
                or other.validation_semantic_alignment_mean > point.validation_semantic_alignment_mean
                or other.validation_dead_fraction_mean < point.validation_dead_fraction_mean
                or other.expansion_E < point.expansion_E
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        keep.append(not dominated)
    return candidates.loc[keep].copy()


def with_validated_sparsity_arm(
    surface: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    if "sparsity_arm" not in manifest:
        raise RuntimeError("training manifest is missing sparsity_arm")
    manifest_arms = sorted(manifest.sparsity_arm.dropna().astype(str).unique())
    if manifest.sparsity_arm.isna().any() or manifest_arms != [PRIMARY_SPARSITY_ARM]:
        raise RuntimeError(
            f"expected one {PRIMARY_SPARSITY_ARM} manifest arm, found {manifest_arms}"
        )

    validated = surface.copy()
    if "sparsity_arm" not in validated:
        validated["sparsity_arm"] = PRIMARY_SPARSITY_ARM
        return validated

    surface_arms = sorted(validated.sparsity_arm.dropna().astype(str).unique())
    if validated.sparsity_arm.isna().any() or surface_arms != [PRIMARY_SPARSITY_ARM]:
        raise RuntimeError(
            f"surface sparsity arm disagrees with manifest: {surface_arms}"
        )
    return validated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--min-recon-r2", type=float, default=0.90)
    parser.add_argument("--max-dead-fraction", type=float, default=0.20)
    args = parser.parse_args()

    audit = json.loads((args.root / "audit.json").read_text())
    if not audit.get("audit_pass"):
        raise RuntimeError(f"multi-scale audit is incomplete: {audit}")
    surface = pd.read_csv(args.root / "layer_scale_surface.csv")
    manifest = pd.read_csv(args.root / "training_manifest.csv")
    surface = with_validated_sparsity_arm(surface, manifest)
    selected_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    model_diagnostics = {}
    for model, model_surface in surface.groupby("model", sort=True):
        eligible = model_surface[
            (model_surface.validation_recon_R2_mean >= args.min_recon_r2)
            & (model_surface.validation_dead_fraction_mean < args.max_dead_fraction)
        ].copy()
        model_diagnostics[model] = {
            "candidate_cells": int(len(model_surface)),
            "eligible_cells": int(len(eligible)),
        }
        if eligible.empty:
            model_diagnostics[model]["status"] = "no_fidelity_eligible_point"
            continue
        frontier = pareto_frontier(eligible)
        model_diagnostics[model]["pareto_cells"] = int(len(frontier))
        model_diagnostics[model]["status"] = "eligible"
        role_choices = {
            "compact": frontier.sort_values(
                ["expansion_E", "validation_semantic_alignment_mean", "validation_recon_R2_mean"],
                ascending=[True, False, False],
            ).iloc[0],
            "semantic": frontier.sort_values(
                ["validation_semantic_alignment_mean", "validation_recon_R2_mean", "expansion_E"],
                ascending=[False, False, True],
            ).iloc[0],
            "faithful": frontier.sort_values(
                ["validation_recon_R2_mean", "validation_dead_fraction_mean", "expansion_E"],
                ascending=[False, True, True],
            ).iloc[0],
        }
        deduplicated: dict[tuple[int, float], dict[str, Any]] = {}
        for role, point in role_choices.items():
            key = (int(point.layer), float(point.expansion_E))
            if key not in deduplicated:
                deduplicated[key] = {
                    "model": model,
                    "layer": int(point.layer),
                    "relative_depth": float(point.relative_depth),
                    "actual_relative_depth": float(point.actual_relative_depth),
                    "expansion_E": int(point.expansion_E),
                    "sparsity_arm": str(point.sparsity_arm),
                    "validation_recon_R2": float(point.validation_recon_R2_mean),
                    "validation_dead_fraction": float(point.validation_dead_fraction_mean),
                    "validation_semantic_alignment": float(point.validation_semantic_alignment_mean),
                    "validation_concept_coverage_020": float(
                        point.validation_concept_coverage_020_mean
                    ),
                    "roles": [],
                    "selection_split": "validation",
                    "test_metrics_used_for_selection": False,
                }
            deduplicated[key]["roles"].append(role)
        for point in deduplicated.values():
            point["roles"] = "+".join(point["roles"])
            point["selection_index"] = len(selected_rows)
            selected_rows.append(point)
            matches = manifest[
                (manifest.model == model)
                & (manifest.layer == point["layer"])
                & (manifest.expansion_E == point["expansion_E"])
            ].sort_values("seed")
            if tuple(matches.seed.astype(int)) != SEEDS:
                raise RuntimeError(
                    f"incomplete selected checkpoint set for {model}/L{point['layer']}/E{point['expansion_E']}"
                )
            for row in matches.to_dict("records"):
                checkpoint_rows.append(
                    {
                        "selection_index": point["selection_index"],
                        "roles": point["roles"],
                        "model": model,
                        "layer": point["layer"],
                        "relative_depth": point["relative_depth"],
                        "expansion_E": point["expansion_E"],
                        "seed": int(row["seed"]),
                        "N": int(row["N"]),
                        "k": int(row["k"]),
                        "checkpoint": row["checkpoint"],
                        "metrics": row["metrics"],
                        "activation_path": row["activation_path"],
                        "records_path": row["records_path"],
                    }
                )

    if not selected_rows:
        raise RuntimeError("no model has a fidelity-eligible validation operating point")
    write_csv(args.root / "selected_operating_points.csv", selected_rows)
    write_csv(args.root / "selected_checkpoint_manifest.csv", checkpoint_rows)
    metadata = {
        "status": "complete",
        "selection_split": "validation",
        "test_metrics_used_for_selection": False,
        "min_recon_r2": args.min_recon_r2,
        "max_dead_fraction": args.max_dead_fraction,
        "selected_points": len(selected_rows),
        "selected_checkpoints": len(checkpoint_rows),
        "models_with_points": sorted({row["model"] for row in selected_rows}),
        "model_diagnostics": model_diagnostics,
    }
    (args.root / "operating_point_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
