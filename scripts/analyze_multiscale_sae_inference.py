#!/usr/bin/env python
"""Paired layer/seed inference for multi-scale ECG-FM profiles."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["CARDIAC-FM", "CSFM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]
EXPECTED_EXPANSIONS = (1, 4, 8, 16, 32)
EXPECTED_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_SEEDS = (4311, 4312, 4313)


def normalized_log_auc(expansion: np.ndarray, values: np.ndarray) -> float:
    order = np.argsort(expansion)
    x = np.log(expansion[order].astype(float))
    y = values[order].astype(float)
    return float(np.trapz(y, x=x) / (x[-1] - x[0]))


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty inference table")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def build_auc_units(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for (model, depth, seed), part in cells.groupby(["model", "relative_depth", "seed"], sort=True):
        if part.expansion_E.nunique() != 5:
            raise RuntimeError(f"incomplete expansion grid for {model}/{depth}/{seed}")
        rows.append(
            {
                "model": model,
                "relative_depth": float(depth),
                "seed": int(seed),
                "value": normalized_log_auc(part.expansion_E.to_numpy(), part[metric].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def validate_matched_scale_cells(cells: pd.DataFrame) -> int:
    """Reject incomplete or asymmetric FM support before any ranking is computed."""
    expected_keys = {
        (model, depth, expansion, seed)
        for model in MODEL_ORDER
        for depth in EXPECTED_DEPTHS
        for expansion in EXPECTED_EXPANSIONS
        for seed in EXPECTED_SEEDS
    }
    observed_keys = {
        (str(row.model), float(row.relative_depth), int(row.expansion_E), int(row.seed))
        for row in cells.itertuples(index=False)
    }
    if len(cells) != len(expected_keys) or observed_keys != expected_keys:
        raise RuntimeError(
            "inference requires the complete matched-scale FM grid: "
            f"rows={len(cells)}, unique={len(observed_keys)}, expected={len(expected_keys)}"
        )
    for (depth, expansion, seed), part in cells.groupby(
        ["relative_depth", "expansion_E", "seed"], sort=True
    ):
        if set(part.model) != set(MODEL_ORDER) or len(part) != len(MODEL_ORDER):
            raise RuntimeError(
                f"unmatched FM block at depth={depth}, E={expansion}, seed={seed}"
            )
    return len(EXPECTED_DEPTHS) * len(EXPECTED_EXPANSIONS) * len(EXPECTED_SEEDS)


def bootstrap_indices(depths: np.ndarray, seeds: np.ndarray, rng: np.random.Generator, n_boot: int):
    for _ in range(n_boot):
        yield rng.choice(depths, size=len(depths), replace=True), rng.choice(
            seeds, size=len(seeds), replace=True
        )


def bootstrap_profile(
    units: pd.DataFrame,
    model: str,
    draws: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    lookup = units[units.model == model].set_index(["relative_depth", "seed"]).value
    output = np.empty(len(draws), dtype=float)
    for index, (depth_draw, seed_draw) in enumerate(draws):
        values = [float(lookup.loc[(float(depth), int(seed))]) for depth in depth_draw for seed in seed_draw]
        output[index] = float(np.mean(values))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    audit = json.loads((args.root / "audit.json").read_text())
    if not audit.get("audit_pass"):
        raise RuntimeError(f"multi-scale audit is incomplete: {audit}")
    cells = pd.read_csv(args.root / "cell_metrics.csv")
    matched_blocks = validate_matched_scale_cells(cells)
    depths = np.sort(cells.relative_depth.unique())
    seeds = np.sort(cells.seed.unique())
    if len(depths) != 5 or len(seeds) != 3:
        raise RuntimeError(f"unexpected depth/seed support: {depths}, {seeds}")

    metrics = {
        "reconstruction": f"{args.split}_recon_R2",
        "semantic_alignment": f"{args.split}_semantic_alignment",
        "concept_coverage": f"{args.split}_concept_coverage_020",
        "dead_fraction": f"{args.split}_dead_fraction",
    }
    rng = np.random.default_rng(args.seed)
    draws = list(bootstrap_indices(depths, seeds, rng, args.bootstrap_samples))
    profile_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    boot_cache: dict[tuple[str, str], np.ndarray] = {}
    for metric_name, column in metrics.items():
        units = build_auc_units(cells, column)
        for model in MODEL_ORDER:
            observed = float(units[units.model == model].value.mean())
            bootstrap = bootstrap_profile(units, model, draws)
            boot_cache[(metric_name, model)] = bootstrap
            profile_rows.append(
                {
                    "split": args.split,
                    "metric": metric_name,
                    "model": model,
                    "observed_multiscale_auc": observed,
                    "ci_low": float(np.quantile(bootstrap, 0.025)),
                    "ci_high": float(np.quantile(bootstrap, 0.975)),
                    "depths": len(depths),
                    "seeds": len(seeds),
                    "bootstrap_samples": args.bootstrap_samples,
                }
            )
        metric_pair_start = len(pair_rows)
        for left_index, left in enumerate(MODEL_ORDER):
            for right in MODEL_ORDER[left_index + 1 :]:
                difference = boot_cache[(metric_name, left)] - boot_cache[(metric_name, right)]
                observed_left = next(
                    row["observed_multiscale_auc"]
                    for row in profile_rows
                    if row["metric"] == metric_name and row["model"] == left
                )
                observed_right = next(
                    row["observed_multiscale_auc"]
                    for row in profile_rows
                    if row["metric"] == metric_name and row["model"] == right
                )
                p_two_sided = float(
                    min(
                        1.0,
                        2.0
                        * min(
                            (1.0 + np.sum(difference <= 0)) / (len(difference) + 1.0),
                            (1.0 + np.sum(difference >= 0)) / (len(difference) + 1.0),
                        ),
                    )
                )
                pair_rows.append(
                    {
                        "split": args.split,
                        "metric": metric_name,
                        "model_left": left,
                        "model_right": right,
                        "observed_difference": observed_left - observed_right,
                        "ci_low": float(np.quantile(difference, 0.025)),
                        "ci_high": float(np.quantile(difference, 0.975)),
                        "p_two_sided": p_two_sided,
                        "q_bh": float("nan"),
                        "bootstrap_samples": args.bootstrap_samples,
                    }
                )
        metric_indices = list(range(metric_pair_start, len(pair_rows)))
        q_values = bh_adjust(np.asarray([pair_rows[index]["p_two_sided"] for index in metric_indices]))
        for index, q_value in zip(metric_indices, q_values):
            pair_rows[index]["q_bh"] = float(q_value)

    rank_rows = []
    for metric_name, column in metrics.items():
        ranks: dict[float, dict[str, int]] = {}
        beneficial_high = metric_name != "dead_fraction"
        for expansion, part in cells.groupby("expansion_E", sort=True):
            means = part.groupby("model")[column].mean()
            ordered = means.sort_values(ascending=not beneficial_high).index.tolist()
            ranks[float(expansion)] = {model: rank + 1 for rank, model in enumerate(ordered)}
        expansion_values = sorted(ranks)
        for left_index, expansion_left in enumerate(expansion_values):
            for expansion_right in expansion_values[left_index + 1 :]:
                left_rank = [ranks[expansion_left][model] for model in MODEL_ORDER]
                right_rank = [ranks[expansion_right][model] for model in MODEL_ORDER]
                tau, p_value = kendalltau(left_rank, right_rank)
                rank_rows.append(
                    {
                        "split": args.split,
                        "metric": metric_name,
                        "expansion_left": expansion_left,
                        "expansion_right": expansion_right,
                        "kendall_tau": float(tau),
                        "p_value": float(p_value),
                        "rank_order_left": ">".join(
                            sorted(MODEL_ORDER, key=lambda model: ranks[expansion_left][model])
                        ),
                        "rank_order_right": ">".join(
                            sorted(MODEL_ORDER, key=lambda model: ranks[expansion_right][model])
                        ),
                    }
                )

    write_csv(args.root / f"{args.split}_model_profile_inference.csv", profile_rows)
    write_csv(args.root / f"{args.split}_model_pair_inference.csv", pair_rows)
    write_csv(args.root / f"{args.split}_scale_rank_stability.csv", rank_rows)
    metadata = {
        "status": "complete",
        "split": args.split,
        "profile_rows": len(profile_rows),
        "pair_rows": len(pair_rows),
        "rank_stability_rows": len(rank_rows),
        "bootstrap_samples": args.bootstrap_samples,
        "matched_scale_blocks": matched_blocks,
        "comparison_rule": "FM comparisons use identical relative expansion E at each depth/seed; per-model best-scale ranking is prohibited.",
        "bootstrap_unit": "crossed relative-depth and SAE-seed profile units; expansion integrated by log-scale AUC",
        "patient_level_inference": False,
        "claim_boundary": "These intervals quantify layer/seed design variation, not patient sampling uncertainty.",
    }
    (args.root / f"{args.split}_inference_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
