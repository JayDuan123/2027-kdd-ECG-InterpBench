#!/usr/bin/env python
"""Aggregate paired patient bootstraps over the complete matched-scale SAE grid."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_multiscale_sae_inference import (
    EXPECTED_DEPTHS,
    EXPECTED_EXPANSIONS,
    EXPECTED_SEEDS,
    MODEL_ORDER,
    bh_adjust,
    validate_matched_scale_cells,
)


METRICS = ("recon_R2", "semantic_alignment", "concept_coverage_020")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def log_auc(values: np.ndarray) -> np.ndarray:
    """Integrate arrays ordered by the frozen common expansion grid."""
    x = np.log(np.asarray(EXPECTED_EXPANSIONS, dtype=np.float64))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return trapezoid(values, x=x, axis=0) / (x[-1] - x[0])


def log_auc_weights() -> dict[int, float]:
    identity = np.eye(len(EXPECTED_EXPANSIONS), dtype=np.float64)
    weights = log_auc(identity)
    return {
        int(expansion): float(weight)
        for expansion, weight in zip(EXPECTED_EXPANSIONS, weights)
    }


def paired_p_value(difference: np.ndarray) -> float:
    return float(
        min(
            1.0,
            2.0
            * min(
                (1.0 + np.sum(difference <= 0)) / (len(difference) + 1.0),
                (1.0 + np.sum(difference >= 0)) / (len(difference) + 1.0),
            ),
        )
    )


def task_paths(root: Path, task_index: int, split: str) -> tuple[Path, Path]:
    task_root = root / "patient_bootstrap" / f"task_{task_index:06d}"
    return (
        task_root / f"{split}_patient_bootstrap.json",
        task_root / f"{split}_patient_bootstrap.npz",
    )


def load_metric_matrix(
    root: Path, manifest: pd.DataFrame, split: str, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    distributions = []
    observed = []
    for row in manifest.itertuples(index=False):
        _, distribution_path = task_paths(root, int(row.task_index), split)
        with np.load(distribution_path, allow_pickle=False) as archive:
            distributions.append(np.asarray(archive[metric], dtype=np.float64))
            observed.append(float(np.asarray(archive[f"observed_{metric}"])))
    return np.stack(distributions), np.asarray(observed, dtype=np.float64)


def rank_summary(
    metric: str, scope: str, profiles: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    matrix = np.stack([profiles[model] for model in MODEL_ORDER])
    order = np.argsort(-matrix, axis=0)
    ranks = np.empty_like(order)
    for draw in range(matrix.shape[1]):
        ranks[order[:, draw], draw] = np.arange(1, len(MODEL_ORDER) + 1)
    rows = []
    for model_index, model in enumerate(MODEL_ORDER):
        rows.append(
            {
                "metric": metric,
                "scope": scope,
                "model": model,
                "mean_rank": float(ranks[model_index].mean()),
                "median_rank": float(np.median(ranks[model_index])),
                "probability_rank_1": float(np.mean(ranks[model_index] == 1)),
                "probability_top_2": float(np.mean(ranks[model_index] <= 2)),
                "bootstrap_samples": int(matrix.shape[1]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    audit = json.loads((args.root / "audit.json").read_text())
    if not audit.get("audit_pass") or not audit.get("matched_scale_grid_pass"):
        raise RuntimeError(f"complete matched-scale audit required: {audit}")
    manifest = pd.read_csv(args.root / "training_manifest.csv").sort_values("task_index")
    matched_blocks = validate_matched_scale_cells(manifest)

    task_metadata = []
    for row in manifest.itertuples(index=False):
        summary_path, distribution_path = task_paths(args.root, int(row.task_index), args.split)
        if not summary_path.exists() or not distribution_path.exists():
            raise FileNotFoundError(f"missing patient bootstrap for task {row.task_index}")
        metadata = json.loads(summary_path.read_text())
        required = {
            "status": "complete",
            "task_index": int(row.task_index),
            "config_hash": str(row.config_hash),
            "model": str(row.model),
            "relative_depth": float(row.relative_depth),
            "expansion_E": int(row.expansion_E),
            "seed": int(row.seed),
            "split": args.split,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in required.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"patient bootstrap identity mismatch for task {row.task_index}: {mismatches}")
        task_metadata.append(metadata)
    patient_hashes = {row["patient_id_hash"] for row in task_metadata}
    patient_cluster_hashes = {row["patient_cluster_hash"] for row in task_metadata}
    bootstrap_design_hashes = {row["bootstrap_design_hash"] for row in task_metadata}
    bootstrap_protocols = {row["patient_bootstrap_protocol"] for row in task_metadata}
    patient_counts = {int(row["n_patients"]) for row in task_metadata}
    sample_counts = {int(row["bootstrap_samples"]) for row in task_metadata}
    bootstrap_seeds = {int(row["bootstrap_seed"]) for row in task_metadata}
    invariants = (
        patient_hashes,
        patient_cluster_hashes,
        bootstrap_design_hashes,
        bootstrap_protocols,
        patient_counts,
        sample_counts,
        bootstrap_seeds,
    )
    if any(len(values) != 1 for values in invariants):
        raise RuntimeError(
            "paired bootstrap invariant failed: "
            f"hashes={len(patient_hashes)}, patients={patient_counts}, "
            f"cluster_hashes={len(patient_cluster_hashes)}, "
            f"design_hashes={len(bootstrap_design_hashes)}, "
            f"protocols={bootstrap_protocols}, samples={sample_counts}, "
            f"seeds={bootstrap_seeds}"
        )
    bootstrap_samples = next(iter(sample_counts))

    surface_rows: list[dict[str, Any]] = []
    scale_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    semantic_profile_cache: dict[str, tuple[float, np.ndarray]] = {}

    for metric in METRICS:
        distributions, observed = load_metric_matrix(args.root, manifest, args.split, metric)
        if distributions.shape != (len(manifest), bootstrap_samples):
            raise RuntimeError(
                f"unexpected {metric} matrix shape: {distributions.shape}"
            )
        scale_cache: dict[tuple[str, int], tuple[float, np.ndarray]] = {}
        for model in MODEL_ORDER:
            for depth in EXPECTED_DEPTHS:
                for expansion in EXPECTED_EXPANSIONS:
                    mask = (
                        (manifest.model == model)
                        & np.isclose(manifest.relative_depth, depth)
                        & (manifest.expansion_E == expansion)
                    ).to_numpy()
                    if int(mask.sum()) != len(EXPECTED_SEEDS):
                        raise RuntimeError(
                            f"seed support mismatch for {model}/depth={depth}/E={expansion}"
                        )
                    values = distributions[mask].mean(axis=0)
                    point = float(observed[mask].mean())
                    low, high = interval(values)
                    surface_rows.append(
                        {
                            "split": args.split,
                            "metric": metric,
                            "model": model,
                            "relative_depth": depth,
                            "expansion_E": expansion,
                            "observed": point,
                            "ci_low": low,
                            "ci_high": high,
                            "sae_seeds": len(EXPECTED_SEEDS),
                            "bootstrap_samples": bootstrap_samples,
                        }
                    )
            for expansion in EXPECTED_EXPANSIONS:
                mask = (
                    (manifest.model == model)
                    & (manifest.expansion_E == expansion)
                ).to_numpy()
                expected = len(EXPECTED_DEPTHS) * len(EXPECTED_SEEDS)
                if int(mask.sum()) != expected:
                    raise RuntimeError(f"matched-scale support mismatch for {model}/E={expansion}")
                values = distributions[mask].mean(axis=0)
                point = float(observed[mask].mean())
                scale_cache[(model, int(expansion))] = (point, values)
                low, high = interval(values)
                scale_rows.append(
                    {
                        "split": args.split,
                        "metric": metric,
                        "model": model,
                        "expansion_E": expansion,
                        "observed": point,
                        "ci_low": low,
                        "ci_high": high,
                        "depth_seed_cells": expected,
                        "bootstrap_samples": bootstrap_samples,
                    }
                )
        profiles: dict[str, np.ndarray] = {}
        observed_profiles: dict[str, float] = {}
        for model in MODEL_ORDER:
            scale_observed = np.asarray(
                [scale_cache[(model, int(expansion))][0] for expansion in EXPECTED_EXPANSIONS]
            )
            scale_distributions = np.stack(
                [scale_cache[(model, int(expansion))][1] for expansion in EXPECTED_EXPANSIONS]
            )
            point = float(log_auc(scale_observed))
            values = np.asarray(log_auc(scale_distributions), dtype=np.float64)
            profiles[model] = values
            observed_profiles[model] = point
            if metric == "semantic_alignment":
                semantic_profile_cache[model] = (point, values)
            low, high = interval(values)
            profile_rows.append(
                {
                    "split": args.split,
                    "metric": metric,
                    "model": model,
                    "observed_common_scale_auc": point,
                    "ci_low": low,
                    "ci_high": high,
                    "depths": len(EXPECTED_DEPTHS),
                    "expansion_scales": len(EXPECTED_EXPANSIONS),
                    "sae_seeds": len(EXPECTED_SEEDS),
                    "bootstrap_samples": bootstrap_samples,
                }
            )
        pair_start = len(pair_rows)
        for left_index, left in enumerate(MODEL_ORDER):
            for right in MODEL_ORDER[left_index + 1 :]:
                difference = profiles[left] - profiles[right]
                low, high = interval(difference)
                pair_rows.append(
                    {
                        "split": args.split,
                        "metric": metric,
                        "model_left": left,
                        "model_right": right,
                        "observed_difference": observed_profiles[left] - observed_profiles[right],
                        "ci_low": low,
                        "ci_high": high,
                        "p_two_sided": paired_p_value(difference),
                        "q_bh": float("nan"),
                        "bootstrap_samples": bootstrap_samples,
                    }
                )
        pair_indices = list(range(pair_start, len(pair_rows)))
        q_values = bh_adjust(
            np.asarray([pair_rows[index]["p_two_sided"] for index in pair_indices])
        )
        for index, q_value in zip(pair_indices, q_values):
            pair_rows[index]["q_bh"] = float(q_value)
        rank_rows.extend(rank_summary(metric, "common_scale_auc", profiles))
        for expansion in EXPECTED_EXPANSIONS:
            rank_rows.extend(
                rank_summary(
                    metric,
                    f"E{int(expansion)}",
                    {
                        model: scale_cache[(model, int(expansion))][1]
                        for model in MODEL_ORDER
                    },
                )
            )

    concept_rows: list[dict[str, Any]] = []
    auc_weights = log_auc_weights()
    concept_names_reference: list[str] | None = None
    concept_profile_consistency = {}
    for model in MODEL_ORDER:
        model_rows = manifest[manifest.model == model]
        concept_distribution = None
        concept_observed = None
        for row in model_rows.itertuples(index=False):
            _, distribution_path = task_paths(args.root, int(row.task_index), args.split)
            with np.load(distribution_path, allow_pickle=False) as archive:
                names = [str(value) for value in archive["concept_names"].tolist()]
                if concept_names_reference is None:
                    concept_names_reference = names
                elif names != concept_names_reference:
                    raise RuntimeError(f"concept order mismatch for task {row.task_index}")
                values = np.abs(np.asarray(archive["concept_correlation"], dtype=np.float64))
                observed_values = np.abs(
                    np.asarray(archive["observed_concept_correlation"], dtype=np.float64)
                )
            coefficient = (
                auc_weights[int(row.expansion_E)]
                / len(EXPECTED_DEPTHS)
                / len(EXPECTED_SEEDS)
            )
            if concept_distribution is None:
                concept_distribution = np.zeros_like(values)
                concept_observed = np.zeros_like(observed_values)
            concept_distribution += coefficient * values
            concept_observed += coefficient * observed_values
        assert concept_distribution is not None and concept_observed is not None
        expected_point, expected_distribution = semantic_profile_cache[model]
        consistency = {
            "observed_error": float(abs(concept_observed.mean() - expected_point)),
            "distribution_max_error": float(
                np.max(np.abs(concept_distribution.mean(axis=1) - expected_distribution))
            ),
        }
        if max(consistency.values()) > 2e-6:
            raise RuntimeError(f"concept/profile consistency failed for {model}: {consistency}")
        concept_profile_consistency[model] = consistency
        for concept_index, concept in enumerate(concept_names_reference or []):
            values = concept_distribution[:, concept_index]
            low, high = interval(values)
            concept_rows.append(
                {
                    "split": args.split,
                    "model": model,
                    "concept": concept,
                    "observed_common_scale_auc_abs_correlation": float(
                        concept_observed[concept_index]
                    ),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_samples": bootstrap_samples,
                }
            )

    atomic_csv(args.root / f"{args.split}_patient_bootstrap_surface.csv", surface_rows)
    atomic_csv(args.root / f"{args.split}_patient_bootstrap_matched_scales.csv", scale_rows)
    atomic_csv(args.root / f"{args.split}_patient_bootstrap_model_profiles.csv", profile_rows)
    atomic_csv(args.root / f"{args.split}_patient_bootstrap_model_pairs.csv", pair_rows)
    atomic_csv(args.root / f"{args.split}_patient_bootstrap_rank_probabilities.csv", rank_rows)
    atomic_csv(args.root / f"{args.split}_patient_bootstrap_concepts.csv", concept_rows)
    metadata = {
        "status": "complete",
        "split": args.split,
        "verified_tasks": len(task_metadata),
        "expected_tasks": len(manifest),
        "matched_scale_blocks": matched_blocks,
        "n_patients": next(iter(patient_counts)),
        "patient_id_hash": next(iter(patient_hashes)),
        "patient_cluster_hash": next(iter(patient_cluster_hashes)),
        "bootstrap_design_hash": next(iter(bootstrap_design_hashes)),
        "patient_bootstrap_protocol": next(iter(bootstrap_protocols)),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": next(iter(bootstrap_seeds)),
        "surface_rows": len(surface_rows),
        "matched_scale_rows": len(scale_rows),
        "model_profile_rows": len(profile_rows),
        "model_pair_rows": len(pair_rows),
        "rank_probability_rows": len(rank_rows),
        "concept_rows": len(concept_rows),
        "concept_profile_consistency": concept_profile_consistency,
        "comparison_rule": "FMs are compared at identical relative expansion E; common-scale AUC integrates the same five E values for every FM.",
        "per_model_best_scale_ranking": False,
        "resampling_unit": "PTB-XL patient",
        "claim_boundary": "patient sampling uncertainty conditional on frozen FM/SAE weights and train-selected features; the three fixed SAE seeds are averaged, not resampled",
    }
    atomic_json(args.root / f"{args.split}_patient_bootstrap_audit.json", metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
