#!/usr/bin/env python
"""Audit and summarize the frozen multi-scale SAE result surface."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multiscale_sae import (
    DEFAULT_DEPTHS,
    DEFAULT_EXPANSIONS,
    DEFAULT_SEEDS,
    MODEL_SUFFIXES,
    canonical_config_hash,
    read_csv,
)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def numeric(row: dict[str, str], key: str, cast=float):
    return cast(float(row[key]))


def metric_value(metrics: dict[str, Any], split: str, key: str) -> float:
    try:
        return float(metrics[split][key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def normalized_log_auc(points: list[tuple[float, float]]) -> float:
    finite = sorted((x, y) for x, y in points if x > 0 and np.isfinite(y))
    if len(finite) < 2:
        return float("nan")
    x = np.log(np.asarray([point[0] for point in finite], dtype=float))
    y = np.asarray([point[1] for point in finite], dtype=float)
    if x[-1] <= x[0]:
        return float("nan")
    return float(np.trapz(y, x=x) / (x[-1] - x[0]))


def matched_scale_manifest_audit(
    manifest: list[dict[str, str]],
    expected_models: tuple[str, ...] = tuple(MODEL_SUFFIXES),
    expected_depths: tuple[float, ...] = DEFAULT_DEPTHS,
    expected_expansions: tuple[int, ...] = DEFAULT_EXPANSIONS,
    expected_seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    """Require every common depth/scale/seed block to contain all six FMs once."""
    expected_keys = {
        (model, float(depth), int(expansion), int(seed))
        for model in expected_models
        for depth in expected_depths
        for expansion in expected_expansions
        for seed in expected_seeds
    }
    observed_keys = [
        (
            row["model"],
            numeric(row, "relative_depth"),
            numeric(row, "expansion_E", int),
            numeric(row, "seed", int),
        )
        for row in manifest
    ]
    counts = Counter(observed_keys)
    issues: list[str] = []
    missing = expected_keys - set(counts)
    unexpected = set(counts) - expected_keys
    duplicated = {key: count for key, count in counts.items() if count != 1}
    if missing:
        issues.append(f"missing_expected_cells={len(missing)}")
    if unexpected:
        issues.append(f"unexpected_cells={len(unexpected)}")
    if duplicated:
        issues.append(f"nonunique_cells={len(duplicated)}")

    for model in expected_models:
        for depth in expected_depths:
            layer_keys = {
                (numeric(row, "layer", int), numeric(row, "actual_relative_depth"))
                for row in manifest
                if row["model"] == model
                and math.isclose(numeric(row, "relative_depth"), float(depth))
            }
            if len(layer_keys) != 1:
                issues.append(f"nonconstant_layer_mapping={model}@{float(depth):g}")

    block_rows: list[dict[str, Any]] = []
    for depth in expected_depths:
        for expansion in expected_expansions:
            for seed in expected_seeds:
                block = [
                    row
                    for row in manifest
                    if math.isclose(numeric(row, "relative_depth"), float(depth))
                    and numeric(row, "expansion_E", int) == int(expansion)
                    and numeric(row, "seed", int) == int(seed)
                ]
                observed_models = [row["model"] for row in block]
                model_counts = Counter(observed_models)
                widths = sorted({numeric(row, "N", int) for row in block})
                active_budgets = sorted({numeric(row, "k", int) for row in block})
                hidden_dimensions = sorted({numeric(row, "d_hidden", int) for row in block})
                reasons = []
                if set(model_counts) != set(expected_models):
                    reasons.append("model_support_mismatch")
                if any(count != 1 for count in model_counts.values()):
                    reasons.append("duplicate_model")
                if len(hidden_dimensions) != 1:
                    reasons.append("hidden_dimension_mismatch")
                if len(widths) != 1:
                    reasons.append("absolute_dictionary_width_mismatch")
                if len(active_budgets) != 1:
                    reasons.append("active_budget_mismatch")
                block_rows.append(
                    {
                        "relative_depth": float(depth),
                        "expansion_E": int(expansion),
                        "seed": int(seed),
                        "expected_models": ";".join(expected_models),
                        "observed_models": ";".join(sorted(observed_models)),
                        "n_rows": len(block),
                        "hidden_dimensions": ";".join(map(str, hidden_dimensions)),
                        "absolute_dictionary_widths": ";".join(map(str, widths)),
                        "active_budgets": ";".join(map(str, active_budgets)),
                        "manifest_status": "pass" if not reasons else "fail",
                        "manifest_reasons": ";".join(reasons),
                    }
                )
    grid_pass = not issues and all(row["manifest_status"] == "pass" for row in block_rows)
    return block_rows, grid_pass, issues


def record_manifest_alignment_audit(
    manifest: list[dict[str, str]], expected_models: tuple[str, ...]
) -> tuple[list[dict[str, Any]], bool]:
    rows: list[dict[str, Any]] = []
    for model in expected_models:
        paths = sorted(
            {str(row["records_path"]) for row in manifest if row["model"] == model}
        )
        reasons: list[str] = []
        record_count = 0
        digest = ""
        path_text = ";".join(paths)
        if len(paths) != 1:
            reasons.append("nonunique_record_manifest")
        else:
            path = Path(paths[0])
            if not path.exists():
                reasons.append("missing_record_manifest")
            else:
                records = read_csv(path)
                record_count = len(records)
                if not records or not {"ecg_id", "split"}.issubset(records[0]):
                    reasons.append("invalid_record_manifest")
                else:
                    ordered = "\n".join(
                        f"{row['ecg_id']}\t{row['split']}" for row in records
                    )
                    digest = hashlib.sha256(ordered.encode()).hexdigest()
        rows.append(
            {
                "model": model,
                "records_path": path_text,
                "record_count": record_count,
                "ordered_ecg_split_sha256": digest,
                "status": "pass" if not reasons else "fail",
                "reasons": ";".join(reasons),
            }
        )
    hashes = {row["ordered_ecg_split_sha256"] for row in rows if row["status"] == "pass"}
    counts = {int(row["record_count"]) for row in rows if row["status"] == "pass"}
    passed = (
        len(rows) == len(expected_models)
        and all(row["status"] == "pass" for row in rows)
        and len(hashes) == 1
        and len(counts) == 1
    )
    return rows, passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--expected-concepts", type=int, default=49)
    args = parser.parse_args()

    manifest_path = args.root / "training_manifest.csv"
    manifest = read_csv(manifest_path)
    if not manifest:
        raise RuntimeError(f"empty or missing manifest: {manifest_path}")
    protocol_path = args.root / "protocol.json"
    if not protocol_path.exists():
        raise FileNotFoundError(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    expected_models = tuple(str(value) for value in protocol["models"])
    expected_depths = tuple(float(value) for value in protocol["relative_depths"])
    expected_expansions = tuple(int(value) for value in protocol["expansion_E"])
    expected_seeds = tuple(int(value) for value in protocol["seeds"])
    record_manifest_rows, record_manifest_alignment_pass = (
        record_manifest_alignment_audit(manifest, expected_models)
    )
    write_csv(args.root / "record_manifest_audit.csv", record_manifest_rows)
    matched_block_rows, matched_scale_grid_pass, matched_scale_grid_issues = (
        matched_scale_manifest_audit(
            manifest,
            expected_models=expected_models,
            expected_depths=expected_depths,
            expected_expansions=expected_expansions,
            expected_seeds=expected_seeds,
        )
    )
    if not record_manifest_alignment_pass:
        matched_scale_grid_issues.append("record_manifest_alignment_failed")
        matched_scale_grid_pass = False
    coverage_rows: list[dict[str, Any]] = []
    complete_metrics: list[dict[str, Any]] = []
    for row in manifest:
        reasons = []
        expected_hash = canonical_config_hash(
            {
                **row,
                "task_index": numeric(row, "task_index", int),
                "layer": numeric(row, "layer", int),
                "n_layers": numeric(row, "n_layers", int),
                "d_hidden": numeric(row, "d_hidden", int),
                "expansion_E": numeric(row, "expansion_E", int),
                "N": numeric(row, "N", int),
                "k": numeric(row, "k", int),
                "seed": numeric(row, "seed", int),
                "steps": numeric(row, "steps", int),
                "batch_size": numeric(row, "batch_size", int),
                "relative_depth": numeric(row, "relative_depth"),
                "actual_relative_depth": numeric(row, "actual_relative_depth"),
                "learning_rate": numeric(row, "learning_rate"),
            }
        )
        if expected_hash != row["config_hash"]:
            reasons.append("manifest_hash_mismatch")
        d_hidden = numeric(row, "d_hidden", int)
        expansion = numeric(row, "expansion_E", int)
        n_features = numeric(row, "N", int)
        k = numeric(row, "k", int)
        if n_features != d_hidden * expansion:
            reasons.append("N_over_d_invariant_failed")
        if row["sparsity_arm"] == "fixed_k_over_d" and not math.isclose(k / d_hidden, 1 / 8):
            reasons.append("k_over_d_invariant_failed")
        if row["sparsity_arm"] == "fixed_k_over_n" and not math.isclose(k / n_features, 1 / 64):
            reasons.append("k_over_N_invariant_failed")

        checkpoint = Path(row["checkpoint"])
        metrics_path = Path(row["metrics"])
        concept_path = Path(row["concept_metrics"])
        firing_path = Path(row["firing_rate"])
        metrics: dict[str, Any] = {}
        if not checkpoint.exists() or checkpoint.stat().st_size == 0:
            reasons.append("missing_checkpoint")
        if not metrics_path.exists():
            reasons.append("missing_metrics")
        else:
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                reasons.append("invalid_metrics_json")
            if metrics and metrics.get("status") != "complete":
                reasons.append("metrics_not_complete")
            if metrics and metrics.get("config_hash") != row["config_hash"]:
                reasons.append("metrics_hash_mismatch")
            required_metrics = (
                metric_value(metrics, "validation", "recon_R2"),
                metric_value(metrics, "validation", "dead_fraction"),
                metric_value(metrics, "validation", "mean_train_selected_abs_correlation"),
                metric_value(metrics, "test", "recon_R2"),
                metric_value(metrics, "test", "mean_train_selected_abs_correlation"),
            )
            if metrics and not all(np.isfinite(required_metrics)):
                reasons.append("nonfinite_required_metric")
        concept_count = 0
        if not concept_path.exists():
            reasons.append("missing_concept_metrics")
        else:
            concept_count = len(read_csv(concept_path))
            if concept_count != 2 * args.expected_concepts:
                reasons.append("unexpected_concept_row_count")
        firing_shape = ""
        if not firing_path.exists():
            reasons.append("missing_firing_rate")
        else:
            try:
                firing = np.load(firing_path, mmap_mode="r")
                firing_shape = str(tuple(firing.shape))
                if firing.shape != (n_features,):
                    reasons.append("firing_rate_shape_mismatch")
            except Exception:
                reasons.append("invalid_firing_rate")

        status = "complete" if not reasons else "incomplete"
        coverage_rows.append(
            {
                "task_index": numeric(row, "task_index", int),
                "model": row["model"],
                "layer": numeric(row, "layer", int),
                "relative_depth": numeric(row, "relative_depth"),
                "actual_relative_depth": numeric(row, "actual_relative_depth"),
                "expansion_E": expansion,
                "sparsity_arm": row["sparsity_arm"],
                "seed": numeric(row, "seed", int),
                "N": n_features,
                "k": k,
                "status": status,
                "reasons": ";".join(reasons),
                "checkpoint_bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
                "concept_rows": concept_count,
                "firing_shape": firing_shape,
            }
        )
        if status == "complete":
            complete_metrics.append(
                {
                    "model": row["model"],
                    "layer": numeric(row, "layer", int),
                    "relative_depth": numeric(row, "relative_depth"),
                    "actual_relative_depth": numeric(row, "actual_relative_depth"),
                    "expansion_E": expansion,
                    "sparsity_arm": row["sparsity_arm"],
                    "seed": numeric(row, "seed", int),
                    "validation_recon_R2": metric_value(metrics, "validation", "recon_R2"),
                    "validation_dead_fraction": metric_value(metrics, "validation", "dead_fraction"),
                    "validation_mean_l0": metric_value(metrics, "validation", "mean_l0"),
                    "validation_semantic_alignment": metric_value(
                        metrics, "validation", "mean_train_selected_abs_correlation"
                    ),
                    "validation_concept_coverage_020": metric_value(
                        metrics, "validation", "coverage_abs_r_ge_0_20"
                    ),
                    "test_recon_R2": metric_value(metrics, "test", "recon_R2"),
                    "test_dead_fraction": metric_value(metrics, "test", "dead_fraction"),
                    "test_mean_l0": metric_value(metrics, "test", "mean_l0"),
                    "test_semantic_alignment": metric_value(
                        metrics, "test", "mean_train_selected_abs_correlation"
                    ),
                    "test_concept_coverage_020": metric_value(metrics, "test", "coverage_abs_r_ge_0_20"),
                }
            )

    write_csv(args.root / "coverage_audit.csv", coverage_rows)
    completion_by_key = {
        (
            row["model"],
            float(row["relative_depth"]),
            int(row["expansion_E"]),
            int(row["seed"]),
        ): row["status"] == "complete"
        for row in coverage_rows
    }
    for block in matched_block_rows:
        complete_models = [
            model
            for model in expected_models
            if completion_by_key.get(
                (
                    model,
                    float(block["relative_depth"]),
                    int(block["expansion_E"]),
                    int(block["seed"]),
                ),
                False,
            )
        ]
        block["complete_models"] = ";".join(complete_models)
        block["n_complete_models"] = len(complete_models)
        block["result_status"] = "complete" if len(complete_models) == len(expected_models) else "incomplete"
    write_csv(args.root / "matched_scale_grid_audit.csv", matched_block_rows)
    complete_fields = [
        "model",
        "layer",
        "relative_depth",
        "actual_relative_depth",
        "expansion_E",
        "sparsity_arm",
        "seed",
        "validation_recon_R2",
        "validation_dead_fraction",
        "validation_mean_l0",
        "validation_semantic_alignment",
        "validation_concept_coverage_020",
        "test_recon_R2",
        "test_dead_fraction",
        "test_mean_l0",
        "test_semantic_alignment",
        "test_concept_coverage_020",
    ]
    write_csv(args.root / "cell_metrics.csv", complete_metrics, complete_fields)

    grouped: dict[tuple[str, int, float], list[dict[str, Any]]] = {}
    for row in complete_metrics:
        grouped.setdefault((row["model"], row["expansion_E"], row["relative_depth"]), []).append(row)
    surface_rows = []
    metric_names = complete_fields[7:]
    for (model, expansion, depth), rows in sorted(grouped.items()):
        surface = {
            "model": model,
            "expansion_E": expansion,
            "relative_depth": depth,
            "actual_relative_depth": float(np.mean([row["actual_relative_depth"] for row in rows])),
            "layer": int(rows[0]["layer"]),
            "seeds": len({int(row["seed"]) for row in rows}),
        }
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in rows], dtype=float)
            surface[f"{name}_mean"] = float(np.mean(values))
            surface[f"{name}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        surface_rows.append(surface)
    write_csv(args.root / "layer_scale_surface.csv", surface_rows)

    model_profiles = []
    for model in sorted({row["model"] for row in surface_rows}):
        model_rows = [row for row in surface_rows if row["model"] == model]
        expansion_rows = []
        for expansion in sorted({row["expansion_E"] for row in model_rows}):
            rows = [row for row in model_rows if row["expansion_E"] == expansion]
            expansion_rows.append(
                {
                    "E": float(expansion),
                    "recon": float(np.mean([row["validation_recon_R2_mean"] for row in rows])),
                    "semantic": float(np.mean([row["validation_semantic_alignment_mean"] for row in rows])),
                    "coverage": float(np.mean([row["validation_concept_coverage_020_mean"] for row in rows])),
                    "dead": float(np.mean([row["validation_dead_fraction_mean"] for row in rows])),
                }
            )
        model_profiles.append(
            {
                "model": model,
                "n_layer_scale_cells": len(model_rows),
                "validation_multiscale_recon_auc": normalized_log_auc(
                    [(row["E"], row["recon"]) for row in expansion_rows]
                ),
                "validation_multiscale_semantic_auc": normalized_log_auc(
                    [(row["E"], row["semantic"]) for row in expansion_rows]
                ),
                "validation_multiscale_coverage_auc": normalized_log_auc(
                    [(row["E"], row["coverage"]) for row in expansion_rows]
                ),
                "test_multiscale_recon_auc": normalized_log_auc(
                    [
                        (
                            float(expansion),
                            float(
                                np.mean(
                                    [
                                        row["test_recon_R2_mean"]
                                        for row in model_rows
                                        if row["expansion_E"] == expansion
                                    ]
                                )
                            ),
                        )
                        for expansion in sorted({row["expansion_E"] for row in model_rows})
                    ]
                ),
                "test_multiscale_semantic_auc": normalized_log_auc(
                    [
                        (
                            float(expansion),
                            float(
                                np.mean(
                                    [
                                        row["test_semantic_alignment_mean"]
                                        for row in model_rows
                                        if row["expansion_E"] == expansion
                                    ]
                                )
                            ),
                        )
                        for expansion in sorted({row["expansion_E"] for row in model_rows})
                    ]
                ),
                "test_multiscale_coverage_auc": normalized_log_auc(
                    [
                        (
                            float(expansion),
                            float(
                                np.mean(
                                    [
                                        row["test_concept_coverage_020_mean"]
                                        for row in model_rows
                                        if row["expansion_E"] == expansion
                                    ]
                                )
                            ),
                        )
                        for expansion in sorted({row["expansion_E"] for row in model_rows})
                    ]
                ),
                "validation_mean_dead_fraction": float(
                    np.mean([row["validation_dead_fraction_mean"] for row in model_rows])
                ),
                "test_mean_dead_fraction": float(
                    np.mean([row["test_dead_fraction_mean"] for row in model_rows])
                ),
                "validation_fidelity_pass_fraction": float(
                    np.mean(
                        [
                            row["validation_recon_R2_mean"] >= 0.90
                            and row["validation_dead_fraction_mean"] < 0.20
                            for row in model_rows
                        ]
                    )
                ),
            }
        )
    write_csv(
        args.root / "model_profiles.csv",
        model_profiles,
        [
            "model",
            "n_layer_scale_cells",
            "validation_multiscale_recon_auc",
            "validation_multiscale_semantic_auc",
            "validation_multiscale_coverage_auc",
            "test_multiscale_recon_auc",
            "test_multiscale_semantic_auc",
            "test_multiscale_coverage_auc",
            "validation_mean_dead_fraction",
            "test_mean_dead_fraction",
            "validation_fidelity_pass_fraction",
        ],
    )

    expected_cells = (
        len(expected_models)
        * len(expected_depths)
        * len(expected_expansions)
        * len(expected_seeds)
    )
    expected_matched_blocks = (
        len(expected_depths) * len(expected_expansions) * len(expected_seeds)
    )
    complete_matched_blocks = sum(
        row["result_status"] == "complete" for row in matched_block_rows
    )
    exact_absolute_scale_blocks = sum(
        row["manifest_status"] == "pass"
        and ";" not in row["absolute_dictionary_widths"]
        and ";" not in row["active_budgets"]
        for row in matched_block_rows
    )
    n_complete = sum(row["status"] == "complete" for row in coverage_rows)
    model_complete_counts = Counter(
        row["model"] for row in coverage_rows if row["status"] == "complete"
    )
    cells_per_model = len(expected_depths) * len(expected_expansions) * len(expected_seeds)
    summary = {
        "comparison_rule": "matched records, SAE architecture, d, absolute N, and k; no per-model best-scale ranking",
        "record_manifest_alignment_pass": record_manifest_alignment_pass,
        "record_manifest_hash": (
            record_manifest_rows[0]["ordered_ecg_split_sha256"]
            if record_manifest_alignment_pass
            else ""
        ),
        "record_count": (
            int(record_manifest_rows[0]["record_count"])
            if record_manifest_alignment_pass
            else 0
        ),
        "expected_cells": expected_cells,
        "manifest_cells": len(manifest),
        "complete_cells": n_complete,
        "incomplete_cells": expected_cells - n_complete,
        "coverage_fraction": n_complete / expected_cells,
        "expected_matched_blocks": expected_matched_blocks,
        "complete_matched_blocks": complete_matched_blocks,
        "exact_absolute_scale_blocks": exact_absolute_scale_blocks,
        "matched_scale_grid_pass": matched_scale_grid_pass,
        "matched_scale_grid_issues": matched_scale_grid_issues,
        "models_complete": sorted(
            model for model, count in model_complete_counts.items() if count == cells_per_model
        ),
        "audit_pass": (
            matched_scale_grid_pass
            and len(manifest) == expected_cells
            and n_complete == expected_cells
            and complete_matched_blocks == expected_matched_blocks
            and exact_absolute_scale_blocks == expected_matched_blocks
        ),
    }
    atomic = args.root / "audit.json.tmp"
    atomic.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(atomic, args.root / "audit.json")
    lines = [
        "# Multi-Scale SAE Audit",
        "",
        f"- Expected cells: {summary['expected_cells']}",
        f"- Complete cells: {summary['complete_cells']}",
        f"- Incomplete cells: {summary['incomplete_cells']}",
        f"- Coverage: {summary['coverage_fraction']:.1%}",
        f"- Matched-scale manifest grid: {summary['matched_scale_grid_pass']}",
        f"- Ordered record-manifest alignment: {summary['record_manifest_alignment_pass']}",
        f"- Record count: {summary['record_count']}",
        f"- Complete matched blocks: {summary['complete_matched_blocks']}/{summary['expected_matched_blocks']}",
        f"- Exact absolute-width/budget blocks: {summary['exact_absolute_scale_blocks']}/{summary['expected_matched_blocks']}",
        f"- Audit pass: {summary['audit_pass']}",
        "",
        "Primary comparisons require the same ordered records, SAE architecture, hidden width, absolute dictionary width, and active budget; per-model best-scale ranking is prohibited.",
        "",
        "Model profiles are emitted only from complete cells; they are not final unless coverage is 100%.",
    ]
    (args.root / "audit_report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["audit_pass"] and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
