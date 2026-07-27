#!/usr/bin/env python
"""Audit and summarize final-layer matched-budget sparse-accessibility curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.sparse_accessibility import (  # noqa: E402
    bh_adjust,
    normalized_log2_curve_auc,
)
from scripts.run_accessibility_calibration_worker import atomic_json  # noqa: E402


PROTOCOL = "final_layer_sparse_accessibility_e8_v2"
METHOD_ORDER = (
    "dense_matched",
    "pca_matched",
    "sae_matched",
    "random_matched",
    "sae_full",
    "random_full",
)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/workers",
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/bootstrap",
    )
    parser.add_argument(
        "--dense-ceiling-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v1/workers",
    )
    parser.add_argument(
        "--feature-yield-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/feature_yield",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/summary",
    )
    parser.add_argument("--expected-random-replicates", type=int, default=20)
    parser.add_argument("--expected-budget-replicates", type=int, default=20)
    return parser.parse_args()


def method_name(source_kind: str, candidate_arm: str) -> str:
    if source_kind in {"dense", "pca"}:
        return f"{source_kind}_matched"
    suffix = "full" if candidate_arm == "full_6144" else "matched"
    return f"{source_kind}_{suffix}"


def worker_tables(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables = []
    summaries = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("protocol") != PROTOCOL:
            continue
        metrics_path = Path(payload["metrics"])
        predictions_path = Path(payload["test_predictions"])
        if not metrics_path.exists() or not predictions_path.exists():
            raise FileNotFoundError(f"missing worker artifact for {path}")
        table = pd.read_csv(metrics_path)
        table["worker_summary"] = str(path)
        tables.append(table)
        summaries.append(payload)
    if not tables:
        raise RuntimeError("no completed sparse-accessibility workers")
    return pd.concat(tables, ignore_index=True), summaries


def audit_worker_counts(
    summaries: list[dict[str, Any]], random_replicates: int, budget_replicates: int
) -> None:
    frame = pd.DataFrame(summaries)
    expected = {"dense": 6, "pca": 6, "sae": 18, "random": 6 * random_replicates}
    observed = frame.groupby("source_kind").size().to_dict()
    if observed != expected:
        raise RuntimeError(f"worker count mismatch: {observed}, expected {expected}")
    if frame.model.nunique() != 6 or not np.allclose(frame.relative_depth, 1.0):
        raise RuntimeError("workers do not cover six final-layer models")
    sae = frame[frame.source_kind == "sae"]
    if not (sae.budget_replicates == budget_replicates).all():
        raise RuntimeError("SAE matched-budget replicate count mismatch")


def dense_ceilings(root: Path) -> pd.DataFrame:
    tables = []
    for path in sorted(root.glob("task_*/calibration.csv")):
        table = pd.read_csv(path)
        values = table[
            np.isclose(table.relative_depth, 1.0) & (table.method == "dense_fm")
        ]
        if len(values):
            tables.append(values)
    if not tables:
        raise RuntimeError("missing final-layer dense full-ridge ceiling")
    values = pd.concat(tables, ignore_index=True)
    return (
        values.groupby("model", as_index=False)
        .agg(dense_full_mean_abs_r=("test_abs_r", "mean"))
    )


def make_curves(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = metrics.copy()
    values["method"] = [
        method_name(source, arm)
        for source, arm in zip(values.source_kind, values.candidate_arm)
    ]
    concept = (
        values.groupby(["model", "method", "k", "concept", "family"], as_index=False)
        .agg(
            replicates=("source_index", "size"),
            mean_test_abs_r=("test_abs_r", "mean"),
            coverage_probability=("covered_020", "mean"),
            mean_validation_abs_r=("validation_abs_r", "mean"),
        )
    )
    model = (
        concept.groupby(["model", "method", "k"], as_index=False)
        .agg(
            concepts=("concept", "nunique"),
            mean_test_abs_r=("mean_test_abs_r", "mean"),
            median_test_abs_r=("mean_test_abs_r", "median"),
            coverage_020=("coverage_probability", "mean"),
        )
    )
    return concept, model


def curve_profiles(curves: pd.DataFrame, ceilings: pd.DataFrame) -> pd.DataFrame:
    ceiling = ceilings.set_index("model").dense_full_mean_abs_r
    rows = []
    for (model, method), values in curves.groupby(["model", "method"], sort=False):
        values = values.sort_values("k")
        threshold = 0.90 * float(ceiling.loc[model])
        reached = values[values.mean_test_abs_r >= threshold]
        rows.append(
            {
                "model": model,
                "method": method,
                "curve_auc_log2_k": normalized_log2_curve_auc(
                    values.k, values.mean_test_abs_r
                ),
                "coverage_auc_log2_k": normalized_log2_curve_auc(
                    values.k, values.coverage_020
                ),
                "dense_full_mean_abs_r": float(ceiling.loc[model]),
                "dense_90pct_threshold": threshold,
                "minimum_k_to_90pct_dense_full": (
                    int(reached.k.iloc[0]) if len(reached) else np.nan
                ),
                "mean_abs_r_at_k32": float(
                    values.loc[values.k == values.k.max(), "mean_test_abs_r"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_tables(root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tables = []
    summaries = []
    for path in sorted(root.glob("model_*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("protocol") != PROTOCOL:
            continue
        table_path = Path(payload["paired_table"])
        archive_path = Path(payload["bootstrap_archive"])
        if not table_path.exists() or not archive_path.exists():
            raise FileNotFoundError(f"missing bootstrap artifact for {path}")
        tables.append(pd.read_csv(table_path))
        summaries.append(payload)
    if len(tables) != 6:
        raise RuntimeError(f"expected six model bootstrap outputs, found {len(tables)}")
    table = pd.concat(tables, ignore_index=True)
    table["mean_delta_q_value_bh"] = bh_adjust(table.mean_delta_p_value)
    table["coverage_delta_q_value_bh"] = bh_adjust(table.coverage_delta_p_value)
    return table, summaries


def make_figure(curves: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "dense_matched": ("Dense 768", "#4C78A8", "-", "o"),
        "pca_matched": ("PCA 768", "#B279A2", "-", "s"),
        "sae_matched": ("SAE matched 768", "#E45756", "-", "^"),
        "random_matched": ("Random matched 768", "#54A24B", "-", "x"),
        "sae_full": ("SAE full 6144", "#F58518", "--", "^"),
        "random_full": ("Random full 6144", "#72B7B2", "--", "x"),
    }
    models = list(curves.model.drop_duplicates())
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True, sharey=True)
    for axis, model in zip(axes.ravel(), models):
        values = curves[curves.model == model]
        for method in METHOD_ORDER:
            selected = values[values.method == method].sort_values("k")
            if selected.empty:
                continue
            label, color, linestyle, marker = style[method]
            axis.plot(
                selected.k,
                selected.mean_test_abs_r,
                label=label,
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=4.5,
            )
        axis.set_title(model)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(values.k.unique()))
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("Selected feature budget k")
    for axis in axes[:, 0]:
        axis.set_ylabel("Test mean |r|")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(output / "final_layer_sparse_accessibility_curves.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "final_layer_sparse_accessibility_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metrics, worker_summaries = worker_tables(args.workers_root)
    audit_worker_counts(
        worker_summaries,
        args.expected_random_replicates,
        args.expected_budget_replicates,
    )
    concept_curves, model_curves = make_curves(metrics)
    ceilings = dense_ceilings(args.dense_ceiling_root)
    profiles = curve_profiles(model_curves, ceilings)
    paired, bootstrap_summaries = bootstrap_tables(args.bootstrap_root)
    feature_yield_audit = args.feature_yield_root / "audit.json"
    if not feature_yield_audit.exists() or json.loads(feature_yield_audit.read_text()).get("status") != "complete":
        raise RuntimeError("final-layer feature-yield audit is missing or incomplete")

    concept_curves.to_csv(args.output_root / "concept_method_curves.csv", index=False)
    model_curves.to_csv(args.output_root / "model_method_curves.csv", index=False)
    profiles.to_csv(args.output_root / "curve_profiles.csv", index=False)
    paired.to_csv(args.output_root / "paired_model_bootstrap_fdr.csv", index=False)
    make_figure(model_curves, args.output_root)
    audit = {
        "status": "complete",
        "protocol": PROTOCOL,
        "models": int(model_curves.model.nunique()),
        "methods": sorted(model_curves.method.unique().tolist()),
        "ks": sorted(int(value) for value in model_curves.k.unique()),
        "worker_cells": len(worker_summaries),
        "bootstrap_cells": len(bootstrap_summaries),
        "bootstrap_draws": int(bootstrap_summaries[0]["bootstrap_draws"]),
        "concepts": int(concept_curves.concept.nunique()),
        "matched_candidate_budget": 768,
        "full_dictionary_width": 6144,
        "fdr_family": "all model x k x method-pair tests, separately for mean accessibility and coverage",
        "selection": "train-only top-k selection and fit; validation-only ridge alpha; frozen patient-disjoint test",
        "claim_boundary": "sparse accessibility and association yield, not clinical performance, monosemanticity, or mechanism",
    }
    atomic_json(args.output_root / "audit.json", audit)
    final_k = model_curves[model_curves.k == model_curves.k.max()].copy()
    report = [
        "# Final-layer sparse accessibility",
        "",
        "Dense, PCA, SAE, and random representations are compared at identical "
        "selected-feature budgets k={1,2,4,8,16,32}. The primary arm matches 768 "
        "candidate features; full 6,144-feature SAE/random dictionaries are secondary "
        "capacity analyses. Selection and fitting use training data, ridge alpha uses "
        "validation data, and patient-disjoint test data are evaluated once.",
        "",
        "## k=32 summary",
        "",
        markdown_table(final_k.round(4)),
        "",
        "## Curve profiles",
        "",
        markdown_table(profiles.round(4)),
        "",
        "## Paired patient bootstrap",
        "",
        markdown_table(paired.round(4)),
        "",
        "Results are reported regardless of direction. Dense full ridge is an information "
        "ceiling, while the matched-k curves test sparse access efficiency.",
    ]
    (args.output_root / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
