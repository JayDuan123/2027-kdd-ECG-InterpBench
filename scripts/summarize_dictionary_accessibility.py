#!/usr/bin/env python
"""Audit and summarize the held-out E=8 dictionary accessibility benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "dictionary_accessibility_e8_v1"
METHODS = (
    "dense_native_768",
    "sae_full_6144",
    "sae_matched_768",
    "random_full_6144",
    "random_matched_768",
)
METHOD_LABELS = {
    "dense_native_768": "Dense native (768)",
    "sae_full_6144": "SAE full (6144)",
    "sae_matched_768": "SAE matched (768)",
    "random_full_6144": "Random full (6144)",
    "random_matched_768": "Random matched (768)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/dictionary_accessibility_e8_v1/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/dictionary_accessibility_e8_v1/summary",
    )
    parser.add_argument("--expected-groups", type=int, default=30)
    parser.add_argument("--expected-feature-rows", type=int, default=208)
    parser.add_argument("--expected-target-rows", type=int, default=6032)
    parser.add_argument("--skip-calibration-reproduction", action="store_true")
    return parser.parse_args()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def finite(frame: pd.DataFrame, columns: list[str], label: str, errors: list[str]) -> None:
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        errors.append(f"non-finite values in {label}: {columns}")


def read_workers(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str], set[str]]:
    errors: list[str] = []
    feature_tables = []
    target_tables = []
    hashes: set[str] = set()
    summaries = sorted(args.workers_root.glob("group_*/summary.json"))
    if len(summaries) != args.expected_groups:
        errors.append(f"expected {args.expected_groups} worker summaries, found {len(summaries)}")
    seen_groups = set()
    for path in summaries:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("protocol") != PROTOCOL:
            errors.append(f"incomplete or wrong-protocol worker: {path}")
            continue
        group = int(payload["group_index"])
        identity = (payload["model"], int(payload["layer"]))
        if group in seen_groups:
            errors.append(f"duplicate group index: {group}")
        seen_groups.add(group)
        hashes.add(str(payload.get("test_record_sha256")))
        if payload.get("selection_policy") != "target and direction selected on train; test used once for frozen evaluation":
            errors.append(f"wrong selection policy: {path}")
        normalization = payload.get("normalization_audit", {})
        if float(normalization.get("max_abs_mean_difference", 1.0)) > 1e-7:
            errors.append(f"normalization mean mismatch: {path}")
        if float(normalization.get("max_abs_scale_difference", 1.0)) > 1e-7:
            errors.append(f"normalization scale mismatch: {path}")
        feature_path = Path(payload["feature_profiles"])
        target_path = Path(payload["target_profiles"])
        raw_path = Path(payload["feature_score_arrays"])
        if not feature_path.exists() or not target_path.exists() or not raw_path.exists():
            errors.append(f"missing worker artifact: {path}")
            continue
        feature = pd.read_csv(feature_path)
        target = pd.read_csv(target_path)
        if len(feature) != args.expected_feature_rows:
            errors.append(f"wrong feature row count {len(feature)}: {feature_path}")
        if len(target) != args.expected_target_rows:
            errors.append(f"wrong target row count {len(target)}: {target_path}")
        if set(feature.method) != set(METHODS) or set(target.method) != set(METHODS):
            errors.append(f"method support mismatch: {path}")
        if set(feature.target_type) != {"waveform", "diagnosis"}:
            errors.append(f"target type support mismatch: {feature_path}")
        finite(
            feature,
            [
                "mean_test_oriented_score",
                "median_test_oriented_score",
                "q90_test_oriented_score",
                "q95_test_oriented_score",
                "max_test_oriented_score",
                "fraction_above_primary",
                "live_fraction_above_primary",
            ],
            str(feature_path),
            errors,
        )
        finite(
            target,
            ["train_value", "test_value", "test_oriented_score", "test_descriptive_score"],
            str(target_path),
            errors,
        )
        feature_key = [
            "model",
            "relative_depth",
            "target_type",
            "method",
            "sae_seed",
            "random_seed",
            "budget_replicate",
        ]
        target_key = feature_key + ["target"]
        if feature.duplicated(feature_key).any():
            errors.append(f"duplicate feature profile key: {feature_path}")
        if target.duplicated(target_key).any():
            errors.append(f"duplicate target profile key: {target_path}")
        with np.load(raw_path, allow_pickle=False) as archive:
            required = {
                "waveform_targets",
                "diagnosis_targets",
                "test_ecg_ids",
                "random_seeds",
                "budget_subsets",
                "dense_waveform_feature_score",
                "dense_diagnosis_feature_score",
                "dense_live",
                "sae_live",
                "random_live",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                errors.append(f"missing arrays {missing}: {raw_path}")
            else:
                if archive["dense_waveform_feature_score"].shape != (768,):
                    errors.append(f"dense waveform profile shape mismatch: {raw_path}")
                if archive["sae_live"].shape != (3, 6144):
                    errors.append(f"SAE live mask shape mismatch: {raw_path}")
                if archive["random_live"].shape != (20, 6144):
                    errors.append(f"random live mask shape mismatch: {raw_path}")
                if archive["budget_subsets"].shape != (20, 768):
                    errors.append(f"budget subset shape mismatch: {raw_path}")
        feature_tables.append(feature)
        target_tables.append(target)
    if len(hashes) != 1:
        errors.append(f"test record hashes differ across workers: {sorted(hashes)}")
    if not feature_tables or not target_tables:
        raise RuntimeError("no complete dictionary accessibility worker tables")
    return pd.concat(feature_tables, ignore_index=True), pd.concat(target_tables, ignore_index=True), errors, hashes


def summarize_depth(feature: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "relative_depth", "target_type", "metric", "method"]
    feature_summary = (
        feature.groupby(keys, as_index=False)
        .agg(
            dictionary_width=("dictionary_width", "max"),
            candidate_budget=("candidate_budget", "max"),
            dictionary_replicates=("replicate", "size"),
            mean_n_live=("n_live", "mean"),
            mean_feature_median=("median_test_oriented_score", "mean"),
            mean_feature_q95=("q95_test_oriented_score", "mean"),
            mean_feature_max=("max_test_oriented_score", "mean"),
            mean_high_feature_count=("n_above_primary", "mean"),
            mean_high_feature_fraction=("fraction_above_primary", "mean"),
            mean_live_high_feature_fraction=("live_fraction_above_primary", "mean"),
            primary_threshold=("primary_threshold", "first"),
        )
    )
    target_summary = (
        target.groupby(keys, as_index=False)
        .agg(
            targets=("target", "nunique"),
            target_replicate_rows=("target", "size"),
            mean_best_target_score=("test_oriented_score", "mean"),
            median_best_target_score=("test_oriented_score", "median"),
            target_coverage=("covered_primary", "mean"),
        )
    )
    return feature_summary.merge(target_summary, on=keys, validate="one_to_one")


def summarize_models(depth: pd.DataFrame) -> pd.DataFrame:
    return (
        depth.groupby(["model", "target_type", "metric", "method"], as_index=False)
        .agg(
            depths=("relative_depth", "nunique"),
            dictionary_width=("dictionary_width", "max"),
            candidate_budget=("candidate_budget", "max"),
            mean_feature_median=("mean_feature_median", "mean"),
            mean_feature_q95=("mean_feature_q95", "mean"),
            mean_feature_max=("mean_feature_max", "mean"),
            mean_high_feature_count=("mean_high_feature_count", "mean"),
            mean_high_feature_fraction=("mean_high_feature_fraction", "mean"),
            mean_live_high_feature_fraction=("mean_live_high_feature_fraction", "mean"),
            mean_best_target_score=("mean_best_target_score", "mean"),
            target_coverage=("target_coverage", "mean"),
            primary_threshold=("primary_threshold", "first"),
        )
    )


def paired_depth_summary(depth: pd.DataFrame) -> pd.DataFrame:
    values = [
        "mean_high_feature_count",
        "mean_high_feature_fraction",
        "mean_live_high_feature_fraction",
        "mean_feature_q95",
        "mean_feature_max",
        "mean_best_target_score",
        "target_coverage",
    ]
    wide = depth.pivot(
        index=["model", "relative_depth", "target_type", "metric"],
        columns="method",
        values=values,
    )
    rows = []
    comparisons = {
        "sae_full_minus_dense_native": ("sae_full_6144", "dense_native_768"),
        "sae_matched_minus_dense_native": ("sae_matched_768", "dense_native_768"),
        "sae_full_minus_random_full": ("sae_full_6144", "random_full_6144"),
        "sae_matched_minus_random_matched": ("sae_matched_768", "random_matched_768"),
    }
    for index, row in wide.iterrows():
        output = dict(zip(["model", "relative_depth", "target_type", "metric"], index))
        for comparison, (left, right) in comparisons.items():
            for value in values:
                output[f"{comparison}__{value}"] = float(row[(value, left)] - row[(value, right)])
        rows.append(output)
    return pd.DataFrame(rows)


def target_summary(target: pd.DataFrame) -> pd.DataFrame:
    return (
        target.groupby(
            ["model", "target_type", "target", "family", "metric", "method"],
            as_index=False,
        )
        .agg(
            depths=("relative_depth", "nunique"),
            mean_test_oriented_score=("test_oriented_score", "mean"),
            mean_test_descriptive_score=("test_descriptive_score", "mean"),
            coverage=("covered_primary", "mean"),
        )
    )


def calibration_reproduction(target: pd.DataFrame, errors: list[str]) -> dict[str, Any]:
    dense_new = target[
        (target.method == "dense_native_768") & (target.target_type == "waveform")
    ].copy()
    dense_old = pd.read_csv(
        ROOT / "results/accessibility_calibration_e8_v2/summary/all_dense_single.csv"
    ).rename(columns={"concept": "target"})
    dense_keys = ["model", "layer", "relative_depth", "target"]
    dense = dense_new.merge(
        dense_old[dense_keys + ["selected_feature", "test_r"]],
        on=dense_keys,
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    dense_mismatch = int(
        np.sum(
            dense.selected_feature_new.astype(int).to_numpy()
            != dense.selected_feature_old.astype(int).to_numpy()
        )
    )
    dense_error = float(np.max(np.abs(dense.test_value - dense.test_r)))
    if len(dense) != 1470 or dense_mismatch != 0 or dense_error > 2e-6:
        errors.append(
            f"dense v2 reproduction failed: rows={len(dense)}, mismatches={dense_mismatch}, error={dense_error}"
        )

    sae_new = target[
        (target.method == "sae_full_6144") & (target.target_type == "waveform")
    ].copy()
    sae_old = pd.read_csv(
        ROOT / "results/accessibility_calibration_e8_v1/summary/all_cell_concepts.csv"
    )
    sae_old = sae_old[sae_old.method == "sae_single"].rename(
        columns={"concept": "target", "seed": "sae_seed"}
    )
    sae_keys = ["model", "layer", "relative_depth", "sae_seed", "target"]
    sae = sae_new.merge(
        sae_old[sae_keys + ["selected_features", "test_r"]],
        on=sae_keys,
        validate="one_to_one",
    )
    same_feature = sae.selected_feature.astype(int) == sae.selected_features.astype(int)
    sae_mismatch = int((~same_feature).sum())
    sae_same_feature_error = float(
        np.max(np.abs(sae.loc[same_feature, "test_value"] - sae.loc[same_feature, "test_r"]))
    )
    sae_mean_abs_error = abs(
        float(sae.test_descriptive_score.mean()) - float(np.abs(sae.test_r).mean())
    )
    expected_mismatch = 0
    for path in (ROOT / "results/accessibility_calibration_e8_v1/workers").glob(
        "task_*/summary.json"
    ):
        payload = json.loads(path.read_text())
        expected_mismatch += int(
            payload.get("single_atom_reproduction", {}).get(
                "recomputed_ranking_mismatch_count", 0
            )
        )
    if (
        len(sae) != 4410
        or sae_mismatch != expected_mismatch
        or sae_same_feature_error > 2e-4
        or sae_mean_abs_error > 1e-4
    ):
        errors.append(
            "SAE v1 reproduction failed: "
            f"rows={len(sae)}, mismatches={sae_mismatch}/{expected_mismatch}, "
            f"same_feature_error={sae_same_feature_error}, mean_abs_error={sae_mean_abs_error}"
        )
    return {
        "dense_rows_reproduced": len(dense),
        "dense_selected_feature_mismatches": dense_mismatch,
        "dense_max_abs_test_r_error": dense_error,
        "sae_rows_reproduced": len(sae),
        "sae_recomputed_tie_mismatches": sae_mismatch,
        "sae_expected_v1_tie_mismatches": expected_mismatch,
        "sae_same_feature_max_abs_test_r_error": sae_same_feature_error,
        "sae_macro_mean_abs_r_error": sae_mean_abs_error,
    }


def make_figure(depth: pd.DataFrame, target_type: str, output_root: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = depth[depth.target_type == target_type]
    mean = (
        data.groupby(["relative_depth", "method"], as_index=False)
        .agg(
            feature_q95=("mean_feature_q95", "mean"),
            high_count=("mean_high_feature_count", "mean"),
            high_fraction=("mean_high_feature_fraction", "mean"),
            target_score=("mean_best_target_score", "mean"),
        )
    )
    colors = {
        "dense_native_768": "#3b6ea8",
        "sae_full_6144": "#d36b32",
        "sae_matched_768": "#a33f1f",
        "random_full_6144": "#49845a",
        "random_matched_768": "#78a887",
    }
    markers = {method: marker for method, marker in zip(METHODS, "osD^v")}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))
    for method in METHODS:
        values = mean[mean.method == method].sort_values("relative_depth")
        label = METHOD_LABELS[method]
        axes[0].plot(values.relative_depth, values.feature_q95, marker=markers[method], color=colors[method], label=label)
        axes[1].plot(values.relative_depth, values.high_count, marker=markers[method], color=colors[method], label=label)
        axes[2].plot(values.relative_depth, values.target_score, marker=markers[method], color=colors[method], label=label)
    metric_label = "train-oriented test AUROC" if target_type == "diagnosis" else "train-oriented test |r|"
    axes[0].set_ylabel(f"Feature-centric q95 {metric_label}")
    axes[1].set_ylabel("Features above primary threshold")
    axes[2].set_ylabel(f"Concept-centric mean {metric_label}")
    for axis in axes:
        axis.set_xlabel("Relative depth")
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(alpha=0.2)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    outputs = []
    for extension in ("png", "pdf"):
        path = output_root / f"dictionary_accessibility_{target_type}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def latex_table(models: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Target & Method & High features & High fraction & Q95 & Best-target score \\",
        r"\midrule",
    ]
    aggregate = (
        models.groupby(["target_type", "method"], as_index=False)
        .agg(
            count=("mean_high_feature_count", "mean"),
            fraction=("mean_high_feature_fraction", "mean"),
            q95=("mean_feature_q95", "mean"),
            target=("mean_best_target_score", "mean"),
        )
    )
    for target_type in ("waveform", "diagnosis"):
        for method in METHODS:
            row = aggregate[(aggregate.target_type == target_type) & (aggregate.method == method)].iloc[0]
            lines.append(
                f"{target_type.title()} & {METHOD_LABELS[method]} & {row['count']:.1f} & "
                f"{row['fraction']:.3f} & {row['q95']:.3f} & {row['target']:.3f} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    atomic_text(path, "\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    feature, target, errors, hashes = read_workers(args)
    if errors:
        audit = {
            "status": "failed",
            "audit_pass": False,
            "protocol": PROTOCOL,
            "errors": errors,
        }
        atomic_json(args.output_root / "audit.json", audit)
        raise RuntimeError("dictionary accessibility audit failed: " + "; ".join(errors))

    depth = summarize_depth(feature, target)
    models = summarize_models(depth)
    paired = paired_depth_summary(depth)
    targets = target_summary(target)
    reproduction = (
        {"status": "not_applicable_for_cohort_specific_target_panel"}
        if args.skip_calibration_reproduction
        else calibration_reproduction(target, errors)
    )
    if errors:
        audit = {
            "status": "failed",
            "audit_pass": False,
            "protocol": PROTOCOL,
            "errors": errors,
            "calibration_reproduction": reproduction,
        }
        atomic_json(args.output_root / "audit.json", audit)
        raise RuntimeError("dictionary accessibility reproduction failed: " + "; ".join(errors))
    feature.to_csv(args.output_root / "all_feature_profiles.csv", index=False)
    target.to_csv(args.output_root / "all_target_profiles.csv", index=False)
    depth.to_csv(args.output_root / "depth_method_summary.csv", index=False)
    models.to_csv(args.output_root / "model_method_summary.csv", index=False)
    paired.to_csv(args.output_root / "paired_depth_summary.csv", index=False)
    targets.to_csv(args.output_root / "model_target_summary.csv", index=False)
    figures = make_figure(depth, "waveform", args.output_root)
    figures.extend(make_figure(depth, "diagnosis", args.output_root))
    table_path = args.output_root / "paper_table_dictionary_accessibility.tex"
    latex_table(models, table_path)

    display = models.copy()
    display["method"] = display.method.map(METHOD_LABELS)
    report_lines = [
        "# Held-out dictionary accessibility benchmark",
        "",
        "- Feature-centric: each feature selects its strongest target and direction on train; test is frozen evaluation.",
        "- Target-centric: each target selects its strongest feature on train; test is frozen evaluation.",
        "- Full dictionaries and matched 768-feature budgets are both reported.",
        "- Waveform targets use Pearson r; diagnosis targets use tie-aware AUROC.",
        "- Results measure association/accessibility, not mechanism or causal use.",
        "- Dominance counts and directions are taken from the generated paired-depth tables; no cohort-independent winner is assumed.",
        "",
        markdown_table(
            display[
                [
                    "model",
                    "target_type",
                    "method",
                    "mean_high_feature_count",
                    "mean_high_feature_fraction",
                    "mean_feature_q95",
                    "mean_best_target_score",
                    "target_coverage",
                ]
            ]
        ),
        "",
    ]
    report_path = args.output_root / "report.md"
    atomic_text(report_path, "\n".join(report_lines))
    audit = {
        "status": "complete",
        "audit_pass": True,
        "protocol": PROTOCOL,
        "errors": [],
        "complete_groups": int(feature[["model", "relative_depth"]].drop_duplicates().shape[0]),
        "feature_rows": len(feature),
        "target_rows": len(target),
        "depth_method_rows": len(depth),
        "model_method_rows": len(models),
        "paired_depth_rows": len(paired),
        "test_record_sha256": next(iter(hashes)),
        "methods": list(METHODS),
        "selection_policy": "target and direction selected on train; test used once for frozen evaluation",
        "pooling_policy": "identical precomputed record-level activations for dense, SAE, and random",
        "calibration_reproduction": reproduction,
        "paper_table": str(table_path),
        "report": str(report_path),
        "figures": figures,
    }
    atomic_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
