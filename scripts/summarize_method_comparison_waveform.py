#!/usr/bin/env python
"""Summarize method-specific waveform-to-representation triangle validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_common import bh  # noqa: E402
from scripts.benchmark_extension_v2_common import interval_and_p  # noqa: E402
from scripts.method_comparison_common import BASE, METHODS, stable_seed, write_json  # noqa: E402
from scripts.summarize_waveform_triangle import MIN_CHANGE  # noqa: E402


RANDOM_COLUMNS = [f"random_{index:02d}_contribution" for index in range(20)]
PRIMARY_METRICS = (
    "raw_oriented_delta",
    "selected_oriented_delta",
    "selected_vs_random_oriented",
    "selected_attenuation_advantage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE / "waveform_triangle")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--hierarchical-bootstrap", type=int, default=10000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def merge_identity(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "model_suffix", "cohort", "phenotype", "target", "seed", "method", "record_id"]
    value_columns = [
        "measurement_value",
        "raw_target_head",
        "reconstructed_target_head",
        "selected_contribution",
        "selected_ablated_target_head",
        *RANDOM_COLUMNS,
    ]
    identity = frame[frame.variant.eq("identity")][keys + value_columns].copy()
    if identity.duplicated(keys).any():
        raise RuntimeError("Duplicate identity rows in method waveform output")
    identity = identity.rename(columns={column: f"{column}_identity" for column in value_columns})
    edited = frame[~frame.variant.eq("identity")].copy()
    merged = edited.merge(identity, on=keys, how="left", validate="many_to_one")
    if merged[[f"{column}_identity" for column in value_columns]].isna().any().any():
        raise RuntimeError("Missing identity matches")
    merged["measurement_delta"] = merged.measurement_value - merged.measurement_value_identity
    merged["raw_head_delta"] = merged.raw_target_head - merged.raw_target_head_identity
    merged["reconstructed_head_delta"] = (
        merged.reconstructed_target_head - merged.reconstructed_target_head_identity
    )
    merged["selected_contribution_delta"] = (
        merged.selected_contribution - merged.selected_contribution_identity
    )
    merged["selected_ablated_head_delta"] = (
        merged.selected_ablated_target_head - merged.selected_ablated_target_head_identity
    )
    for column in RANDOM_COLUMNS:
        merged[f"delta_{column}"] = merged[column] - merged[f"{column}_identity"]
    return merged


def aggregate_seeds(merged: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model",
        "model_suffix",
        "cohort",
        "phenotype",
        "target",
        "method",
        "reconstruction_applicable",
        "record_id",
        "variant",
        "direction_sign",
        "strength",
    ]
    value_columns = [
        "measurement_delta",
        "raw_head_delta",
        "reconstructed_head_delta",
        "selected_contribution_delta",
        "selected_ablated_head_delta",
        *(f"delta_{column}" for column in RANDOM_COLUMNS),
    ]
    records = merged.groupby(keys, as_index=False)[value_columns].mean()
    random_values = records[[f"delta_{column}" for column in RANDOM_COLUMNS]].to_numpy(dtype=float)
    raw = records.raw_head_delta.to_numpy(dtype=float)
    reconstructed = records.reconstructed_head_delta.to_numpy(dtype=float)
    selected = records.selected_contribution_delta.to_numpy(dtype=float)
    direction = records.direction_sign.to_numpy(dtype=float)
    records["raw_oriented_delta"] = direction * raw
    records["selected_oriented_delta"] = direction * selected
    records["random_oriented_delta_mean"] = direction * random_values.mean(axis=1)
    records["selected_vs_random_oriented"] = (
        records.selected_oriented_delta - records.random_oriented_delta_mean
    )
    records["selected_attenuation"] = np.abs(reconstructed) - np.abs(reconstructed - selected)
    random_attenuation = np.abs(reconstructed[:, None]) - np.abs(
        reconstructed[:, None] - random_values
    )
    records["random_attenuation_mean"] = random_attenuation.mean(axis=1)
    records["selected_attenuation_advantage"] = (
        records.selected_attenuation - records.random_attenuation_mean
    )
    records["selected_raw_sign_concordant"] = (selected * raw > 0).astype(float)
    records["random_raw_sign_concordance_mean"] = (
        random_values * raw[:, None] > 0
    ).mean(axis=1)
    records["measurement_qc_pass"] = (
        direction * records.measurement_delta.to_numpy(dtype=float)
        >= records.phenotype.map(MIN_CHANGE).to_numpy(dtype=float)
    )
    return records


def p_one_sided(samples: np.ndarray) -> float:
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    return (1.0 + float((finite <= 0).sum())) / (len(finite) + 1.0)


def summarize_cells(records: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    keys = [
        "model",
        "model_suffix",
        "cohort",
        "phenotype",
        "target",
        "method",
        "reconstruction_applicable",
        "variant",
        "direction_sign",
        "strength",
    ]
    rows = []
    for group_keys, full_group in records.groupby(keys, sort=True):
        for analysis_set, group in (
            ("unfiltered_complete_case", full_group),
            ("measurement_qc", full_group[full_group.measurement_qc_pass]),
        ):
            if not len(group):
                continue
            rng = np.random.default_rng(
                stable_seed("method-waveform-cell", *group_keys, analysis_set)
            )
            index = rng.integers(0, len(group), size=(bootstrap, len(group)))
            row = dict(zip(keys, group_keys))
            row.update(
                {
                    "analysis_set": analysis_set,
                    "attempted_records": len(full_group),
                    "analysis_records": len(group),
                    "analysis_fraction": len(group) / len(full_group),
                    "selected_raw_sign_concordance": float(
                        group.selected_raw_sign_concordant.mean()
                    ),
                    "random_raw_sign_concordance_mean": float(
                        group.random_raw_sign_concordance_mean.mean()
                    ),
                    "bootstrap_samples": bootstrap,
                }
            )
            for metric in PRIMARY_METRICS:
                values = group[metric].to_numpy(dtype=float)
                samples = values[index].mean(axis=1)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_ci_low"] = float(np.quantile(samples, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(samples, 0.975))
                row[f"{metric}_p_one_sided"] = p_one_sided(samples)
            raw_mean = row["raw_oriented_delta_mean"]
            row["aggregate_selected_share"] = (
                row["selected_oriented_delta_mean"] / raw_mean if abs(raw_mean) > 1e-8 else np.nan
            )
            rows.append(row)
    profile = pd.DataFrame(rows)
    for (analysis_set, method), indices in profile.groupby(["analysis_set", "method"]).groups.items():
        for metric in PRIMARY_METRICS:
            profile.loc[indices, f"{metric}_q"] = bh(
                profile.loc[indices, f"{metric}_p_one_sided"].to_numpy()
            )
    profile["raw_direction_pass"] = (
        profile.raw_oriented_delta_ci_low > 0
    ) & profile.raw_oriented_delta_q.lt(0.05)
    profile["selected_direction_pass"] = (
        profile.selected_oriented_delta_ci_low > 0
    ) & profile.selected_oriented_delta_q.lt(0.05)
    profile["selected_vs_random_pass"] = (
        profile.selected_vs_random_oriented_ci_low > 0
    ) & profile.selected_vs_random_oriented_q.lt(0.05)
    profile["selected_attenuation_advantage_pass"] = (
        profile.selected_attenuation_advantage_ci_low > 0
    ) & profile.selected_attenuation_advantage_q.lt(0.05)
    profile["triangle_joint_pass"] = (
        profile.raw_direction_pass
        & profile.selected_direction_pass
        & profile.selected_vs_random_pass
        & profile.selected_attenuation_advantage_pass
    )
    return profile


def crossed_bootstrap(
    frame: pd.DataFrame,
    values: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.ones((samples, len(frame)), dtype=np.float64)
    for factor in ("model", "cohort", "phenotype"):
        levels, inverse = np.unique(frame[factor].astype(str), return_inverse=True)
        factor_weights = rng.multinomial(
            len(levels), np.full(len(levels), 1.0 / len(levels)), size=samples
        )
        weights *= factor_weights[:, inverse]
    denominator = weights.sum(axis=1)
    return np.divide(
        weights @ np.asarray(values, dtype=float),
        denominator,
        out=np.full(samples, np.nan),
        where=denominator > 0,
    )


def method_inference(profile: pd.DataFrame, bootstrap: int) -> pd.DataFrame:
    metrics = (
        "selected_oriented_delta_mean",
        "selected_vs_random_oriented_mean",
        "selected_attenuation_advantage_mean",
        "selected_raw_sign_concordance",
    )
    keys = [
        "model",
        "cohort",
        "phenotype",
        "variant",
        "direction_sign",
        "strength",
        "analysis_set",
    ]
    specs = {
        "common64": ("sae_common64", [method for method in METHODS if method != "sae_common64"]),
        "existing_sae": ("sae_existing_8d", list(METHODS)),
    }
    rows = []
    for regime, (reference, methods) in specs.items():
        reference_frame = profile[profile.method.eq(reference)][keys + list(metrics)].copy()
        reference_frame = reference_frame.rename(
            columns={metric: f"{metric}_reference" for metric in metrics}
        )
        for method in methods:
            current = profile[profile.method.eq(method)].merge(
                reference_frame, on=keys, validate="one_to_one"
            )
            for analysis_set, subset in current.groupby("analysis_set"):
                for metric in metrics:
                    values = subset[metric] - subset[f"{metric}_reference"]
                    samples = crossed_bootstrap(
                        subset,
                        values.to_numpy(dtype=float),
                        bootstrap,
                        stable_seed("method-waveform-hierarchical", regime, method, analysis_set, metric),
                    )
                    stats = interval_and_p(samples)
                    rows.append(
                        {
                            "regime": regime,
                            "method": method,
                            "reference": reference,
                            "analysis_set": analysis_set,
                            "metric": metric,
                            "intervention_cells": len(subset),
                            "mean_delta": float(values.mean()),
                            **stats,
                            "bootstrap_samples": bootstrap,
                        }
                    )
    inference = pd.DataFrame(rows)
    for (regime, analysis_set, metric), indices in inference.groupby(
        ["regime", "analysis_set", "metric"]
    ).groups.items():
        inference.loc[indices, "q_two_sided"] = bh(
            inference.loc[indices, "p_two_sided"].to_numpy()
        )
    return inference


def main() -> None:
    args = parse_args()
    complete_paths = sorted((args.base / "workers").glob("*/*/*/complete.json"))
    metric_paths = sorted((args.base / "workers").glob("*/*/*/method_triangle_per_variant.csv"))
    if len(complete_paths) != 12 or len(metric_paths) != 12:
        raise RuntimeError(
            f"Method triangle workers incomplete: complete={len(complete_paths)} metrics={len(metric_paths)}"
        )
    metadata = [json.loads(path.read_text()) for path in complete_paths]
    invalid = [
        item
        for item in metadata
        if item.get("status") != "complete"
        or item.get("output_rows") != item.get("expected_output_rows")
        or item.get("waveforms_written") is not False
        or item.get("data_files_modified") is not False
    ]
    if invalid:
        raise RuntimeError(f"Method triangle audit failed: {invalid}")
    if args.preflight_only:
        print({"workers": len(metadata), "metric_files": len(metric_paths)})
        return
    frame = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    merged = merge_identity(frame)
    records = aggregate_seeds(merged)
    profile = summarize_cells(records, args.bootstrap)
    method_summary = (
        profile.groupby(["method", "analysis_set"], as_index=False)
        .agg(
            intervention_cells=("variant", "size"),
            analysis_records=("analysis_records", "sum"),
            analysis_fraction=("analysis_fraction", "mean"),
            joint_pass_cells=("triangle_joint_pass", "sum"),
            selected_raw_sign_concordance=("selected_raw_sign_concordance", "mean"),
            random_raw_sign_concordance=("random_raw_sign_concordance_mean", "mean"),
            selected_vs_random_mean=("selected_vs_random_oriented_mean", "mean"),
            attenuation_advantage_mean=("selected_attenuation_advantage_mean", "mean"),
        )
    )
    inference = method_inference(profile, args.hierarchical_bootstrap)
    args.base.mkdir(parents=True, exist_ok=True)
    records.to_csv(args.base / "method_triangle_paired_records.csv", index=False)
    profile.to_csv(args.base / "method_triangle_profile.csv", index=False)
    method_summary.to_csv(args.base / "method_triangle_summary.csv", index=False)
    inference.to_csv(args.base / "method_triangle_hierarchical_inference.csv", index=False)
    output_metadata = {
        "schema_version": 1,
        "workers": len(metadata),
        "seed_variant_rows": len(frame),
        "seed_averaged_record_rows": len(records),
        "profile_rows": len(profile),
        "summary_rows": len(method_summary),
        "hierarchical_inference_rows": len(inference),
        "record_bootstrap_samples": args.bootstrap,
        "hierarchical_bootstrap_samples": args.hierarchical_bootstrap,
        "methods": sorted(frame.method.unique().tolist()),
        "all_complete": True,
        "waveforms_written": False,
        "record_level_activations_written": False,
        "data_files_modified": False,
    }
    write_json(args.base / "metadata.json", output_metadata)
    print(method_summary.to_string(index=False))
    print(json.dumps(output_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
