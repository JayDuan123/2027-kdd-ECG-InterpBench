#!/usr/bin/env python
"""Crossed-factor and leave-one-factor-out sensitivity analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_common import bh  # noqa: E402
from scripts.benchmark_extension_v2_common import (  # noqa: E402
    V1,
    V2,
    interval_and_p,
    stable_seed,
    write_json,
)


OUT = V2 / "hierarchical_robustness"
QUALITY_METRICS = ("recon_r2", "dead_fraction", "readout_retention_median")
STEERING_METRICS = (
    "ste",
    "otd_mean",
    "selectivity_margin",
    "wbi",
    "excess_selectivity",
    "behavior_excess",
)
PROTOCOL_METRICS = (
    "ste",
    "otd_mean",
    "selectivity_margin",
    "wbi",
    "excess_selectivity",
    "behavior_excess",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}")


def paired_method_deltas(
    frame: pd.DataFrame, keys: list[str], metrics: tuple[str, ...]
) -> pd.DataFrame:
    baseline = frame[frame.method.eq("frozen")][keys + list(metrics)].copy()
    baseline = baseline.rename(columns={metric: f"{metric}_frozen" for metric in metrics})
    rows = []
    for method in sorted(set(frame.method) - {"frozen"}):
        current = frame[frame.method.eq(method)][keys + list(metrics)].copy()
        merged = current.merge(baseline, on=keys, how="inner", validate="one_to_one")
        merged["method"] = method
        for metric in metrics:
            merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_frozen"]
        rows.append(merged)
    if not rows:
        raise RuntimeError("No non-frozen methods found")
    return pd.concat(rows, ignore_index=True)


def crossed_bootstrap(
    units: pd.DataFrame, value: str, n_bootstrap: int, seed: int
) -> np.ndarray:
    models = sorted(units.model.unique())
    cohorts = sorted(units.cohort.unique())
    matrix = units.pivot(index="model", columns="cohort", values=value).reindex(
        index=models, columns=cohorts
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    model_weights = rng.multinomial(
        len(models), np.full(len(models), 1.0 / len(models)), size=n_bootstrap
    ).astype(float)
    cohort_weights = rng.multinomial(
        len(cohorts), np.full(len(cohorts), 1.0 / len(cohorts)), size=n_bootstrap
    ).astype(float)
    finite = np.isfinite(matrix).astype(float)
    values = np.nan_to_num(matrix)
    numerator = np.einsum("bi,ij,bj->b", model_weights, values, cohort_weights)
    denominator = np.einsum("bi,ij,bj->b", model_weights, finite, cohort_weights)
    return np.divide(
        numerator,
        denominator,
        out=np.full(n_bootstrap, np.nan),
        where=denominator > 0,
    )


def factor_bayesian_bootstrap(
    units: pd.DataFrame, value: str, factors: tuple[str, ...], n_bootstrap: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    row_weights = np.ones((n_bootstrap, len(units)), dtype=float)
    for factor in factors:
        levels, inverse = np.unique(units[factor].astype(str), return_inverse=True)
        level_weights = rng.exponential(size=(n_bootstrap, len(levels)))
        row_weights *= level_weights[:, inverse]
    values = units[value].to_numpy(dtype=float)
    finite = np.isfinite(values)
    numerator = row_weights[:, finite] @ values[finite]
    denominator = row_weights[:, finite].sum(axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def filtered_unit_mean(
    raw: pd.DataFrame,
    value: str,
    dimension: str | None = None,
    held_out: str | None = None,
) -> tuple[float, int]:
    work = raw
    if dimension is not None:
        work = work[work[dimension].astype(str).ne(str(held_out))]
    units = work.groupby(["model", "cohort"], as_index=False)[value].mean()
    return float(units[value].mean()), len(units)


def transport_analysis(
    quality: pd.DataFrame, steering: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    quality_deltas = paired_method_deltas(
        quality,
        ["model", "model_suffix", "cohort", "seed"],
        QUALITY_METRICS,
    )
    steering_deltas = paired_method_deltas(
        steering,
        ["model", "model_suffix", "cohort", "seed", "target"],
        STEERING_METRICS,
    )
    inference_rows = []
    loo_rows = []
    for domain, raw, metrics in (
        ("quality", quality_deltas, QUALITY_METRICS),
        ("steering", steering_deltas, STEERING_METRICS),
    ):
        for method, method_frame in raw.groupby("method"):
            for metric in metrics:
                value = f"delta_{metric}"
                units = method_frame.groupby(["model", "cohort"], as_index=False)[value].mean()
                samples = crossed_bootstrap(
                    units,
                    value,
                    n_bootstrap,
                    stable_seed("transport", domain, method, metric),
                )
                stats = interval_and_p(samples)
                full_mean = float(units[value].mean())
                inference_rows.append(
                    {
                        "domain": domain,
                        "method": method,
                        "metric": metric,
                        "mean_delta": full_mean,
                        **stats,
                        "bootstrap_method": "crossed_model_cohort_multinomial",
                        "bootstrap_samples": n_bootstrap,
                        "model_clusters": int(units.model.nunique()),
                        "cohort_clusters": int(units.cohort.nunique()),
                        "model_cohort_units": len(units),
                    }
                )
                dimensions = ["model", "cohort"]
                if domain == "steering":
                    dimensions.append("target")
                for dimension in dimensions:
                    for level in sorted(method_frame[dimension].astype(str).unique()):
                        estimate, count = filtered_unit_mean(method_frame, value, dimension, level)
                        loo_rows.append(
                            {
                                "domain": domain,
                                "method": method,
                                "metric": metric,
                                "held_out_dimension": dimension,
                                "held_out_level": level,
                                "full_mean_delta": full_mean,
                                "leave_one_out_mean_delta": estimate,
                                "remaining_model_cohort_units": count,
                                "same_sign_as_full": bool(
                                    estimate == 0 or full_mean == 0 or np.sign(estimate) == np.sign(full_mean)
                                ),
                            }
                        )
    inference = pd.DataFrame(inference_rows)
    for domain, index in inference.groupby("domain").groups.items():
        inference.loc[index, "q_two_sided"] = bh(
            inference.loc[index, "p_two_sided"].to_numpy()
        )
    loo = pd.DataFrame(loo_rows)
    envelopes = (
        loo.groupby(["domain", "method", "metric"], as_index=False)
        .agg(
            leave_one_out_min=("leave_one_out_mean_delta", "min"),
            leave_one_out_max=("leave_one_out_mean_delta", "max"),
            leave_one_out_all_same_sign=("same_sign_as_full", "all"),
            leave_one_out_checks=("same_sign_as_full", "size"),
        )
    )
    return inference, loo, envelopes


def protocol_analysis(
    frame: pd.DataFrame, n_bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [f"delta_{metric}" for metric in PROTOCOL_METRICS]
    unit_keys = [
        "model",
        "model_suffix",
        "cohort",
        "target",
        "contrast",
        "frozen_tier0_eligible",
    ]
    units = frame.groupby(unit_keys, as_index=False)[metric_columns].mean()
    inference_rows = []
    loo_rows = []
    subsets = {
        "all_cells": units,
        "frozen_tier0_only": units[units.frozen_tier0_eligible.astype(bool)],
    }
    for subset_name, subset in subsets.items():
        for contrast, contrast_frame in subset.groupby("contrast"):
            for metric in PROTOCOL_METRICS:
                value = f"delta_{metric}"
                samples = factor_bayesian_bootstrap(
                    contrast_frame,
                    value,
                    ("model", "cohort", "target"),
                    n_bootstrap,
                    stable_seed("protocol", subset_name, contrast, metric),
                )
                stats = interval_and_p(samples)
                full_mean = float(contrast_frame[value].mean())
                inference_rows.append(
                    {
                        "subset": subset_name,
                        "contrast": contrast,
                        "metric": metric,
                        "mean_delta": full_mean,
                        **stats,
                        "bootstrap_method": "model_cohort_target_bayesian_factor_weights",
                        "bootstrap_samples": n_bootstrap,
                        "target_units": len(contrast_frame),
                        "model_clusters": int(contrast_frame.model.nunique()),
                        "cohort_clusters": int(contrast_frame.cohort.nunique()),
                        "target_clusters": int(contrast_frame.target.nunique()),
                    }
                )
                for dimension in ("model", "cohort", "target"):
                    for level in sorted(contrast_frame[dimension].astype(str).unique()):
                        remaining = contrast_frame[
                            contrast_frame[dimension].astype(str).ne(level)
                        ]
                        if not len(remaining):
                            continue
                        estimate = float(remaining[value].mean())
                        loo_rows.append(
                            {
                                "subset": subset_name,
                                "contrast": contrast,
                                "metric": metric,
                                "held_out_dimension": dimension,
                                "held_out_level": level,
                                "full_mean_delta": full_mean,
                                "leave_one_out_mean_delta": estimate,
                                "remaining_target_units": len(remaining),
                                "same_sign_as_full": bool(
                                    estimate == 0 or full_mean == 0 or np.sign(estimate) == np.sign(full_mean)
                                ),
                            }
                        )
    inference = pd.DataFrame(inference_rows)
    for subset_name, index in inference.groupby("subset").groups.items():
        inference.loc[index, "q_two_sided"] = bh(
            inference.loc[index, "p_two_sided"].to_numpy()
        )
    loo = pd.DataFrame(loo_rows)
    envelopes = (
        loo.groupby(["subset", "contrast", "metric"], as_index=False)
        .agg(
            leave_one_out_min=("leave_one_out_mean_delta", "min"),
            leave_one_out_max=("leave_one_out_mean_delta", "max"),
            leave_one_out_all_same_sign=("same_sign_as_full", "all"),
            leave_one_out_checks=("same_sign_as_full", "size"),
        )
    )
    return inference, loo, envelopes


def main() -> None:
    args = parse_args()
    quality_path = V1 / "transport_ladder" / "transport_quality_seed_cells.csv"
    steering_path = V1 / "transport_ladder" / "transport_steering_seed_cells.csv"
    protocol_path = V1 / "paired_protocols" / "paired_protocol_seed_cells.csv"
    quality = pd.read_csv(quality_path)
    steering = pd.read_csv(steering_path)
    protocol = pd.read_csv(protocol_path)
    require_columns(
        quality,
        {"model", "model_suffix", "cohort", "seed", "method", *QUALITY_METRICS},
        quality_path,
    )
    require_columns(
        steering,
        {"model", "model_suffix", "cohort", "seed", "target", "method", *STEERING_METRICS},
        steering_path,
    )
    require_columns(
        protocol,
        {
            "model",
            "model_suffix",
            "cohort",
            "seed",
            "target",
            "contrast",
            "frozen_tier0_eligible",
            *(f"delta_{metric}" for metric in PROTOCOL_METRICS),
        },
        protocol_path,
    )
    preflight = {
        "quality_rows": len(quality),
        "steering_rows": len(steering),
        "protocol_rows": len(protocol),
        "transport_models": int(quality.model.nunique()),
        "transport_cohorts": int(quality.cohort.nunique()),
        "protocol_tier0_rows": int(protocol.frozen_tier0_eligible.astype(bool).sum()),
    }
    if args.preflight_only:
        print(preflight)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    transport, transport_loo, transport_envelopes = transport_analysis(
        quality, steering, args.bootstrap
    )
    protocol_inf, protocol_loo, protocol_envelopes = protocol_analysis(
        protocol, args.bootstrap
    )
    transport.to_csv(args.out / "transport_crossed_inference.csv", index=False)
    transport_loo.to_csv(args.out / "transport_leave_one_out.csv", index=False)
    transport_envelopes.to_csv(args.out / "transport_leave_one_out_envelopes.csv", index=False)
    protocol_inf.to_csv(args.out / "protocol_factor_inference.csv", index=False)
    protocol_loo.to_csv(args.out / "protocol_leave_one_out.csv", index=False)
    protocol_envelopes.to_csv(args.out / "protocol_leave_one_out_envelopes.csv", index=False)
    metadata = {
        "schema_version": 1,
        **preflight,
        "bootstrap_samples": args.bootstrap,
        "transport_inference_rows": len(transport),
        "transport_leave_one_out_rows": len(transport_loo),
        "protocol_inference_rows": len(protocol_inf),
        "protocol_leave_one_out_rows": len(protocol_loo),
        "transport_bootstrap": "independent multinomial resampling of model and cohort factors",
        "protocol_bootstrap": "independent Bayesian weights for model, cohort, and target factors",
        "all_complete": True,
    }
    write_json(args.out / "metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
