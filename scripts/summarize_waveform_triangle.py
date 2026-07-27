#!/usr/bin/env python
"""Record-paired summary of waveform/SAE/readout triangle experiments."""
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
from scripts.benchmark_extension_v2_common import V2, stable_seed, write_json  # noqa: E402


BASE = V2 / "waveform_triangle"
MIN_CHANGE = {"rr_irregularity": 0.005, "qrs_duration": 5.0, "qt_interval": 10.0}
RANDOM_COLUMNS = tuple(f"random_contribution_{index:02d}" for index in range(20))
PRIMARY_METRICS = (
    "raw_oriented_delta",
    "selected_oriented_delta",
    "selected_vs_random_oriented",
    "selected_attenuation_advantage",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=5_000)
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def p_one_sided(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    return float((1.0 + (values <= 0).sum()) / (len(values) + 1.0))


def merge_identity(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "model_suffix", "cohort", "phenotype", "target", "seed", "record_id"]
    value_columns = [
        "measurement_value",
        "raw_target_head",
        "sae_reconstructed_target_head",
        "selected_top5_contribution",
        "selected_top5_ablated_target_head",
        *RANDOM_COLUMNS,
    ]
    identity = frame[frame.variant.eq("identity")][keys + value_columns].copy()
    edited = frame[~frame.variant.eq("identity")].copy()
    merged = edited.merge(
        identity,
        on=keys,
        suffixes=("", "_identity"),
        validate="many_to_one",
    )
    merged["measurement_delta"] = (
        merged.measurement_value - merged.measurement_value_identity
    )
    merged["raw_head_delta"] = merged.raw_target_head - merged.raw_target_head_identity
    merged["reconstructed_head_delta"] = (
        merged.sae_reconstructed_target_head
        - merged.sae_reconstructed_target_head_identity
    )
    merged["selected_contribution_delta"] = (
        merged.selected_top5_contribution
        - merged.selected_top5_contribution_identity
    )
    merged["selected_ablated_head_delta"] = (
        merged.selected_top5_ablated_target_head
        - merged.selected_top5_ablated_target_head_identity
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
    random_delta_columns = [f"delta_{column}" for column in RANDOM_COLUMNS]
    random_values = records[random_delta_columns].to_numpy(dtype=float)
    reconstructed = records.reconstructed_head_delta.to_numpy(dtype=float)
    selected = records.selected_contribution_delta.to_numpy(dtype=float)
    raw = records.raw_head_delta.to_numpy(dtype=float)
    direction = records.direction_sign.to_numpy(dtype=float)
    records["raw_oriented_delta"] = direction * raw
    records["selected_oriented_delta"] = direction * selected
    records["random_oriented_delta_mean"] = direction * random_values.mean(axis=1)
    records["selected_vs_random_oriented"] = (
        records.selected_oriented_delta - records.random_oriented_delta_mean
    )
    records["selected_attenuation"] = np.abs(reconstructed) - np.abs(
        reconstructed - selected
    )
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


def summarize_cells(records: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    keys = [
        "model",
        "cohort",
        "phenotype",
        "target",
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
                stable_seed("triangle-summary", *group_keys, analysis_set)
            )
            index = rng.integers(0, len(group), size=(n_bootstrap, len(group)))
            row = dict(zip(keys, group_keys))
            row.update(
                {
                    "analysis_set": analysis_set,
                    "attempted_records": len(full_group),
                    "analysis_records": len(group),
                    "analysis_fraction": len(group) / len(full_group),
                    "bootstrap_samples": n_bootstrap,
                    "selected_raw_sign_concordance": float(
                        group.selected_raw_sign_concordant.mean()
                    ),
                    "random_raw_sign_concordance_mean": float(
                        group.random_raw_sign_concordance_mean.mean()
                    ),
                }
            )
            samples_by_metric = {}
            for metric in PRIMARY_METRICS:
                values = group[metric].to_numpy(dtype=float)
                samples = values[index].mean(axis=1)
                samples_by_metric[metric] = samples
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_ci_low"] = float(np.quantile(samples, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(samples, 0.975))
                row[f"{metric}_p_one_sided"] = p_one_sided(samples)
            raw_samples = samples_by_metric["raw_oriented_delta"]
            selected_samples = samples_by_metric["selected_oriented_delta"]
            valid_ratio = np.abs(raw_samples) > 1e-8
            ratio_samples = selected_samples[valid_ratio] / raw_samples[valid_ratio]
            raw_mean = row["raw_oriented_delta_mean"]
            row["aggregate_selected_share"] = (
                row["selected_oriented_delta_mean"] / raw_mean
                if abs(raw_mean) > 1e-8
                else np.nan
            )
            row["aggregate_selected_share_ci_low"] = (
                float(np.quantile(ratio_samples, 0.025)) if len(ratio_samples) else np.nan
            )
            row["aggregate_selected_share_ci_high"] = (
                float(np.quantile(ratio_samples, 0.975)) if len(ratio_samples) else np.nan
            )
            rows.append(row)
    profile = pd.DataFrame(rows)
    for analysis_set, indices in profile.groupby("analysis_set").groups.items():
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


def main() -> None:
    args = parse_args()
    complete_paths = sorted((args.base / "workers").glob("*/*/*/complete.json"))
    metric_paths = sorted((args.base / "workers").glob("*/*/*/triangle_per_variant.csv"))
    if len(complete_paths) != 12 or len(metric_paths) != 12:
        raise RuntimeError(
            f"Triangle workers incomplete: complete={len(complete_paths)} metrics={len(metric_paths)}"
        )
    worker_metadata = [json.loads(path.read_text()) for path in complete_paths]
    invalid = [
        item
        for item in worker_metadata
        if item.get("status") != "complete"
        or item.get("output_rows") != item.get("expected_output_rows")
        or item.get("waveforms_written") is not False
    ]
    if invalid:
        raise RuntimeError(f"Triangle worker audit failed: {invalid}")
    if args.preflight_only:
        print({"workers": len(worker_metadata), "metric_files": len(metric_paths)})
        return

    frame = pd.concat([pd.read_csv(path) for path in metric_paths], ignore_index=True)
    merged = merge_identity(frame)
    records = aggregate_seeds(merged)
    profile = summarize_cells(records, args.bootstrap)
    summary = (
        profile.groupby(["model", "phenotype", "analysis_set"], as_index=False)
        .agg(
            intervention_cells=("variant", "size"),
            analysis_records=("analysis_records", "sum"),
            analysis_fraction=("analysis_fraction", "mean"),
            raw_direction_pass=("raw_direction_pass", "sum"),
            selected_direction_pass=("selected_direction_pass", "sum"),
            selected_vs_random_pass=("selected_vs_random_pass", "sum"),
            selected_attenuation_advantage_pass=(
                "selected_attenuation_advantage_pass",
                "sum",
            ),
            triangle_joint_pass=("triangle_joint_pass", "sum"),
            selected_raw_sign_concordance=("selected_raw_sign_concordance", "mean"),
            random_raw_sign_concordance_mean=(
                "random_raw_sign_concordance_mean",
                "mean",
            ),
        )
    )
    records.to_csv(args.base / "triangle_paired_records.csv", index=False)
    profile.to_csv(args.base / "triangle_profile.csv", index=False)
    summary.to_csv(args.base / "triangle_summary.csv", index=False)
    pd.DataFrame(worker_metadata).to_csv(args.base / "triangle_worker_audit.csv", index=False)
    metadata = {
        "schema_version": 1,
        "workers": len(worker_metadata),
        "raw_seed_variant_rows": len(frame),
        "seed_level_edited_rows": len(merged),
        "seed_averaged_paired_records": len(records),
        "profile_rows": len(profile),
        "bootstrap_samples": args.bootstrap,
        "random_controls_per_seed": 20,
        "primary_unit": "record after averaging the three SAE seeds",
        "analysis_sets": ["unfiltered_complete_case", "measurement_qc"],
        "triangle_definition": "waveform response, selected-latent contribution, and readout attenuation versus matched random top-5 controls",
        "claim_boundary": "readout mediation sensitivity, not biological or clinical causality",
        "waveforms_written": False,
        "all_complete": True,
    }
    write_json(args.base / "metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
