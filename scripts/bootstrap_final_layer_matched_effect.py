#!/usr/bin/env python
"""Patient-cluster bootstrap for one model's matched-effect interventions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.matched_effect import bootstrap_patient_metric_means  # noqa: E402
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
)


PROTOCOL = "final_layer_matched_effect_v1"
BOOTSTRAP_SEED = 20260722
COMPARISONS = (("sae", "dense"), ("sae", "pca"), ("sae", "random_rotation"))
INFERENCE_METRICS = ("off_cross_rms", "wbi_cross", "off_all_rms", "activation_l2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--expected-models", type=int, default=6)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/bootstrap",
    )
    return parser.parse_args()


def bootstrap_pvalue(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan")
    below = (np.sum(finite <= 0) + 1.0) / (len(finite) + 1.0)
    above = (np.sum(finite >= 0) + 1.0) / (len(finite) + 1.0)
    return float(min(1.0, 2.0 * min(below, above)))


def completed_workers(root: Path) -> list[tuple[Path, dict]]:
    result = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol") == PROTOCOL:
            result.append((path, payload))
    return result


def average_seed_metrics(
    archives: list[Path],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = None
    metrics = []
    eligibility = []
    effects = []
    for path in archives:
        with np.load(path, allow_pickle=False) as payload:
            current = {
                key: np.asarray(payload[key])
                for key in (
                    "arm_names",
                    "ks",
                    "method_names",
                    "concept_names",
                    "concept_families",
                    "metric_names",
                    "patient_ids",
                    "patient_counts",
                )
            }
            if metadata is None:
                metadata = current
            else:
                for key in metadata:
                    if not np.array_equal(current[key], metadata[key]):
                        raise RuntimeError(f"worker archive axis mismatch for {key}: {path}")
            metrics.append(np.asarray(payload["patient_metrics"], dtype=np.float64))
            eligibility.append(np.asarray(payload["eligible_design_counts"], dtype=np.int16))
            effects.append(np.asarray(payload["matched_validation_effects"], dtype=np.float64))
    assert metadata is not None
    stacked = np.stack(metrics)
    valid = np.isfinite(stacked[..., 0, 0])
    count = valid.sum(axis=0)
    summed = np.nansum(stacked, axis=0)
    averaged = np.full_like(summed, np.nan)
    np.divide(
        summed,
        count[..., None, None],
        out=averaged,
        where=count[..., None, None] > 0,
    )
    effect_stack = np.stack(effects)
    effect_sum = np.nansum(effect_stack, axis=0)
    average_effect = np.full_like(effect_sum, np.nan)
    np.divide(effect_sum, count, out=average_effect, where=count > 0)
    return metadata, averaged, count, np.stack(eligibility), average_effect


def main() -> None:
    args = parse_args()
    workers = completed_workers(args.workers_root)
    models = sorted({payload["model"] for _, payload in workers})
    if len(models) != args.expected_models:
        raise RuntimeError(f"expected {args.expected_models} models, found {models}")
    if not 0 <= args.model_index < len(models):
        raise IndexError(f"model index outside 0..{len(models) - 1}")
    model = models[args.model_index]
    selected = [(path, payload) for path, payload in workers if payload["model"] == model]
    if len(selected) != args.expected_seeds:
        raise RuntimeError(
            f"expected {args.expected_seeds} workers for {model}, found {len(selected)}"
        )
    if len({int(payload["source_seed"]) for _, payload in selected}) != args.expected_seeds:
        raise RuntimeError(f"duplicate or missing SAE seeds for {model}")

    output = args.output_root / f"model_{args.model_index:02d}_{model.lower().replace('-', '_')}"
    table_path = output / "paired_bootstrap.csv"
    profile_path = output / "method_profile.csv"
    archive_path = output / "bootstrap_metrics.npz"
    summary_path = output / "summary.json"
    if all(path.exists() for path in (table_path, profile_path, archive_path, summary_path)):
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == PROTOCOL:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    metadata, patient_metrics, valid_seed_counts, design_counts, matched_effects = average_seed_metrics(
        [Path(payload["patient_metrics"]) for _, payload in selected]
    )
    arm_names = metadata["arm_names"].astype(str)
    ks = metadata["ks"].astype(int)
    method_names = metadata["method_names"].astype(str)
    concept_names = metadata["concept_names"].astype(str)
    source_metrics = metadata["metric_names"].astype(str)
    patient_ids = metadata["patient_ids"].astype(str)
    patient_counts = metadata["patient_counts"].astype(np.float64)
    n_patients = len(patient_ids)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        n_patients,
        np.full(n_patients, 1.0 / n_patients),
        size=args.bootstrap_draws,
    ).astype(np.float32)
    output_metric_names = tuple(source_metrics) + ("wbi_cross", "target_retention")
    observed = np.full(patient_metrics.shape[:-2] + (len(output_metric_names),), np.nan, dtype=np.float32)
    bootstrap = np.full(
        patient_metrics.shape[:-2] + (args.bootstrap_draws, len(output_metric_names)),
        np.nan,
        dtype=np.float32,
    )
    for index in np.ndindex(patient_metrics.shape[:-2]):
        values = patient_metrics[index]
        if not np.all(np.isfinite(values)):
            continue
        point = np.sum(values * patient_counts[:, None], axis=0) / np.sum(patient_counts)
        draws = bootstrap_patient_metric_means(values, patient_counts, weights)
        point_wbi = point[1] / max(abs(point[0]), 1e-8)
        draw_wbi = draws[:, 1] / np.maximum(np.abs(draws[:, 0]), 1e-8)
        validation_effect = float(matched_effects[index])
        point_retention = point[0] / validation_effect
        draw_retention = draws[:, 0] / validation_effect
        observed[index] = np.concatenate(
            (point, [point_wbi, point_retention])
        ).astype(np.float32)
        bootstrap[index] = np.column_stack(
            (draws, draw_wbi, draw_retention)
        ).astype(np.float32)

    metric_index = {name: index for index, name in enumerate(output_metric_names)}
    method_index = {name: index for index, name in enumerate(method_names)}
    rows = []
    profile_rows = []
    for arm_index, arm in enumerate(arm_names):
        for k_index, k in enumerate(ks):
            for method_i, method in enumerate(method_names):
                values = observed[arm_index, k_index, method_i]
                finite_targets = np.isfinite(values[:, 0])
                all_seed_targets = finite_targets & (
                    valid_seed_counts[arm_index, k_index, method_i]
                    == args.expected_seeds
                )
                for scope, valid_targets in (
                    ("any_seed_sensitivity", finite_targets),
                    ("all_seeds_primary", all_seed_targets),
                ):
                    profile = {
                        "protocol": PROTOCOL,
                        "model": model,
                        "candidate_arm": arm,
                        "k": int(k),
                        "method": method,
                        "profile_scope": scope,
                        "concepts_total": len(concept_names),
                        "concepts_eligible": int(np.sum(valid_targets)),
                        "concepts_any_seed_eligible": int(np.sum(finite_targets)),
                        "concepts_all_seeds_eligible": int(np.sum(all_seed_targets)),
                    }
                    for metric, metric_i in metric_index.items():
                        profile[metric] = (
                            float(np.mean(values[valid_targets, metric_i]))
                            if np.any(valid_targets)
                            else np.nan
                        )
                    profile_rows.append(profile)

            for left, right in COMPARISONS:
                left_i = method_index[left]
                right_i = method_index[right]
                for metric in INFERENCE_METRICS:
                    metric_i = metric_index[metric]
                    left_point = observed[arm_index, k_index, left_i, :, metric_i]
                    right_point = observed[arm_index, k_index, right_i, :, metric_i]
                    valid_targets = (
                        np.isfinite(left_point)
                        & np.isfinite(right_point)
                        & (
                            valid_seed_counts[arm_index, k_index, left_i]
                            == args.expected_seeds
                        )
                        & (
                            valid_seed_counts[arm_index, k_index, right_i]
                            == args.expected_seeds
                        )
                    )
                    if not np.any(valid_targets):
                        point_delta = np.nan
                        difference = np.full(args.bootstrap_draws, np.nan)
                    else:
                        point_delta = float(np.mean(left_point[valid_targets] - right_point[valid_targets]))
                        difference = np.nanmean(
                            bootstrap[arm_index, k_index, left_i, valid_targets, :, metric_i]
                            - bootstrap[arm_index, k_index, right_i, valid_targets, :, metric_i],
                            axis=0,
                        )
                    finite = difference[np.isfinite(difference)]
                    rows.append(
                        {
                            "protocol": PROTOCOL,
                            "model": model,
                            "candidate_arm": arm,
                            "k": int(k),
                            "comparison": f"{left}_minus_{right}",
                            "left_method": left,
                            "right_method": right,
                            "metric": metric,
                            "concepts_paired": int(np.sum(valid_targets)),
                            "seed_eligibility_gate": f"all_{args.expected_seeds}_seeds",
                            "observed_delta": point_delta,
                            "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                            "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                            "p_value_two_sided": bootstrap_pvalue(finite),
                            "favorable_direction": "negative",
                        }
                    )

    atomic_csv(table_path, rows)
    atomic_csv(profile_path, profile_rows)
    atomic_npz(
        archive_path,
        arm_names=arm_names,
        ks=ks,
        method_names=method_names,
        concept_names=concept_names,
        metric_names=np.asarray(output_metric_names),
        patient_ids=patient_ids,
        observed=observed,
        bootstrap=bootstrap,
        valid_seed_counts=valid_seed_counts,
        design_eligible_counts=design_counts,
        matched_validation_effects=matched_effects,
    )
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "model_index": args.model_index,
        "model": model,
        "worker_cells": len(selected),
        "seeds": sorted(int(payload["source_seed"]) for _, payload in selected),
        "bootstrap_draws": args.bootstrap_draws,
        "patients": n_patients,
        "concepts": len(concept_names),
        "paired_table": str(table_path),
        "method_profile": str(profile_path),
        "bootstrap_archive": str(archive_path),
        "inference_unit": "patient-cluster bootstrap after fixed-subset and SAE-seed averaging",
        "seed_eligibility_gate": f"all_{args.expected_seeds}_seeds for primary paired inference",
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
