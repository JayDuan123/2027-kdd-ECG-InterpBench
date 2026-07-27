#!/usr/bin/env python
"""Patient-cluster bootstrap for one MIMIC matched-effect model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.matched_effect import bootstrap_patient_metric_means  # noqa: E402
from benchmark_v1.mimic_matched_effect import MODEL_SPECS, PROTOCOL, safe_model_name  # noqa: E402
from scripts.run_accessibility_calibration_worker import atomic_csv, atomic_json, atomic_npz  # noqa: E402


BOOTSTRAP_SEED = 20260724
INFERENCE_METRICS = ("off_cross_rms", "wbi_cross", "off_all_rms", "activation_l2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--expected-seeds", type=int, default=3)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/bootstrap",
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


def completed_workers(root: Path, model: str, protocol: str = PROTOCOL) -> list[dict[str, object]]:
    result = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if (
            payload.get("status") == "complete"
            and payload.get("protocol") == protocol
            and payload.get("model") == model
        ):
            result.append(payload)
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.model_index < len(MODEL_SPECS):
        raise IndexError(f"model index outside 0..{len(MODEL_SPECS) - 1}")
    model = MODEL_SPECS[args.model_index][0]
    workers = completed_workers(args.workers_root, model, args.protocol)
    if len(workers) != args.expected_seeds:
        raise RuntimeError(f"expected {args.expected_seeds} workers for {model}, found {len(workers)}")
    seeds = sorted(int(payload["sae_seed"]) for payload in workers)
    if len(set(seeds)) != args.expected_seeds:
        raise RuntimeError(f"duplicate or missing seeds for {model}: {seeds}")
    output = args.output_root / f"model_{args.model_index:02d}_{safe_model_name(model)}"
    table_path = output / "paired_bootstrap.csv"
    profile_path = output / "method_profile.csv"
    archive_path = output / "bootstrap_metrics.npz"
    summary_path = output / "summary.json"
    if all(path.exists() for path in (table_path, profile_path, archive_path, summary_path)):
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == args.protocol:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    metadata = None
    metric_stack = []
    eligibility_stack = []
    effect_stack = []
    quality_passes = []
    quality_gate_modes = []
    candidate_pools = []
    for worker in sorted(workers, key=lambda value: int(value["sae_seed"])):
        with np.load(Path(worker["patient_metrics"]), allow_pickle=False) as payload:
            current = {
                key: np.asarray(payload[key])
                for key in (
                    "arm_names", "ks", "method_names", "concept_names", "concept_families",
                    "metric_names", "patient_ids", "patient_counts",
                )
            }
            if metadata is None:
                metadata = current
            else:
                for key in metadata:
                    if not np.array_equal(metadata[key], current[key]):
                        raise RuntimeError(f"worker archive axis mismatch for {key}")
            metric_stack.append(np.asarray(payload["patient_metrics"], dtype=np.float64))
            eligibility_stack.append(np.asarray(payload["eligible_design_counts"], dtype=np.int16))
            effect_stack.append(np.asarray(payload["matched_validation_effects"], dtype=np.float64))
            quality_passes.append(bool(np.asarray(payload["sae_quality_pass"]).item()))
            quality_gate_modes.append(str(worker.get("sae_quality_gate", {}).get("mode", "training_metrics")))
            candidate_pools.append(str(worker.get("candidate_pool", "all")))
    if len(set(quality_gate_modes)) != 1 or len(set(candidate_pools)) != 1:
        raise RuntimeError("worker quality-gate mode or candidate pool is inconsistent across seeds")
    assert metadata is not None
    stacked = np.stack(metric_stack)
    valid = np.isfinite(stacked[..., 0, 0])
    valid_seed_counts = valid.sum(axis=0)
    averaged = np.full(stacked.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        np.nansum(stacked, axis=0),
        valid_seed_counts[..., None, None],
        out=averaged,
        where=valid_seed_counts[..., None, None] > 0,
    )
    effects = np.stack(effect_stack)
    average_effect = np.full(effects.shape[1:], np.nan, dtype=np.float64)
    np.divide(
        np.nansum(effects, axis=0),
        valid_seed_counts,
        out=average_effect,
        where=valid_seed_counts > 0,
    )

    patient_counts = metadata["patient_counts"].astype(np.float64)
    n_patients = len(patient_counts)
    weights = np.random.default_rng(BOOTSTRAP_SEED).multinomial(
        n_patients,
        np.full(n_patients, 1.0 / n_patients),
        size=args.bootstrap_draws,
    ).astype(np.float32)
    source_metrics = metadata["metric_names"].astype(str)
    output_metric_names = tuple(source_metrics) + ("wbi_cross", "target_retention")
    observed = np.full(averaged.shape[:-2] + (len(output_metric_names),), np.nan, dtype=np.float32)
    bootstrap = np.full(
        averaged.shape[:-2] + (args.bootstrap_draws, len(output_metric_names)),
        np.nan,
        dtype=np.float32,
    )
    for index in np.ndindex(averaged.shape[:-2]):
        values = averaged[index]
        if not np.all(np.isfinite(values)):
            continue
        point = np.sum(values * patient_counts[:, None], axis=0) / np.sum(patient_counts)
        draws = bootstrap_patient_metric_means(values, patient_counts, weights)
        validation_effect = float(average_effect[index])
        point_wbi = point[1] / max(abs(point[0]), 1e-8)
        draw_wbi = draws[:, 1] / np.maximum(np.abs(draws[:, 0]), 1e-8)
        observed[index] = np.concatenate((point, [point_wbi, point[0] / validation_effect]))
        bootstrap[index] = np.column_stack((draws, draw_wbi, draws[:, 0] / validation_effect))

    arm_names = metadata["arm_names"].astype(str)
    ks = metadata["ks"].astype(int)
    method_names = metadata["method_names"].astype(str)
    concept_names = metadata["concept_names"].astype(str)
    metric_index = {name: index for index, name in enumerate(output_metric_names)}
    method_index = {name: index for index, name in enumerate(method_names)}
    all_quality_pass = all(quality_passes)
    profile_rows = []
    paired_rows = []
    for arm_index, arm in enumerate(arm_names):
        for k_index, k in enumerate(ks):
            for method_i, method in enumerate(method_names):
                values = observed[arm_index, k_index, method_i]
                any_seed = np.isfinite(values[:, 0])
                all_seed = any_seed & (
                    valid_seed_counts[arm_index, k_index, method_i] == args.expected_seeds
                )
                primary = all_seed & all_quality_pass
                for scope, selected in (
                    ("any_seed_sensitivity", any_seed),
                    ("all_seeds_effect_gate", all_seed),
                    ("quality_and_all_seeds_primary", primary),
                ):
                    row = {
                        "protocol": args.protocol,
                        "model": model,
                        "candidate_arm": arm,
                        "k": int(k),
                        "method": method,
                        "profile_scope": scope,
                        "sae_all_seed_quality_pass": all_quality_pass,
                        "concepts_total": len(concept_names),
                        "concepts_eligible": int(np.sum(selected)),
                    }
                    for metric, metric_i in metric_index.items():
                        row[metric] = float(np.mean(values[selected, metric_i])) if np.any(selected) else np.nan
                    profile_rows.append(row)
            left_i = method_index["sae"]
            right_i = method_index["dense"]
            for metric in INFERENCE_METRICS:
                metric_i = metric_index[metric]
                left = observed[arm_index, k_index, left_i, :, metric_i]
                right = observed[arm_index, k_index, right_i, :, metric_i]
                all_seed_selected = (
                    np.isfinite(left)
                    & np.isfinite(right)
                    & (valid_seed_counts[arm_index, k_index, left_i] == args.expected_seeds)
                    & (valid_seed_counts[arm_index, k_index, right_i] == args.expected_seeds)
                )
                for inference_scope, selected in (
                    ("quality_and_all_seeds_primary", all_seed_selected & all_quality_pass),
                    ("all_seeds_effect_gate_sensitivity", all_seed_selected),
                ):
                    if np.any(selected):
                        point_delta = float(np.mean(left[selected] - right[selected]))
                        differences = np.mean(
                            bootstrap[arm_index, k_index, left_i, selected, :, metric_i]
                            - bootstrap[arm_index, k_index, right_i, selected, :, metric_i],
                            axis=0,
                        )
                    else:
                        point_delta = np.nan
                        differences = np.full(args.bootstrap_draws, np.nan)
                    finite = differences[np.isfinite(differences)]
                    paired_rows.append(
                        {
                            "protocol": args.protocol,
                            "model": model,
                            "candidate_arm": arm,
                            "k": int(k),
                            "comparison": "sae_minus_dense",
                            "metric": metric,
                            "inference_scope": inference_scope,
                            "sae_all_seed_quality_pass": all_quality_pass,
                            "concepts_paired": int(np.sum(selected)),
                            "observed_delta": point_delta,
                            "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
                            "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
                            "p_value_two_sided": bootstrap_pvalue(finite),
                            "favorable_direction": "negative",
                        }
                    )

    atomic_csv(profile_path, profile_rows)
    atomic_csv(table_path, paired_rows)
    atomic_npz(
        archive_path,
        arm_names=arm_names,
        ks=ks,
        method_names=method_names,
        concept_names=concept_names,
        metric_names=np.asarray(output_metric_names),
        patient_ids=metadata["patient_ids"],
        observed=observed,
        bootstrap=bootstrap,
        valid_seed_counts=valid_seed_counts,
        matched_validation_effects=average_effect,
        sae_seed_quality_pass=np.asarray(quality_passes),
    )
    summary = {
        "status": "complete",
        "protocol": args.protocol,
        "model_index": args.model_index,
        "model": model,
        "worker_cells": len(workers),
        "seeds": seeds,
        "sae_seed_quality_pass": quality_passes,
        "sae_all_seed_quality_pass": all_quality_pass,
        "sae_quality_gate_mode": quality_gate_modes[0],
        "candidate_pool": candidate_pools[0],
        "bootstrap_draws": args.bootstrap_draws,
        "patients": n_patients,
        "concepts": len(concept_names),
        "paired_table": str(table_path),
        "method_profile": str(profile_path),
        "bootstrap_archive": str(archive_path),
        "primary_gate": "all three SAE quality gates and all three matched-effect feasibility gates",
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
