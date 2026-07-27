#!/usr/bin/env python
"""Patient-cluster bootstrap for one model's sparse-accessibility curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.accessibility_calibration import columnwise_pearson  # noqa: E402
from benchmark_v1.sparse_accessibility import PatientClusterBootstrap  # noqa: E402
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
)


PROTOCOL = "final_layer_sparse_accessibility_e8_v2"
BOOTSTRAP_SEED = 20260718
METHODS = (
    "dense_matched",
    "pca_matched",
    "sae_matched",
    "sae_full",
    "random_matched",
    "random_full",
)
COMPARISONS = (
    ("sae_matched", "dense_matched"),
    ("sae_matched", "pca_matched"),
    ("sae_matched", "random_matched"),
    ("sae_full", "random_full"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument("--budget-replicates", type=int, default=20)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/bootstrap",
    )
    return parser.parse_args()


def bootstrap_pvalue(values: np.ndarray) -> float:
    below = (np.sum(values <= 0) + 1.0) / (len(values) + 1.0)
    above = (np.sum(values >= 0) + 1.0) / (len(values) + 1.0)
    return float(min(1.0, 2.0 * min(below, above)))


def completed_workers(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol") == PROTOCOL:
            result.append((path, payload))
    return result


def model_names(workers: list[tuple[Path, dict[str, Any]]]) -> list[str]:
    return sorted({str(payload["model"]) for _, payload in workers})


def load_prediction_sets(
    workers: list[tuple[Path, dict[str, Any]]],
    model: str,
    random_replicates: int,
    budget_replicates: int,
) -> tuple[dict[str, list[np.ndarray]], np.ndarray, np.ndarray, np.ndarray]:
    selected = [(path, payload) for path, payload in workers if payload["model"] == model]
    counts = {
        kind: sum(payload["source_kind"] == kind for _, payload in selected)
        for kind in ("dense", "pca", "sae", "random")
    }
    expected = {"dense": 1, "pca": 1, "sae": 3, "random": random_replicates}
    if counts != expected:
        raise RuntimeError(f"incomplete worker set for {model}: {counts}, expected {expected}")

    prediction_sets = {method: [] for method in METHODS}
    reference_ids = None
    reference_patients = None
    reference_targets = None
    reference_ks = None
    for summary_path, payload in selected:
        archive_path = Path(payload["test_predictions"])
        if not archive_path.exists():
            raise FileNotFoundError(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:
            ids = np.asarray(archive["test_ecg_ids"]).astype(str)
            patients = np.asarray(archive["test_patient_ids"]).astype(str)
            targets = np.asarray(archive["y_test"], dtype=np.float32)
            ks = np.asarray(archive["ks"], dtype=np.int32)
            arm_budgets = np.asarray(archive["arm_budget_replicates"], dtype=np.int32)
            predictions = np.asarray(archive["predictions"], dtype=np.float32)
        if reference_ids is None:
            reference_ids = ids
            reference_patients = patients
            reference_targets = targets
            reference_ks = ks
        elif not (
            np.array_equal(ids, reference_ids)
            and np.array_equal(patients, reference_patients)
            and np.array_equal(ks, reference_ks)
            and np.allclose(targets, reference_targets, atol=0.0, rtol=0.0)
        ):
            raise RuntimeError(f"test identity mismatch: {archive_path}")

        kind = str(payload["source_kind"])
        if kind == "dense":
            prediction_sets["dense_matched"].append(predictions[0])
        elif kind == "pca":
            prediction_sets["pca_matched"].append(predictions[0])
        elif kind == "sae":
            full = np.flatnonzero(arm_budgets < 0)
            matched = np.flatnonzero(arm_budgets >= 0)
            if len(full) != 1 or len(matched) != budget_replicates:
                raise RuntimeError(f"unexpected SAE arm count: {archive_path}")
            prediction_sets["sae_full"].append(predictions[full[0]])
            prediction_sets["sae_matched"].extend(predictions[index] for index in matched)
        else:
            full = np.flatnonzero(arm_budgets < 0)
            matched = np.flatnonzero(arm_budgets >= 0)
            if len(full) != 1 or len(matched) != 1:
                raise RuntimeError(f"unexpected random arm count: {archive_path}")
            prediction_sets["random_full"].append(predictions[full[0]])
            prediction_sets["random_matched"].append(predictions[matched[0]])
    return prediction_sets, reference_ks, reference_patients, reference_targets


def method_bootstrap(
    predictions: list[np.ndarray],
    targets: np.ndarray,
    bootstrap: PatientClusterBootstrap,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not predictions:
        raise ValueError("method has no prediction replicates")
    n_k = predictions[0].shape[0]
    draws = bootstrap.weights.shape[0]
    observed_mean = np.zeros(n_k, dtype=np.float64)
    observed_coverage = np.zeros(n_k, dtype=np.float64)
    boot_mean = np.zeros((n_k, draws), dtype=np.float64)
    boot_coverage = np.zeros((n_k, draws), dtype=np.float64)
    for values in predictions:
        if values.shape[1:] != targets.shape:
            raise ValueError("prediction replicate shape mismatch")
        for k_index in range(n_k):
            observed = np.abs(columnwise_pearson(targets, values[k_index]))
            observed_mean[k_index] += observed.mean()
            observed_coverage[k_index] += np.mean(observed >= 0.20)
            correlations = np.abs(bootstrap.correlations(values[k_index]))
            boot_mean[k_index] += correlations.mean(axis=1)
            boot_coverage[k_index] += (correlations >= 0.20).mean(axis=1)
    scale = 1.0 / len(predictions)
    return (
        observed_mean * scale,
        observed_coverage * scale,
        boot_mean * scale,
        boot_coverage * scale,
    )


def main() -> None:
    args = parse_args()
    workers = completed_workers(args.workers_root)
    names = model_names(workers)
    if len(names) != 6:
        raise RuntimeError(f"expected six models in worker outputs, found {names}")
    if not 0 <= args.model_index < len(names):
        raise IndexError(f"model index outside 0..{len(names) - 1}")
    model = names[args.model_index]
    output = args.output_root / f"model_{args.model_index:02d}_{model.lower().replace('-', '_')}"
    table_path = output / "paired_bootstrap.csv"
    archive_path = output / "method_bootstrap.npz"
    summary_path = output / "summary.json"
    if summary_path.exists() and table_path.exists() and archive_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == PROTOCOL:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    prediction_sets, ks, patient_ids, targets = load_prediction_sets(
        workers,
        model,
        args.random_replicates,
        args.budget_replicates,
    )
    bootstrap = PatientClusterBootstrap(
        patient_ids,
        targets,
        draws=args.bootstrap_draws,
        seed=BOOTSTRAP_SEED,
    )
    results = {
        method: method_bootstrap(values, targets, bootstrap)
        for method, values in prediction_sets.items()
    }
    rows = []
    for left, right in COMPARISONS:
        left_observed_mean, left_observed_coverage, left_mean, left_coverage = results[left]
        right_observed_mean, right_observed_coverage, right_mean, right_coverage = results[right]
        for k_index, k in enumerate(ks):
            delta_mean = left_mean[k_index] - right_mean[k_index]
            delta_coverage = left_coverage[k_index] - right_coverage[k_index]
            rows.append(
                {
                    "model": model,
                    "comparison": f"{left}_minus_{right}",
                    "left_method": left,
                    "right_method": right,
                    "k": int(k),
                    "observed_delta_mean_abs_r": float(
                        left_observed_mean[k_index] - right_observed_mean[k_index]
                    ),
                    "mean_delta_ci_low": float(np.quantile(delta_mean, 0.025)),
                    "mean_delta_ci_high": float(np.quantile(delta_mean, 0.975)),
                    "mean_delta_p_value": bootstrap_pvalue(delta_mean),
                    "observed_delta_coverage": float(
                        left_observed_coverage[k_index] - right_observed_coverage[k_index]
                    ),
                    "coverage_delta_ci_low": float(np.quantile(delta_coverage, 0.025)),
                    "coverage_delta_ci_high": float(np.quantile(delta_coverage, 0.975)),
                    "coverage_delta_p_value": bootstrap_pvalue(delta_coverage),
                }
            )
    atomic_csv(table_path, rows)
    arrays = {"ks": ks, "bootstrap_draws": np.asarray(args.bootstrap_draws)}
    for method, values in results.items():
        observed_mean, observed_coverage, boot_mean, boot_coverage = values
        arrays[f"{method}_observed_mean"] = observed_mean
        arrays[f"{method}_observed_coverage"] = observed_coverage
        arrays[f"{method}_bootstrap_mean"] = boot_mean
        arrays[f"{method}_bootstrap_coverage"] = boot_coverage
    atomic_npz(archive_path, **arrays)
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "model_index": args.model_index,
        "model": model,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "patients": bootstrap.n_patients,
        "test_records": bootstrap.n_records,
        "concepts": bootstrap.n_targets,
        "ks": ks.tolist(),
        "method_replicates": {
            method: len(values) for method, values in prediction_sets.items()
        },
        "paired_table": str(table_path),
        "bootstrap_archive": str(archive_path),
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
