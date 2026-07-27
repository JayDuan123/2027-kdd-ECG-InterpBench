#!/usr/bin/env python
"""Run one MIMIC model/SAE-seed Dense-versus-SAE intervention worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.matched_effect import (  # noqa: E402
    aggregate_patient_means,
    calibrated_record_metrics,
    centroid_unit_intervention,
    common_validation_effect,
)
from benchmark_v1.mimic_matched_effect import (  # noqa: E402
    CONCEPT_SPECS,
    PROTOCOL,
    masked_feature_concept_correlations,
    read_csv,
)
from benchmark_v1.sparse_accessibility import (  # noqa: E402
    candidate_ranking,
    deterministic_feature_subset,
    deterministic_subset_from_candidates,
)
from scripts.fit_mimic_final_layer_matched_effect_readout import (  # noqa: E402
    aligned_concept_matrix,
    limited,
)
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
    encode_sae,
    load_sae,
)


METHODS = ("dense", "sae")
METRICS = ("target_delta", "off_cross_rms", "off_all_rms", "off_max_abs", "activation_l2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/training_manifest.csv",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/derived/concepts_standardized.csv",
    )
    parser.add_argument(
        "--readout-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/readouts",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/workers",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--matched-budget", type=int, default=768)
    parser.add_argument("--budget-replicates", type=int, default=20)
    parser.add_argument("--budget-seed-base", type=int, default=976000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--effect-cap", type=float, default=0.25)
    parser.add_argument("--effect-floor", type=float, default=0.05)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--max-records-per-split", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument(
        "--candidate-pool",
        choices=("all", "train_live"),
        default="all",
        help="Target-independent SAE candidate universe used before train-correlation ranking.",
    )
    parser.add_argument(
        "--quality-gate-mode",
        choices=("training_metrics", "matched_live_capacity"),
        default="training_metrics",
    )
    parser.add_argument("--min-validation-r2", type=float, default=0.90)
    parser.add_argument("--min-validation-live-features", type=int, default=768)
    return parser.parse_args()


def feature_firing_counts(features) -> np.ndarray:
    """Return per-feature nonzero counts for dense or scipy sparse matrices."""
    if hasattr(features, "getnnz"):
        counts = features.getnnz(axis=0)
    else:
        counts = np.count_nonzero(np.asarray(features), axis=0)
    return np.asarray(counts, dtype=np.int64).reshape(-1)


def matched_live_capacity_quality(
    metrics: dict[str, Any], min_r2: float, min_live_features: int
) -> dict[str, Any]:
    """Evaluate a capacity-aligned gate without changing stored SAE metrics."""
    n_features = int(metrics["N"])
    validation = metrics["validation"]
    live_features = int(round(n_features * (1.0 - float(validation["dead_fraction"]))))
    reconstruction_pass = bool(float(validation["reconstruction_r2"]) >= min_r2)
    live_capacity_pass = bool(live_features >= min_live_features)
    return {
        "mode": "matched_live_capacity",
        "description": (
            f"validation reconstruction R2 >= {min_r2:g} and validation live atoms >= "
            f"{min_live_features}; dead fraction remains descriptive"
        ),
        "min_validation_reconstruction_r2": min_r2,
        "min_validation_live_features": min_live_features,
        "validation_live_features": live_features,
        "reconstruction_pass": reconstruction_pass,
        "live_capacity_pass": live_capacity_pass,
        "pass": bool(reconstruction_pass and live_capacity_pass),
    }


def readout_path(root: Path, model_safe: str, layer: int) -> Path:
    matches = sorted(root.glob(f"model_*_{model_safe}_layer{layer:02d}/readout.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one readout for {model_safe} layer {layer}, found {matches}")
    return matches[0]


def point_row(
    base: dict[str, Any],
    status: str,
    unit_effect: float,
    common_effect: float,
    alpha: float,
    metrics: np.ndarray | None,
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "status": status,
            "unit_validation_target_effect": unit_effect,
            "matched_validation_target_effect": common_effect,
            "alpha": alpha,
            "test_target_delta": np.nan,
            "test_off_cross_rms": np.nan,
            "test_off_all_rms": np.nan,
            "test_off_max_abs": np.nan,
            "test_activation_l2": np.nan,
            "test_wbi_cross": np.nan,
        }
    )
    if metrics is not None and len(metrics):
        means = np.mean(metrics, axis=0)
        row.update(
            {
                "test_target_delta": float(means[0]),
                "test_off_cross_rms": float(means[1]),
                "test_off_all_rms": float(means[2]),
                "test_off_max_abs": float(means[3]),
                "test_activation_l2": float(means[4]),
                "test_wbi_cross": float(means[1] / max(abs(means[0]), 1e-8)),
            }
        )
    return row


def aggregate_with_reference(
    values: np.ndarray,
    patient_ids: np.ndarray,
    reference_patients: np.ndarray,
    reference_counts: np.ndarray,
) -> np.ndarray:
    patients, means, counts = aggregate_patient_means(values, patient_ids)
    if not np.array_equal(patients, reference_patients) or not np.array_equal(counts, reference_counts):
        raise RuntimeError("patient aggregation identity mismatch")
    return means


def main() -> None:
    args = parse_args()
    if args.matched_budget != 768 or args.k != 5 or args.budget_replicates != 20:
        raise ValueError("locked protocol requires budget=768, k=5, and 20 subsets")
    manifest = sorted(read_csv(args.manifest), key=lambda item: int(item["task_index"]))
    matches = [row for row in manifest if int(row["task_index"]) == args.task_index]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for task {args.task_index}")
    row = matches[0]
    seed = int(row["seed"])
    model_safe = row["model_safe"]
    layer = int(row["layer"])
    output = args.output_root / f"task_{args.task_index:02d}_{model_safe}_layer{layer:02d}_seed{seed}"
    table_path = output / "design_cells.csv"
    archive_path = output / "patient_metrics.npz"
    summary_path = output / "summary.json"
    if table_path.exists() and archive_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == args.protocol:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    activations = np.load(Path(row["activation_path"]), mmap_mode="r")
    records = read_csv(Path(row["records_path"]))
    concepts, concept_names = aligned_concept_matrix(records, args.concepts)
    concept_families = np.asarray([family for _name, family in CONCEPT_SPECS])
    if concept_names != [name for name, _family in CONCEPT_SPECS]:
        raise RuntimeError("concept registry mismatch")
    splits = np.asarray([item["split"] for item in records])
    indices = {
        "train": limited(np.flatnonzero(splits == "train"), args.max_records_per_split),
        "val": limited(np.flatnonzero(splits == "val"), args.max_records_per_split),
        "test": limited(np.flatnonzero(splits == "test"), args.max_records_per_split),
    }
    model, checkpoint_path = load_sae(row, args.device)
    mean = model.mu.detach().cpu().numpy().astype(np.float32)
    scale = model.sigma.detach().cpu().numpy().astype(np.float32)

    def normalized(selected: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(activations[selected], dtype=np.float32) - mean) / scale
        ).astype(np.float32)

    x_train = normalized(indices["train"])
    x_validation = normalized(indices["val"])
    x_test = normalized(indices["test"])
    y_train = concepts[indices["train"]]
    readout_archive = readout_path(args.readout_root, model_safe, layer)
    with np.load(readout_archive, allow_pickle=False) as payload:
        if not np.array_equal(payload["concept_names"].astype(str), np.asarray(concept_names)):
            raise RuntimeError("readout concept identity mismatch")
        if max(
            float(np.max(np.abs(payload["activation_mean"] - mean))),
            float(np.max(np.abs(payload["activation_scale"] - scale))),
        ) > 1e-5:
            raise RuntimeError("SAE and readout training normalization mismatch")
        readout_coefficients = np.asarray(payload["coefficients"], dtype=np.float32)

    sae_train = encode_sae(model, activations, indices["train"], args.batch_size, args.device)
    sae_validation = encode_sae(model, activations, indices["val"], args.batch_size, args.device)
    sae_test = encode_sae(model, activations, indices["test"], args.batch_size, args.device)
    sae_basis = model.W_dec.detach().cpu().numpy().T.astype(np.float32)
    if sae_basis.shape != (6144, 768):
        raise RuntimeError(f"unexpected SAE decoder shape: {sae_basis.shape}")
    train_firing_counts = feature_firing_counts(sae_train)
    train_live_features = np.flatnonzero(train_firing_counts > 0).astype(np.int64)
    if args.candidate_pool == "train_live":
        candidate_pool = train_live_features
    else:
        candidate_pool = np.arange(sae_basis.shape[0], dtype=np.int64)
    if len(candidate_pool) < args.matched_budget:
        raise RuntimeError(
            f"candidate pool has {len(candidate_pool)} features, fewer than budget {args.matched_budget}"
        )

    dense_ranking = candidate_ranking(masked_feature_concept_correlations(x_train, y_train))
    sae_correlations = masked_feature_concept_correlations(sae_train, y_train)
    dense_units = {}
    for target_index in range(len(concept_names)):
        dense_units[target_index] = centroid_unit_intervention(
            x_train,
            x_validation,
            x_test,
            y_train,
            dense_ranking,
            np.eye(768, dtype=np.float32),
            readout_coefficients,
            target_index=target_index,
            k=args.k,
            high_quantile=args.high_quantile,
        )

    test_patient_ids = np.asarray([records[index]["patient_id"] for index in indices["test"]])
    reference_patients, _, reference_counts = aggregate_patient_means(
        np.zeros((len(indices["test"]), 1), dtype=np.float32), test_patient_ids
    )
    shape = (1, 1, len(METHODS), len(concept_names), len(reference_patients), len(METRICS))
    metric_sums = np.zeros(shape, dtype=np.float64)
    eligible_counts = np.zeros(shape[:-2], dtype=np.int16)
    effect_sums = np.zeros(shape[:-2], dtype=np.float64)
    design_rows: list[dict[str, Any]] = []

    quality = json.loads(Path(row["metrics"]).read_text())
    if args.quality_gate_mode == "matched_live_capacity":
        quality_gate = matched_live_capacity_quality(
            quality, args.min_validation_r2, args.min_validation_live_features
        )
    else:
        quality_gate = {
            **quality["quality_gate"],
            "mode": "training_metrics",
            "description": "stored SAE training quality gate",
            "validation_live_features": int(
                round(int(quality["N"]) * (1.0 - float(quality["validation"]["dead_fraction"])))
            ),
        }
    quality_pass = bool(quality_gate["pass"])
    for replicate in range(args.budget_replicates):
        if args.candidate_pool == "train_live":
            candidates = deterministic_subset_from_candidates(
                candidate_pool, args.matched_budget, args.budget_seed_base + replicate
            )
        else:
            candidates = deterministic_feature_subset(
                sae_basis.shape[0], args.matched_budget, args.budget_seed_base + replicate
            )
        sae_ranking = candidate_ranking(sae_correlations, candidates)
        for target_index, concept in enumerate(concept_names):
            sae_unit = centroid_unit_intervention(
                sae_train,
                sae_validation,
                sae_test,
                y_train,
                sae_ranking,
                sae_basis,
                readout_coefficients,
                target_index=target_index,
                k=args.k,
                high_quantile=args.high_quantile,
            )
            units = {"dense": dense_units[target_index], "sae": sae_unit}
            common, unit_effects, common_status = common_validation_effect(
                units,
                target_index,
                cap=args.effect_cap,
                floor=args.effect_floor,
                max_alpha=args.max_alpha,
            )
            for method_index, method in enumerate(METHODS):
                base = {
                    "protocol": args.protocol,
                    "model": row["model"],
                    "model_safe": model_safe,
                    "layer": layer,
                    "sae_seed": seed,
                    "sae_quality_pass": quality_pass,
                    "candidate_arm": (
                        "live_atom_matched_768"
                        if args.candidate_pool == "train_live"
                        else "matched_768"
                    ),
                    "candidate_pool": args.candidate_pool,
                    "train_live_features": len(train_live_features),
                    "budget_replicate": replicate,
                    "candidate_count": 768,
                    "k": args.k,
                    "concept": concept,
                    "family": concept_families[target_index],
                    "method": method,
                    "selected_features": ";".join(
                        str(value) for value in units[method].selected_features
                    ),
                }
                unit_effect = unit_effects.get(
                    method, units[method].validation_target_effect(target_index)
                )
                if common_status != "eligible":
                    design_rows.append(point_row(base, common_status, unit_effect, common, np.nan, None))
                    continue
                metrics, alpha, status = calibrated_record_metrics(
                    units[method],
                    target_index=target_index,
                    target_family=str(concept_families[target_index]),
                    concept_families=concept_families,
                    target_effect=common,
                    unit_validation_effect=unit_effect,
                    max_alpha=args.max_alpha,
                )
                design_rows.append(point_row(base, status, unit_effect, common, alpha, metrics))
                if status != "eligible":
                    continue
                patient_metrics = aggregate_with_reference(
                    metrics, test_patient_ids, reference_patients, reference_counts
                )
                metric_sums[0, 0, method_index, target_index] += patient_metrics
                eligible_counts[0, 0, method_index, target_index] += 1
                effect_sums[0, 0, method_index, target_index] += common

    patient_metrics = np.full(shape, np.nan, dtype=np.float32)
    matched_effects = np.full(shape[:-2], np.nan, dtype=np.float32)
    for index in np.ndindex(eligible_counts.shape):
        count = int(eligible_counts[index])
        if count:
            patient_metrics[index] = (metric_sums[index] / count).astype(np.float32)
            matched_effects[index] = effect_sums[index] / count
    atomic_csv(table_path, design_rows)
    atomic_npz(
        archive_path,
        arm_names=np.asarray(
            ["live_atom_matched_768" if args.candidate_pool == "train_live" else "matched_768"]
        ),
        ks=np.asarray([args.k], dtype=np.int32),
        method_names=np.asarray(METHODS),
        concept_names=np.asarray(concept_names),
        concept_families=concept_families,
        metric_names=np.asarray(METRICS),
        patient_ids=reference_patients,
        patient_counts=reference_counts,
        patient_metrics=patient_metrics,
        eligible_design_counts=eligible_counts,
        matched_validation_effects=matched_effects,
        sae_quality_pass=np.asarray(quality_pass),
        train_live_feature_indices=train_live_features,
        train_feature_firing_counts=train_firing_counts,
        validation_live_features=np.asarray(quality_gate["validation_live_features"]),
    )
    summary = {
        "status": "complete",
        "protocol": args.protocol,
        "task_index": args.task_index,
        "sae_seed": seed,
        "model": row["model"],
        "model_safe": model_safe,
        "layer": layer,
        "checkpoint": str(checkpoint_path),
        "sae_quality_pass": quality_pass,
        "sae_quality_gate": quality_gate,
        "sae_validation_quality": quality["validation"],
        "readout": str(readout_archive),
        "n_train": len(indices["train"]),
        "n_validation": len(indices["val"]),
        "n_test": len(indices["test"]),
        "n_test_patients": len(reference_patients),
        "concepts": len(concept_names),
        "methods": list(METHODS),
        "budget_replicates": args.budget_replicates,
        "candidate_pool": args.candidate_pool,
        "candidate_pool_features": int(len(candidate_pool)),
        "train_live_features": int(len(train_live_features)),
        "train_firing_count_min_live": int(train_firing_counts[train_live_features].min()),
        "train_firing_count_median_live": float(np.median(train_firing_counts[train_live_features])),
        "design_rows": len(design_rows),
        "eligible_design_rows": int(sum(value["status"] == "eligible" for value in design_rows)),
        "effect_cap": args.effect_cap,
        "effect_floor": args.effect_floor,
        "max_alpha": args.max_alpha,
        "common_effect_methods": list(METHODS),
        "missing_label_policy": "finite labels only for train selection and centroids",
        "design_cells": str(table_path),
        "patient_metrics": str(archive_path),
        "smoke": args.max_records_per_split > 0 or int(row["steps"]) < 8000,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
