#!/usr/bin/env python
"""Run one model/SAE-seed validation-matched intervention worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.accessibility_calibration import feature_concept_correlations  # noqa: E402
from benchmark_v1.matched_effect import (  # noqa: E402
    aggregate_patient_means,
    calibrated_record_metrics,
    centroid_unit_intervention,
    common_validation_effect,
)
from benchmark_v1.multiscale_sae import read_csv, standardized_concepts  # noqa: E402
from benchmark_v1.sparse_accessibility import (  # noqa: E402
    candidate_ranking,
    deterministic_feature_subset,
)
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
    encode_sae,
    load_sae,
    resolved,
)
from scripts.run_final_layer_sparse_accessibility_worker import (  # noqa: E402
    checkpoint_normalization,
    final_groups,
    limited,
    normalize_rows,
    parse_csv_numbers,
    pca_rows,
    source_identity,
)


PROTOCOL = "final_layer_matched_effect_v1"
METHODS = ("dense", "pca", "sae", "random_rotation")
ARMS = ("matched_768", "full_6144")
METRICS = ("target_delta", "off_cross_rms", "off_all_rms", "off_max_abs", "activation_l2")
KS = (1, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument("--matched-budget", type=int, default=768)
    parser.add_argument("--budget-replicates", type=int, default=20)
    parser.add_argument("--budget-seed-base", type=int, default=960000)
    parser.add_argument("--ks", type=lambda x: parse_csv_numbers(x, int), default=KS)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--effect-cap", type=float, default=0.25)
    parser.add_argument("--effect-floor", type=float, default=0.05)
    parser.add_argument("--max-alpha", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=20260720)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-records-per-split", type=int, default=0)
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument(
        "--patient-manifest",
        type=Path,
        default=ROOT
        / "results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv",
    )
    parser.add_argument(
        "--pca-root",
        type=Path,
        default=ROOT / "results/five_scale_pca_comparison_v1/workers",
    )
    parser.add_argument(
        "--readout-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/readouts",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/workers",
    )
    return parser.parse_args()


def readout_path(root: Path, model_safe: str, layer: int) -> Path:
    matches = sorted(root.glob(f"model_*_{model_safe}_layer{layer:02d}/readout.npz"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one readout for {model_safe} layer {layer}, found {matches}")
    return matches[0]


def load_pca_basis(path: Path, expected_mean: np.ndarray, expected_scale: np.ndarray):
    with np.load(path, allow_pickle=False) as payload:
        mean_error = float(np.max(np.abs(payload["activation_mean"] - expected_mean)))
        scale_error = float(np.max(np.abs(payload["activation_scale"] - expected_scale)))
        if max(mean_error, scale_error) > 1e-7:
            raise RuntimeError("PCA/checkpoint normalization mismatch")
        return (
            np.asarray(payload["pca_mean"], dtype=np.float32),
            np.asarray(payload["components"], dtype=np.float32),
        )


def patient_aggregate_with_reference(
    values: np.ndarray,
    patient_ids: np.ndarray,
    reference_patients: np.ndarray,
    reference_counts: np.ndarray,
) -> np.ndarray:
    patients, means, counts = aggregate_patient_means(values, patient_ids)
    if not (
        np.array_equal(patients, reference_patients)
        and np.allclose(counts, reference_counts, atol=0.0, rtol=0.0)
    ):
        raise RuntimeError("patient aggregation identity mismatch")
    return means


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


def main() -> None:
    args = parse_args()
    if tuple(args.ks) != KS:
        raise ValueError(f"locked protocol requires ks={KS}")
    if args.matched_budget < max(args.ks) or args.budget_replicates < 1:
        raise ValueError("invalid matched budget or replicate count")
    groups = final_groups(args.manifest, args.expansion)
    full_group_index, rows, seed_index, source_seed = source_identity(
        groups, "sae", args.source_index, 20
    )
    row = rows[0]
    source_row = rows[seed_index]
    output = args.output_root / (
        f"sae_{args.source_index:03d}_{row['model_safe']}_layer{int(row['layer']):02d}_seed{source_seed}"
    )
    table_path = output / "design_cells.csv"
    archive_path = output / "patient_metrics.npz"
    summary_path = output / "summary.json"
    if table_path.exists() and archive_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == PROTOCOL:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    splits = np.asarray([item["split"] for item in records])
    full_train_mask = splits == "train"
    train_idx = limited(np.flatnonzero(full_train_mask), args.max_records_per_split)
    validation_idx = limited(np.flatnonzero(splits == "val"), args.max_records_per_split)
    test_idx = limited(np.flatnonzero(splits == "test"), args.max_records_per_split)
    semantic_idx = train_idx
    if args.semantic_train_limit > 0 and len(train_idx) > args.semantic_train_limit:
        semantic_idx = np.sort(
            np.random.default_rng(20260714).choice(
                train_idx, args.semantic_train_limit, replace=False
            )
        )
    semantic_positions = np.searchsorted(train_idx, semantic_idx)
    if not np.array_equal(train_idx[semantic_positions], semantic_idx):
        raise RuntimeError("semantic rows are not a subset of training rows")

    concepts, concept_names, _, _ = standardized_concepts(
        [item["ecg_id"] for item in records], read_csv(args.concepts), full_train_mask
    )
    family_by_concept = {
        item["concept_id"]: item["family"]
        for item in read_csv(ROOT / "configs/concepts.csv")
        if item.get("main") == "yes"
    }
    concept_families = np.asarray([family_by_concept[name] for name in concept_names])
    y_train = np.asarray(concepts[train_idx], dtype=np.float32)
    mean, scale = checkpoint_normalization(resolved(row["checkpoint"]))
    x_train = normalize_rows(acts, train_idx, mean, scale)
    x_validation = normalize_rows(acts, validation_idx, mean, scale)
    x_test = normalize_rows(acts, test_idx, mean, scale)
    x_semantic = x_train[semantic_positions]

    readout_archive = readout_path(args.readout_root, row["model_safe"], int(row["layer"]))
    with np.load(readout_archive, allow_pickle=False) as payload:
        if not (
            np.array_equal(payload["concept_names"].astype(str), np.asarray(concept_names))
            and np.max(np.abs(payload["activation_mean"] - mean)) <= 1e-7
            and np.max(np.abs(payload["activation_scale"] - scale)) <= 1e-7
        ):
            raise RuntimeError("readout identity or normalization mismatch")
        readout_coefficients = np.asarray(payload["coefficients"], dtype=np.float32)

    pca_name = f"group_{full_group_index:03d}_{row['model_safe']}_layer{int(row['layer']):02d}"
    pca_model = args.pca_root / pca_name / "pca768_model.npz"
    pca_mean, pca_components = load_pca_basis(pca_model, mean, scale)
    pca_train = ((x_train - pca_mean) @ pca_components.T).astype(np.float32)
    pca_validation = ((x_validation - pca_mean) @ pca_components.T).astype(np.float32)
    pca_test = ((x_test - pca_mean) @ pca_components.T).astype(np.float32)
    pca_semantic = pca_train[semantic_positions]

    rng = np.random.default_rng(args.random_seed + full_group_index)
    random_q, _ = np.linalg.qr(rng.normal(size=(x_train.shape[1], x_train.shape[1])))
    random_q = random_q.astype(np.float32)
    random_train = (x_train @ random_q).astype(np.float32)
    random_validation = (x_validation @ random_q).astype(np.float32)
    random_test = (x_test @ random_q).astype(np.float32)
    random_semantic = random_train[semantic_positions]

    model, checkpoint_path = load_sae(source_row, args.device)
    sae_train = encode_sae(model, acts, train_idx, args.batch_size, args.device)
    sae_semantic = encode_sae(model, acts, semantic_idx, args.batch_size, args.device)
    sae_validation = encode_sae(model, acts, validation_idx, args.batch_size, args.device)
    sae_test = encode_sae(model, acts, test_idx, args.batch_size, args.device)
    sae_basis = model.W_dec.detach().cpu().numpy().T.astype(np.float32)
    if sae_basis.shape[0] != 6144 or args.matched_budget != 768:
        raise RuntimeError("locked protocol expects a 6144-wide SAE and 768 candidates")

    representations = {
        "dense": (x_train, x_semantic, x_validation, x_test, np.eye(x_train.shape[1], dtype=np.float32)),
        "pca": (pca_train, pca_semantic, pca_validation, pca_test, pca_components),
        "random_rotation": (
            random_train,
            random_semantic,
            random_validation,
            random_test,
            random_q.T,
        ),
    }
    baseline_units = {}
    for method, (train, semantic, validation, test, basis) in representations.items():
        ranking = candidate_ranking(feature_concept_correlations(semantic, concepts[semantic_idx]))
        for k in args.ks:
            for target_index in range(len(concept_names)):
                baseline_units[(method, int(k), target_index)] = centroid_unit_intervention(
                    train,
                    validation,
                    test,
                    y_train,
                    ranking,
                    basis,
                    readout_coefficients,
                    target_index=target_index,
                    k=int(k),
                    high_quantile=args.high_quantile,
                )

    patient_by_ecg = {
        str(item["ecg_id"]): str(item["patient_id"])
        for item in read_csv(args.patient_manifest)
    }
    test_ecg_ids = np.asarray([str(records[index]["ecg_id"]) for index in test_idx])
    test_patient_ids = np.asarray([patient_by_ecg[value] for value in test_ecg_ids])
    reference_patients, _, reference_counts = aggregate_patient_means(
        np.zeros((len(test_idx), 1), dtype=np.float32), test_patient_ids
    )
    shape = (
        len(ARMS),
        len(args.ks),
        len(METHODS),
        len(concept_names),
        len(reference_patients),
        len(METRICS),
    )
    metric_sums = np.zeros(shape, dtype=np.float64)
    eligible_counts = np.zeros(shape[:-2], dtype=np.int16)
    target_effect_sums = np.zeros(shape[:-2], dtype=np.float64)
    design_rows = []

    sae_correlations = feature_concept_correlations(sae_semantic, concepts[semantic_idx])
    arm_specs = [("full_6144", -1, np.arange(sae_train.shape[1], dtype=np.int64))]
    for replicate in range(args.budget_replicates):
        arm_specs.append(
            (
                "matched_768",
                replicate,
                deterministic_feature_subset(
                    sae_train.shape[1],
                    args.matched_budget,
                    args.budget_seed_base + replicate,
                ),
            )
        )

    for arm_name, budget_replicate, candidates in arm_specs:
        arm_index = ARMS.index(arm_name)
        sae_ranking = candidate_ranking(sae_correlations, candidates)
        for k_index, k in enumerate(args.ks):
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
                    k=int(k),
                    high_quantile=args.high_quantile,
                )
                units = {
                    "dense": baseline_units[("dense", int(k), target_index)],
                    "pca": baseline_units[("pca", int(k), target_index)],
                    "sae": sae_unit,
                    "random_rotation": baseline_units[("random_rotation", int(k), target_index)],
                }
                common, unit_effects, common_status = common_validation_effect(
                    {name: units[name] for name in ("dense", "pca", "sae")},
                    target_index,
                    cap=args.effect_cap,
                    floor=args.effect_floor,
                    max_alpha=args.max_alpha,
                )
                for method_index, method in enumerate(METHODS):
                    base = {
                        "protocol": PROTOCOL,
                        "model": row["model"],
                        "model_safe": row["model_safe"],
                        "layer": int(row["layer"]),
                        "source_seed": source_seed,
                        "candidate_arm": arm_name,
                        "budget_replicate": budget_replicate,
                        "candidate_count": len(candidates) if method == "sae" else 768,
                        "k": int(k),
                        "concept": concept,
                        "family": concept_families[target_index],
                        "method": method,
                        "selected_features": ";".join(
                            str(value) for value in units[method].selected_features
                        ),
                    }
                    unit_effect = units[method].validation_target_effect(target_index)
                    if common_status != "eligible":
                        design_rows.append(
                            point_row(base, common_status, unit_effect, common, np.nan, None)
                        )
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
                    design_rows.append(
                        point_row(base, status, unit_effect, common, alpha, metrics if len(metrics) else None)
                    )
                    if status != "eligible":
                        continue
                    patient_metrics = patient_aggregate_with_reference(
                        metrics, test_patient_ids, reference_patients, reference_counts
                    )
                    metric_sums[
                        arm_index, k_index, method_index, target_index
                    ] += patient_metrics
                    eligible_counts[arm_index, k_index, method_index, target_index] += 1
                    target_effect_sums[
                        arm_index, k_index, method_index, target_index
                    ] += common

    patient_metrics = np.full(shape, np.nan, dtype=np.float32)
    matched_effects = np.full(shape[:-2], np.nan, dtype=np.float32)
    for index in np.ndindex(eligible_counts.shape):
        count = int(eligible_counts[index])
        if count:
            patient_metrics[index] = (metric_sums[index] / count).astype(np.float32)
            matched_effects[index] = target_effect_sums[index] / count

    atomic_csv(table_path, design_rows)
    atomic_npz(
        archive_path,
        arm_names=np.asarray(ARMS),
        ks=np.asarray(args.ks, dtype=np.int32),
        method_names=np.asarray(METHODS),
        concept_names=np.asarray(concept_names),
        concept_families=concept_families,
        metric_names=np.asarray(METRICS),
        patient_ids=reference_patients,
        patient_counts=reference_counts,
        patient_metrics=patient_metrics,
        eligible_design_counts=eligible_counts,
        matched_validation_effects=matched_effects,
    )
    table_eligible = sum(row_value["status"] == "eligible" for row_value in design_rows)
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "source_index": args.source_index,
        "source_seed": source_seed,
        "model": row["model"],
        "model_safe": row["model_safe"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
        "checkpoint": str(checkpoint_path),
        "readout": str(readout_archive),
        "pca_model": str(pca_model),
        "n_train": len(train_idx),
        "n_semantic_train": len(semantic_idx),
        "n_validation": len(validation_idx),
        "n_test": len(test_idx),
        "n_patients": len(reference_patients),
        "concepts": len(concept_names),
        "ks": list(args.ks),
        "methods": list(METHODS),
        "budget_replicates": args.budget_replicates,
        "design_rows": len(design_rows),
        "eligible_design_rows": table_eligible,
        "effect_cap": args.effect_cap,
        "effect_floor": args.effect_floor,
        "max_alpha": args.max_alpha,
        "high_quantile": args.high_quantile,
        "semantic_encoding_stream": "independent",
        "common_effect_methods": ["dense", "pca", "sae"],
        "random_control_role": "sanity control; does not lower the shared target effect",
        "design_cells": str(table_path),
        "patient_metrics": str(archive_path),
        "smoke": args.max_records_per_split > 0,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
