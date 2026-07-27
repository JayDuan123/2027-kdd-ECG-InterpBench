#!/usr/bin/env python
"""Run dense-coordinate and replicated random controls for one E=8 model-depth."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.accessibility_calibration import (  # noqa: E402
    columnwise_pearson,
    feature_concept_correlations,
    ranked_feature_indices,
    selected_coordinate_predictions,
)
from benchmark_v1.multiscale_sae import read_csv, standardized_concepts  # noqa: E402
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
    calibration_rows,
    encode_random_dictionary,
    load_sae,
    normalized_dense,
    resolved,
)


PROTOCOL = "accessibility_calibration_e8_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--baseline-index", type=int, required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument("--random-seed-base", type=int, default=920000)
    parser.add_argument("--device", default="cuda")
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
        "--output-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v2/workers",
    )
    return parser.parse_args()


def baseline_groups(manifest: Path, expansion: int) -> list[list[dict[str, str]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in calibration_rows(manifest, expansion):
        grouped.setdefault((row["model_safe"], int(row["layer"])), []).append(row)
    groups = sorted(grouped.values(), key=lambda rows: min(int(r["task_index"]) for r in rows))
    for rows in groups:
        rows.sort(key=lambda row: int(row["seed"]))
        if len(rows) != 3 or len({int(row["seed"]) for row in rows}) != 3:
            raise RuntimeError(f"expected three SAE seeds for {rows[0]['model']} layer {rows[0]['layer']}")
        invariant = ("model", "model_safe", "layer", "relative_depth", "N", "k", "d_hidden")
        for key in invariant:
            if len({row[key] for row in rows}) != 1:
                raise RuntimeError(f"group invariant differs for {key}: {rows}")
    return groups


def checkpoint_normalization_audit(rows: list[dict[str, str]]) -> dict[str, float]:
    import torch

    means = []
    scales = []
    for row in rows:
        saved = torch.load(resolved(row["checkpoint"]), map_location="cpu", weights_only=False)
        state = saved["model"]
        means.append(np.asarray(state["mu"], dtype=np.float64))
        scales.append(np.asarray(state["sigma"], dtype=np.float64))
    mean_difference = max(float(np.max(np.abs(means[0] - value))) for value in means[1:])
    scale_difference = max(float(np.max(np.abs(scales[0] - value))) for value in scales[1:])
    if mean_difference > 1e-7 or scale_difference > 1e-7:
        raise RuntimeError(
            "SAE seed normalization mismatch: "
            f"mean={mean_difference}, scale={scale_difference}"
        )
    return {
        "max_abs_mean_difference": mean_difference,
        "max_abs_scale_difference": scale_difference,
    }


def dense_rows(
    row: dict[str, str],
    concept_names: list[str],
    family_by_concept: dict[str, str],
    selected: np.ndarray,
    validation_correlations: np.ndarray,
    test_correlations: np.ndarray,
    n_train: int,
    n_validation: int,
    n_test: int,
) -> list[dict[str, Any]]:
    return [
        {
            "model": row["model"],
            "model_safe": row["model_safe"],
            "layer": int(row["layer"]),
            "relative_depth": float(row["relative_depth"]),
            "concept": concept,
            "family": family_by_concept[concept],
            "method": "dense_single",
            "feature_count": 1,
            "selected_feature": int(selected[index]),
            "validation_r": float(validation_correlations[index]),
            "validation_abs_r": abs(float(validation_correlations[index])),
            "test_r": float(test_correlations[index]),
            "test_abs_r": abs(float(test_correlations[index])),
            "covered_020": int(abs(float(test_correlations[index])) >= 0.20),
            "n_train": n_train,
            "n_validation": n_validation,
            "n_test": n_test,
        }
        for index, concept in enumerate(concept_names)
    ]


def main() -> None:
    args = parse_args()
    if args.random_replicates < 1:
        raise ValueError("random-replicates must be positive")
    groups = baseline_groups(args.manifest, args.expansion)
    if not 0 <= args.baseline_index < len(groups):
        raise IndexError(f"baseline index outside 0..{len(groups) - 1}")
    rows = groups[args.baseline_index]
    row = rows[0]
    cell_name = (
        f"baseline_{args.baseline_index:03d}_{row['model_safe']}_"
        f"layer{int(row['layer']):02d}"
    )
    cell_root = args.output_root / cell_name
    dense_path = cell_root / "dense_single.csv"
    random_path = cell_root / "random_replicates.csv"
    predictions_path = cell_root / "test_predictions.npz"
    summary_path = cell_root / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if (
            existing.get("status") == "complete"
            and existing.get("protocol") == PROTOCOL
            and int(existing.get("random_replicates", -1)) == args.random_replicates
            and dense_path.exists()
            and random_path.exists()
            and predictions_path.exists()
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    normalization_audit = checkpoint_normalization_audit(rows)
    model, checkpoint_path = load_sae(row, args.device)
    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    if len(acts) != len(records):
        raise RuntimeError("activation and record counts differ")
    splits = np.asarray([record["split"] for record in records])
    train_idx = np.flatnonzero(splits == "train")
    validation_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")
    semantic_train_idx = train_idx
    if args.semantic_train_limit and len(train_idx) > args.semantic_train_limit:
        rng = np.random.default_rng(20260714)
        semantic_train_idx = np.sort(
            rng.choice(train_idx, size=args.semantic_train_limit, replace=False)
        )
    concepts, concept_names, _, _ = standardized_concepts(
        [record["ecg_id"] for record in records],
        read_csv(args.concepts),
        splits == "train",
    )
    family_by_concept = {
        registry_row["concept_id"]: registry_row["family"]
        for registry_row in read_csv(ROOT / "configs/concepts.csv")
        if registry_row.get("main") == "yes"
    }
    y_semantic = concepts[semantic_train_idx]
    y_validation = concepts[validation_idx]
    y_test = concepts[test_idx]

    dense_semantic = normalized_dense(model, acts, semantic_train_idx)
    dense_validation = normalized_dense(model, acts, validation_idx)
    dense_test = normalized_dense(model, acts, test_idx)
    dense_correlations = feature_concept_correlations(dense_semantic, y_semantic)
    dense_selected = ranked_feature_indices(dense_correlations)[0]
    dense_validation_prediction = selected_coordinate_predictions(
        dense_validation, dense_selected
    )
    dense_test_prediction = selected_coordinate_predictions(dense_test, dense_selected)
    dense_validation_correlations = columnwise_pearson(
        y_validation, dense_validation_prediction
    )
    dense_test_correlations = columnwise_pearson(y_test, dense_test_prediction)
    dense_output = dense_rows(
        row,
        concept_names,
        family_by_concept,
        dense_selected,
        dense_validation_correlations,
        dense_test_correlations,
        len(train_idx),
        len(validation_idx),
        len(test_idx),
    )

    random_output: list[dict[str, Any]] = []
    random_test_predictions = np.empty(
        (args.random_replicates, len(test_idx), len(concept_names)), dtype=np.float32
    )
    random_seeds = []
    for replicate in range(args.random_replicates):
        random_seed = args.random_seed_base + replicate
        random_seeds.append(random_seed)
        random_semantic = encode_random_dictionary(
            model,
            acts,
            semantic_train_idx,
            args.batch_size,
            args.device,
            random_seed,
        )
        random_correlations = feature_concept_correlations(random_semantic, y_semantic)
        random_selected = ranked_feature_indices(random_correlations)[0]
        random_validation = encode_random_dictionary(
            model,
            acts,
            validation_idx,
            args.batch_size,
            args.device,
            random_seed,
        )
        random_test = encode_random_dictionary(
            model, acts, test_idx, args.batch_size, args.device, random_seed
        )
        validation_prediction = selected_coordinate_predictions(
            random_validation, random_selected
        )
        test_prediction = selected_coordinate_predictions(random_test, random_selected)
        random_test_predictions[replicate] = test_prediction
        validation_correlations = columnwise_pearson(y_validation, validation_prediction)
        test_correlations = columnwise_pearson(y_test, test_prediction)
        for concept_index, concept in enumerate(concept_names):
            test_abs = abs(float(test_correlations[concept_index]))
            random_output.append(
                {
                    "model": row["model"],
                    "model_safe": row["model_safe"],
                    "layer": int(row["layer"]),
                    "relative_depth": float(row["relative_depth"]),
                    "random_replicate": replicate,
                    "random_seed": random_seed,
                    "concept": concept,
                    "family": family_by_concept[concept],
                    "method": "random_single",
                    "feature_count": 1,
                    "selected_feature": int(random_selected[concept_index]),
                    "validation_r": float(validation_correlations[concept_index]),
                    "validation_abs_r": abs(
                        float(validation_correlations[concept_index])
                    ),
                    "test_r": float(test_correlations[concept_index]),
                    "test_abs_r": test_abs,
                    "covered_020": int(test_abs >= 0.20),
                    "n_train": len(train_idx),
                    "n_validation": len(validation_idx),
                    "n_test": len(test_idx),
                }
            )

    patient_by_ecg = {
        str(patient_row["ecg_id"]): str(patient_row["patient_id"])
        for patient_row in read_csv(args.patient_manifest)
    }
    test_ecg_ids = np.asarray([records[index]["ecg_id"] for index in test_idx])
    test_patient_ids = np.asarray(
        [patient_by_ecg[str(ecg_id)] for ecg_id in test_ecg_ids]
    )
    atomic_csv(dense_path, dense_output)
    atomic_csv(random_path, random_output)
    atomic_npz(
        predictions_path,
        concept_names=np.asarray(concept_names),
        test_ecg_ids=test_ecg_ids,
        test_patient_ids=test_patient_ids,
        y_test=np.asarray(y_test, dtype=np.float32),
        prediction_dense_single=dense_test_prediction,
        prediction_random_single=random_test_predictions,
        random_seeds=np.asarray(random_seeds, dtype=np.int64),
    )
    payload = {
        "status": "complete",
        "protocol": PROTOCOL,
        "baseline_index": args.baseline_index,
        "model": row["model"],
        "model_safe": row["model_safe"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
        "expansion_E": int(row["expansion_E"]),
        "N": int(row["N"]),
        "k": int(row["k"]),
        "sae_seeds": [int(value["seed"]) for value in rows],
        "normalization_checkpoint": str(checkpoint_path),
        "normalization_audit": normalization_audit,
        "random_replicates": args.random_replicates,
        "random_seeds": random_seeds,
        "n_concepts": len(concept_names),
        "n_train": len(train_idx),
        "n_semantic_train": len(semantic_train_idx),
        "n_validation": len(validation_idx),
        "n_test": len(test_idx),
        "dense_single_table": str(dense_path),
        "random_replicates_table": str(random_path),
        "test_predictions": str(predictions_path),
        "claim_boundary": (
            "single coordinates are selected on train only; dense tests native-axis "
            "localization and random controls width/sparsity/search multiplicity"
        ),
    }
    atomic_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
