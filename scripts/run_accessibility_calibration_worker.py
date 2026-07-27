#!/usr/bin/env python
"""Run one E=8 cell of the clinical-accessibility calibration ladder."""

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
    canonical_single_atom_features,
    columnwise_pearson,
    feature_concept_correlations,
    fit_ridge_predictions,
    ranked_feature_indices,
    safe_ratio,
)
from benchmark_v1.multiscale_sae import read_csv, standardized_concepts  # noqa: E402


METHODS = (
    "dense_fm",
    "full_sae",
    "sae_top16",
    "sae_top4",
    "sae_single",
    "random_single",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--calibration-index", type=int, required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument(
        "--concept-registry", type=Path, default=ROOT / "configs/concepts.csv"
    )
    parser.add_argument("--complete-case-concepts", action="store_true")
    parser.add_argument(
        "--patient-manifest",
        type=Path,
        default=ROOT
        / "results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v1/workers",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty calibration table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def calibration_rows(manifest: Path, expansion: int) -> list[dict[str, str]]:
    rows = [row for row in read_csv(manifest) if int(row["expansion_E"]) == expansion]
    rows.sort(key=lambda row: int(row["task_index"]))
    return rows


def resolved(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def load_sae(row: dict[str, str], device: str):
    import torch

    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    checkpoint = resolved(row["checkpoint"])
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config = saved.get("config", {})
    if not saved.get("final", False):
        raise RuntimeError(f"checkpoint is not final: {checkpoint}")
    if config.get("config_hash") != row["config_hash"]:
        raise RuntimeError(f"checkpoint/manifest hash mismatch: {checkpoint}")
    model = BatchTopKSAE(
        int(config["d_hidden"]), int(config["N"]), int(config["k"])
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, checkpoint


def encode_sae(model, acts: np.ndarray, indices: np.ndarray, batch_size: int, device: str):
    import torch
    from scipy import sparse

    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            raw = torch.as_tensor(
                np.asarray(acts[batch_indices]), dtype=torch.float32, device=device
            )
            z = model.encode(raw)
            nonzero = torch.nonzero(z, as_tuple=False)
            if len(nonzero):
                coordinates = nonzero.cpu().numpy()
                values = z[nonzero[:, 0], nonzero[:, 1]].cpu().numpy()
                chunk = sparse.csr_matrix(
                    (values, (coordinates[:, 0], coordinates[:, 1])),
                    shape=(len(batch_indices), model.N),
                    dtype=np.float32,
                )
            else:
                chunk = sparse.csr_matrix((len(batch_indices), model.N), dtype=np.float32)
            chunks.append(chunk)
    return sparse.vstack(chunks, format="csr")


def encode_random_dictionary(
    model,
    acts: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: str,
    random_seed: int,
):
    import torch
    from scipy import sparse

    rng = np.random.default_rng(random_seed)
    directions = rng.normal(size=(model.N, model.d)).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True).clip(min=1e-8)
    directions_tensor = torch.from_numpy(directions).to(device)
    chunks = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            raw = torch.as_tensor(
                np.asarray(acts[batch_indices]), dtype=torch.float32, device=device
            )
            normalized = model.normalise(raw)
            pre = torch.relu(normalized @ directions_tensor.t())
            z = model.encode_pre_activations(pre)
            nonzero = torch.nonzero(z, as_tuple=False)
            if len(nonzero):
                coordinates = nonzero.cpu().numpy()
                values = z[nonzero[:, 0], nonzero[:, 1]].cpu().numpy()
                chunk = sparse.csr_matrix(
                    (values, (coordinates[:, 0], coordinates[:, 1])),
                    shape=(len(batch_indices), model.N),
                    dtype=np.float32,
                )
            else:
                chunk = sparse.csr_matrix((len(batch_indices), model.N), dtype=np.float32)
            chunks.append(chunk)
    return sparse.vstack(chunks, format="csr")


def normalized_dense(model, acts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    mean = model.mu.detach().cpu().numpy()
    scale = model.sigma.detach().cpu().numpy()
    return ((np.asarray(acts[indices], dtype=np.float32) - mean) / scale).astype(
        np.float32, copy=False
    )


def topk_ridge_predictions(
    z_train,
    y_train: np.ndarray,
    z_validation,
    z_test,
    ranking: np.ndarray,
    topk: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_concepts = y_train.shape[1]
    validation = np.empty((z_validation.shape[0], n_concepts), dtype=np.float32)
    test = np.empty((z_test.shape[0], n_concepts), dtype=np.float32)
    for concept_index in range(n_concepts):
        selected = ranking[:topk, concept_index]
        validation_prediction, test_prediction = fit_ridge_predictions(
            z_train[:, selected],
            y_train[:, concept_index],
            z_validation[:, selected],
            z_test[:, selected],
            alpha,
        )
        validation[:, concept_index] = np.asarray(validation_prediction).ravel()
        test[:, concept_index] = np.asarray(test_prediction).ravel()
    return validation, test


def selected_predictions(codes, selected: np.ndarray) -> np.ndarray:
    return np.asarray(codes[:, selected].toarray(), dtype=np.float32)


def existing_single_atom_reference(
    concept_metrics_path: Path,
    concept_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(concept_metrics_path) if row["split"] == "test"]
    by_concept = {row["concept"]: row for row in rows}
    selected = np.asarray(
        [int(float(by_concept[name]["selected_feature"])) for name in concept_names]
    )
    correlations = np.asarray(
        [float(by_concept[name]["eval_correlation"]) for name in concept_names]
    )
    return selected, correlations


def existing_single_atom_audit(
    expected_correlations: np.ndarray,
    test_correlations: np.ndarray,
    ranking_mismatch: int,
) -> dict[str, float]:
    correlation_error = float(np.max(np.abs(expected_correlations - test_correlations)))
    if correlation_error > 2e-4:
        raise RuntimeError(
            "single-atom reproduction failed: "
            f"max_abs_error={correlation_error}"
        )
    return {
        "selected_feature_mismatch_count": 0,
        "recomputed_ranking_mismatch_count": ranking_mismatch,
        "max_abs_test_correlation_error": correlation_error,
    }


def main() -> None:
    args = parse_args()
    rows = calibration_rows(args.manifest, args.expansion)
    if not 0 <= args.calibration_index < len(rows):
        raise IndexError(
            f"calibration index {args.calibration_index} outside 0..{len(rows) - 1}"
        )
    row = rows[args.calibration_index]
    cell_name = (
        f"task_{args.calibration_index:03d}_{row['model_safe']}_layer{int(row['layer']):02d}_"
        f"seed{int(row['seed'])}"
    )
    cell_root = args.output_root / cell_name
    summary_path = cell_root / "summary.json"
    table_path = cell_root / "calibration.csv"
    predictions_path = cell_root / "test_predictions.npz"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if (
            existing.get("status") == "complete"
            and int(existing.get("source_task_index", -1)) == int(row["task_index"])
            and table_path.exists()
            and predictions_path.exists()
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    if len(acts) != len(records):
        raise RuntimeError("activation and record counts differ")
    splits = np.asarray([record["split"] for record in records])
    train_idx = np.flatnonzero(splits == "train")
    validation_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")
    concepts, concept_names, _, _ = standardized_concepts(
        [record["ecg_id"] for record in records],
        read_csv(args.concepts),
        splits == "train",
        preserve_missing=args.complete_case_concepts,
    )
    if args.complete_case_concepts:
        complete = np.all(np.isfinite(concepts), axis=1)
        train_idx = train_idx[complete[train_idx]]
        validation_idx = validation_idx[complete[validation_idx]]
        test_idx = test_idx[complete[test_idx]]
    semantic_train_idx = train_idx
    if args.semantic_train_limit and len(train_idx) > args.semantic_train_limit:
        rng = np.random.default_rng(20260714)
        semantic_train_idx = np.sort(
            rng.choice(train_idx, size=args.semantic_train_limit, replace=False)
        )
    family_by_concept = {
        registry_row["concept_id"]: registry_row["family"]
        for registry_row in read_csv(args.concept_registry)
        if registry_row.get("main") == "yes"
    }
    model, checkpoint_path = load_sae(row, args.device)

    z_semantic = encode_sae(
        model, acts, semantic_train_idx, args.batch_size, args.device
    )
    train_correlations = feature_concept_correlations(
        z_semantic, concepts[semantic_train_idx]
    )
    ranking = ranked_feature_indices(train_correlations)
    canonical_selected, expected_single_correlations = existing_single_atom_reference(
        resolved(row["concept_metrics"]), concept_names
    )
    selected, ranking_mismatch = canonical_single_atom_features(
        ranking, canonical_selected
    )

    z_train = encode_sae(model, acts, train_idx, args.batch_size, args.device)
    z_validation = encode_sae(
        model, acts, validation_idx, args.batch_size, args.device
    )
    z_test = encode_sae(model, acts, test_idx, args.batch_size, args.device)
    y_train = concepts[train_idx]
    y_validation = concepts[validation_idx]
    y_test = concepts[test_idx]

    dense_train = normalized_dense(model, acts, train_idx)
    dense_validation = normalized_dense(model, acts, validation_idx)
    dense_test = normalized_dense(model, acts, test_idx)
    dense_validation_prediction, dense_test_prediction = fit_ridge_predictions(
        dense_train,
        y_train,
        dense_validation,
        dense_test,
        args.ridge_alpha,
    )
    full_validation_prediction, full_test_prediction = fit_ridge_predictions(
        z_train,
        y_train,
        z_validation,
        z_test,
        args.ridge_alpha,
    )
    top16_validation_prediction, top16_test_prediction = topk_ridge_predictions(
        z_train,
        y_train,
        z_validation,
        z_test,
        ranking,
        16,
        args.ridge_alpha,
    )
    top4_validation_prediction, top4_test_prediction = topk_ridge_predictions(
        z_train,
        y_train,
        z_validation,
        z_test,
        ranking,
        4,
        args.ridge_alpha,
    )
    single_validation_prediction = selected_predictions(z_validation, selected)
    single_test_prediction = selected_predictions(z_test, selected)

    random_seed = 910000 + int(row["seed"])
    random_semantic = encode_random_dictionary(
        model,
        acts,
        semantic_train_idx,
        args.batch_size,
        args.device,
        random_seed,
    )
    random_correlations = feature_concept_correlations(
        random_semantic, concepts[semantic_train_idx]
    )
    random_ranking = ranked_feature_indices(random_correlations)
    random_selected = random_ranking[0]
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
    random_validation_prediction = selected_predictions(random_validation, random_selected)
    random_test_prediction = selected_predictions(random_test, random_selected)

    predictions = {
        "dense_fm": (dense_validation_prediction, dense_test_prediction),
        "full_sae": (full_validation_prediction, full_test_prediction),
        "sae_top16": (top16_validation_prediction, top16_test_prediction),
        "sae_top4": (top4_validation_prediction, top4_test_prediction),
        "sae_single": (single_validation_prediction, single_test_prediction),
        "random_single": (random_validation_prediction, random_test_prediction),
    }
    validation_correlations = {
        method: columnwise_pearson(y_validation, values[0])
        for method, values in predictions.items()
    }
    test_correlations = {
        method: columnwise_pearson(y_test, values[1])
        for method, values in predictions.items()
    }
    reproduction = existing_single_atom_audit(
        expected_single_correlations,
        test_correlations["sae_single"],
        ranking_mismatch,
    )
    dense_abs = np.abs(test_correlations["dense_fm"])
    full_abs = np.abs(test_correlations["full_sae"])
    method_feature_count = {
        "dense_fm": int(row["d_hidden"]),
        "full_sae": int(row["N"]),
        "sae_top16": 16,
        "sae_top4": 4,
        "sae_single": 1,
        "random_single": 1,
    }
    output_rows: list[dict[str, Any]] = []
    for method in METHODS:
        ratio_dense = safe_ratio(np.abs(test_correlations[method]), dense_abs)
        ratio_full = safe_ratio(np.abs(test_correlations[method]), full_abs)
        for concept_index, concept in enumerate(concept_names):
            if method == "sae_top16":
                feature_ids = ranking[:16, concept_index]
            elif method == "sae_top4":
                feature_ids = ranking[:4, concept_index]
            elif method == "sae_single":
                feature_ids = selected[concept_index : concept_index + 1]
            elif method == "random_single":
                feature_ids = random_selected[concept_index : concept_index + 1]
            else:
                feature_ids = np.asarray([], dtype=int)
            output_rows.append(
                {
                    "calibration_index": args.calibration_index,
                    "source_task_index": int(row["task_index"]),
                    "model": row["model"],
                    "layer": int(row["layer"]),
                    "relative_depth": float(row["relative_depth"]),
                    "seed": int(row["seed"]),
                    "expansion_E": int(row["expansion_E"]),
                    "N": int(row["N"]),
                    "k": int(row["k"]),
                    "concept": concept,
                    "family": family_by_concept[concept],
                    "method": method,
                    "feature_count": method_feature_count[method],
                    "selected_features": ";".join(map(str, feature_ids.tolist())),
                    "ridge_alpha": args.ridge_alpha if method in METHODS[:4] else "",
                    "validation_r": float(validation_correlations[method][concept_index]),
                    "validation_abs_r": abs(
                        float(validation_correlations[method][concept_index])
                    ),
                    "test_r": float(test_correlations[method][concept_index]),
                    "test_abs_r": abs(float(test_correlations[method][concept_index])),
                    "ratio_to_dense_fm": float(ratio_dense[concept_index]),
                    "ratio_to_full_sae": float(ratio_full[concept_index]),
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
    test_patient_ids = np.asarray([patient_by_ecg[str(ecg_id)] for ecg_id in test_ecg_ids])
    atomic_csv(table_path, output_rows)
    atomic_npz(
        predictions_path,
        concept_names=np.asarray(concept_names),
        test_ecg_ids=test_ecg_ids,
        test_patient_ids=test_patient_ids,
        y_test=np.asarray(y_test, dtype=np.float32),
        **{
            f"prediction_{method}": np.asarray(predictions[method][1], dtype=np.float32)
            for method in METHODS
        },
    )
    payload = {
        "status": "complete",
        "protocol": "accessibility_calibration_e8_v1",
        "calibration_index": args.calibration_index,
        "source_task_index": int(row["task_index"]),
        "model": row["model"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
        "seed": int(row["seed"]),
        "expansion_E": int(row["expansion_E"]),
        "N": int(row["N"]),
        "k": int(row["k"]),
        "checkpoint": str(checkpoint_path),
        "ridge_alpha": args.ridge_alpha,
        "random_dictionary_seed": random_seed,
        "random_baseline": "Gaussian unit directions with matched N, BatchTopK k, batching, and train-only coordinate selection",
        "n_concepts": len(concept_names),
        "n_train": len(train_idx),
        "n_semantic_train": len(semantic_train_idx),
        "n_validation": len(validation_idx),
        "n_test": len(test_idx),
        "methods": list(METHODS),
        "single_atom_reproduction": reproduction,
        "calibration_table": str(table_path),
        "test_predictions": str(predictions_path),
        "claim_boundary": "dense/full/top-k are fixed-alpha linear readouts; single-coordinate methods are fit-free held-out correlations",
    }
    atomic_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
