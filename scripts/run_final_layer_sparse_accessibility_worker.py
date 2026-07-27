#!/usr/bin/env python
"""Run one final-layer dense/PCA/SAE/random sparse-accessibility worker."""

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
from benchmark_v1.multiscale_sae import read_csv, standardized_concepts  # noqa: E402
from benchmark_v1.sparse_accessibility import (  # noqa: E402
    candidate_ranking,
    deterministic_feature_subset,
    fit_sparse_ridge_curve,
)
from scripts.run_accessibility_baselines_v2_worker import baseline_groups  # noqa: E402
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
    encode_random_dictionary,
    encode_sae,
    load_sae,
    resolved,
)


PROTOCOL = "final_layer_sparse_accessibility_e8_v2"
SOURCE_KINDS = ("dense", "pca", "sae", "random")
KS = (1, 2, 4, 8, 16, 32)
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def parse_csv_numbers(value: str, cast) -> tuple:
    result = tuple(cast(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--source-kind", choices=SOURCE_KINDS, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument("--matched-budget", type=int, default=768)
    parser.add_argument("--budget-replicates", type=int, default=20)
    parser.add_argument("--budget-seed-base", type=int, default=960000)
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument("--random-seed-base", type=int, default=950000)
    parser.add_argument("--ks", type=lambda x: parse_csv_numbers(x, int), default=KS)
    parser.add_argument(
        "--alphas", type=lambda x: parse_csv_numbers(x, float), default=ALPHAS
    )
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
        "--concept-registry", type=Path, default=ROOT / "configs/concepts.csv"
    )
    parser.add_argument("--expected-concepts", type=int, default=49)
    parser.add_argument("--complete-case-concepts", action="store_true")
    parser.add_argument(
        "--pca-root",
        type=Path,
        default=ROOT / "results/five_scale_pca_comparison_v1/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/workers",
    )
    return parser.parse_args()


def final_groups(manifest: Path, expansion: int) -> list[tuple[int, list[dict[str, str]]]]:
    groups = baseline_groups(manifest, expansion)
    result = [
        (index, rows)
        for index, rows in enumerate(groups)
        if np.isclose(float(rows[0]["relative_depth"]), 1.0)
    ]
    if len(result) != 6:
        raise RuntimeError(f"expected six final-layer groups, found {len(result)}")
    return result


def source_identity(
    groups: list[tuple[int, list[dict[str, str]]]],
    kind: str,
    index: int,
    random_replicates: int,
) -> tuple[int, list[dict[str, str]], int, int]:
    """Return full group index, rows, source replicate, and source seed."""
    if kind in {"dense", "pca"}:
        if not 0 <= index < len(groups):
            raise IndexError(f"{kind} source index outside 0..{len(groups) - 1}")
        full_index, rows = groups[index]
        return full_index, rows, 0, 0
    if kind == "sae":
        if not 0 <= index < len(groups) * 3:
            raise IndexError(f"sae source index outside 0..{len(groups) * 3 - 1}")
        group_index, seed_index = divmod(index, 3)
        full_index, rows = groups[group_index]
        return full_index, rows, seed_index, int(rows[seed_index]["seed"])
    if not 0 <= index < len(groups) * random_replicates:
        raise IndexError(
            f"random source index outside 0..{len(groups) * random_replicates - 1}"
        )
    group_index, replicate = divmod(index, random_replicates)
    full_index, rows = groups[group_index]
    return full_index, rows, replicate, replicate


def checkpoint_normalization(checkpoint: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not payload.get("final", False):
        raise RuntimeError(f"checkpoint is not final: {checkpoint}")
    state = payload["model"]
    return (
        np.asarray(state["mu"], dtype=np.float32),
        np.asarray(state["sigma"], dtype=np.float32),
    )


def normalize_rows(
    acts: np.ndarray, indices: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (
        (np.asarray(acts[indices], dtype=np.float32) - mean) / scale
    ).astype(np.float32, copy=False)


def pca_rows(
    normalized: np.ndarray, model_path: Path, expected_mean: np.ndarray, expected_scale: np.ndarray
) -> np.ndarray:
    with np.load(model_path, allow_pickle=False) as payload:
        mean_error = float(
            np.max(np.abs(np.asarray(payload["activation_mean"]) - expected_mean))
        )
        scale_error = float(
            np.max(np.abs(np.asarray(payload["activation_scale"]) - expected_scale))
        )
        if max(mean_error, scale_error) > 1e-7:
            raise RuntimeError(
                f"PCA/checkpoint normalization mismatch: mean={mean_error}, scale={scale_error}"
            )
        pca_mean = np.asarray(payload["pca_mean"], dtype=np.float32)
        components = np.asarray(payload["components"], dtype=np.float32)
    return ((normalized - pca_mean) @ components.T).astype(np.float32)


def limited(indices: np.ndarray, limit: int) -> np.ndarray:
    return indices if limit <= 0 else indices[: min(limit, len(indices))]


def main() -> None:
    args = parse_args()
    if args.matched_budget < max(args.ks):
        raise ValueError("matched budget must be at least the largest k")
    if args.budget_replicates < 1 or args.random_replicates < 1:
        raise ValueError("replicate counts must be positive")
    groups = final_groups(args.manifest, args.expansion)
    full_group_index, rows, source_replicate, source_seed = source_identity(
        groups, args.source_kind, args.source_index, args.random_replicates
    )
    row = rows[0]
    name = (
        f"{args.source_kind}_{args.source_index:03d}_{row['model_safe']}_"
        f"layer{int(row['layer']):02d}"
    )
    output = args.output_root / name
    metrics_path = output / "curve_metrics.csv"
    predictions_path = output / "test_predictions.npz"
    summary_path = output / "summary.json"
    if summary_path.exists() and metrics_path.exists() and predictions_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == PROTOCOL:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    if len(acts) != len(records) or acts.shape[1] != int(row["d_hidden"]):
        raise RuntimeError(f"unexpected activation/record shape: {acts.shape}, {len(records)}")
    splits = np.asarray([record["split"] for record in records])
    train_idx = limited(np.flatnonzero(splits == "train"), args.max_records_per_split)
    validation_idx = limited(
        np.flatnonzero(splits == "val"), args.max_records_per_split
    )
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
        raise RuntimeError("semantic training rows are not a subset of training rows")

    concepts, concept_names, _, _ = standardized_concepts(
        [record["ecg_id"] for record in records],
        read_csv(args.concepts),
        splits == "train",
        preserve_missing=args.complete_case_concepts,
    )
    if len(concept_names) != args.expected_concepts:
        raise RuntimeError(
            f"expected {args.expected_concepts} waveform concepts, found {len(concept_names)}"
        )
    if args.complete_case_concepts:
        complete = np.all(np.isfinite(concepts), axis=1)
        train_idx = train_idx[complete[train_idx]]
        validation_idx = validation_idx[complete[validation_idx]]
        test_idx = test_idx[complete[test_idx]]
        semantic_idx = train_idx
        if args.semantic_train_limit > 0 and len(train_idx) > args.semantic_train_limit:
            semantic_idx = np.sort(
                np.random.default_rng(20260714).choice(
                    train_idx, args.semantic_train_limit, replace=False
                )
            )
        semantic_positions = np.searchsorted(train_idx, semantic_idx)
    family_by_concept = {
        item["concept_id"]: item["family"]
        for item in read_csv(args.concept_registry)
        if item.get("main") == "yes"
    }
    patient_by_ecg = {
        str(item["ecg_id"]): str(item["patient_id"])
        for item in read_csv(args.patient_manifest)
    }
    test_ecg_ids = np.asarray([str(records[index]["ecg_id"]) for index in test_idx])
    test_patient_ids = np.asarray([patient_by_ecg[value] for value in test_ecg_ids])
    y_train = concepts[train_idx]
    y_validation = concepts[validation_idx]
    y_test = concepts[test_idx]

    reference_row = rows[0]
    mean, scale = checkpoint_normalization(resolved(reference_row["checkpoint"]))
    checkpoint_path = resolved(reference_row["checkpoint"])
    if args.source_kind in {"dense", "pca"}:
        x_train = normalize_rows(acts, train_idx, mean, scale)
        x_validation = normalize_rows(acts, validation_idx, mean, scale)
        x_test = normalize_rows(acts, test_idx, mean, scale)
        if args.source_kind == "pca":
            pca_name = (
                f"group_{full_group_index:03d}_{row['model_safe']}_"
                f"layer{int(row['layer']):02d}"
            )
            pca_model = args.pca_root / pca_name / "pca768_model.npz"
            if not pca_model.exists():
                raise FileNotFoundError(f"missing train-fitted PCA model: {pca_model}")
            x_train = pca_rows(x_train, pca_model, mean, scale)
            x_validation = pca_rows(x_validation, pca_model, mean, scale)
            x_test = pca_rows(x_test, pca_model, mean, scale)
            checkpoint_path = pca_model
        x_semantic = x_train[semantic_positions]
    else:
        source_row = rows[source_replicate] if args.source_kind == "sae" else rows[0]
        model, checkpoint_path = load_sae(source_row, args.device)
        if args.source_kind == "sae":
            encoder = lambda indices: encode_sae(  # noqa: E731
                model, acts, indices, args.batch_size, args.device
            )
        else:
            random_seed = args.random_seed_base + source_replicate
            source_seed = random_seed
            encoder = lambda indices: encode_random_dictionary(  # noqa: E731
                model,
                acts,
                indices,
                args.batch_size,
                args.device,
                random_seed,
            )
        x_train = encoder(train_idx)
        # BatchTopK codes depend on batch composition. Match the locked benchmark
        # protocol by encoding the semantic-selection subset as its own stream.
        x_semantic = encoder(semantic_idx)
        x_validation = encoder(validation_idx)
        x_test = encoder(test_idx)

    if not (x_train.shape[1] == x_validation.shape[1] == x_test.shape[1]):
        raise RuntimeError("feature widths differ across splits")
    width = int(x_train.shape[1])
    if args.matched_budget > width:
        raise RuntimeError(f"matched budget {args.matched_budget} exceeds width {width}")
    train_correlations = feature_concept_correlations(
        x_semantic, concepts[semantic_idx]
    )

    arms: list[tuple[str, int, np.ndarray]] = []
    if width == args.matched_budget:
        arms.append(("matched_768", 0, np.arange(width, dtype=np.int64)))
    else:
        arms.append(("full_6144", -1, np.arange(width, dtype=np.int64)))
        if args.source_kind == "sae":
            budget_replicates = range(args.budget_replicates)
        else:
            budget_replicates = (source_replicate % args.budget_replicates,)
        for budget_replicate in budget_replicates:
            subset = deterministic_feature_subset(
                width,
                args.matched_budget,
                args.budget_seed_base + int(budget_replicate),
            )
            arms.append((f"matched_768_r{int(budget_replicate):02d}", int(budget_replicate), subset))

    output_rows: list[dict[str, Any]] = []
    prediction_blocks = []
    arm_names = []
    arm_budgets = []
    for arm_name, budget_replicate, candidates in arms:
        ranking = candidate_ranking(train_correlations, candidates)
        curve = fit_sparse_ridge_curve(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            ranking,
            ks=args.ks,
            alphas=args.alphas,
        )
        prediction_blocks.append(curve.test_predictions)
        arm_names.append(arm_name)
        arm_budgets.append(budget_replicate)
        for k_index, k in enumerate(curve.ks):
            for concept_index, concept in enumerate(concept_names):
                output_rows.append(
                    {
                        "protocol": PROTOCOL,
                        "model": row["model"],
                        "model_safe": row["model_safe"],
                        "layer": int(row["layer"]),
                        "relative_depth": float(row["relative_depth"]),
                        "source_kind": args.source_kind,
                        "source_index": args.source_index,
                        "source_replicate": source_replicate,
                        "source_seed": source_seed,
                        "candidate_arm": "full_6144" if budget_replicate < 0 else "matched_768",
                        "budget_replicate": budget_replicate,
                        "candidate_count": len(candidates),
                        "k": int(k),
                        "concept": concept,
                        "family": family_by_concept[concept],
                        "selected_features": ";".join(
                            str(value)
                            for value in curve.selected_features[k_index, concept_index]
                        ),
                        "selected_alpha": float(
                            curve.selected_alphas[k_index, concept_index]
                        ),
                        "validation_r": float(
                            curve.validation_correlations[k_index, concept_index]
                        ),
                        "validation_abs_r": abs(
                            float(curve.validation_correlations[k_index, concept_index])
                        ),
                        "test_r": float(curve.test_correlations[k_index, concept_index]),
                        "test_abs_r": abs(
                            float(curve.test_correlations[k_index, concept_index])
                        ),
                        "covered_020": int(
                            abs(float(curve.test_correlations[k_index, concept_index]))
                            >= 0.20
                        ),
                        "n_train": len(train_idx),
                        "n_validation": len(validation_idx),
                        "n_test": len(test_idx),
                    }
                )

    atomic_csv(metrics_path, output_rows)
    atomic_npz(
        predictions_path,
        ks=np.asarray(args.ks, dtype=np.int32),
        concept_names=np.asarray(concept_names),
        arm_names=np.asarray(arm_names),
        arm_budget_replicates=np.asarray(arm_budgets, dtype=np.int32),
        test_ecg_ids=test_ecg_ids,
        test_patient_ids=test_patient_ids,
        y_test=np.asarray(y_test, dtype=np.float32),
        predictions=np.asarray(prediction_blocks, dtype=np.float32),
    )
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "source_kind": args.source_kind,
        "source_index": args.source_index,
        "source_replicate": source_replicate,
        "source_seed": source_seed,
        "model": row["model"],
        "model_safe": row["model_safe"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
        "expansion_E": args.expansion,
        "dictionary_width": width,
        "matched_budget": args.matched_budget,
        "budget_replicates": len([value for value in arm_budgets if value >= 0]),
        "ks": list(args.ks),
        "alphas": list(args.alphas),
        "n_train": len(train_idx),
        "n_semantic_train": len(semantic_idx),
        "n_validation": len(validation_idx),
        "n_test": len(test_idx),
        "n_concepts": len(concept_names),
        "semantic_encoding_stream": (
            "independent" if args.source_kind in {"sae", "random"} else "dense_subset"
        ),
        "checkpoint_or_pca": str(checkpoint_path),
        "metrics": str(metrics_path),
        "test_predictions": str(predictions_path),
        "smoke": args.max_records_per_split > 0,
        "selection": "top-k features ranked on fixed semantic train; ridge fit on train; alpha selected on validation",
        "evaluation": "patient-disjoint test with no reselection",
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
