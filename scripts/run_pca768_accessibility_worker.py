#!/usr/bin/env python3
"""Fit one train-only PCA-768 baseline and evaluate frozen concept coordinates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.accessibility_calibration import feature_concept_correlations  # noqa: E402
from benchmark_v1.multiscale_sae import (  # noqa: E402
    read_csv,
    selected_concept_metrics,
    standardized_concepts,
)
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
    resolved,
)

PROTOCOL = "pca768_accessibility_v1"
SEMANTIC_SEED = 20_260_714


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/five_scale_pca_comparison_v1/workers",
    )
    parser.add_argument(
        "--concept-registry", type=Path, default=ROOT / "configs/concepts.csv"
    )
    parser.add_argument("--expected-concepts", type=int, default=49)
    parser.add_argument("--complete-case-concepts", action="store_true")
    return parser.parse_args()


def pca_groups(manifest: Path) -> list[list[dict[str, str]]]:
    rows = [row for row in read_csv(manifest) if int(row["expansion_E"]) == 1]
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["model_safe"], int(row["layer"])), []).append(row)
    groups = sorted(grouped.values(), key=lambda values: min(int(row["task_index"]) for row in values))
    if len(groups) != 30:
        raise RuntimeError(f"expected 30 model-depth groups, got {len(groups)}")
    for group in groups:
        group.sort(key=lambda row: int(row["seed"]))
        if [int(row["seed"]) for row in group] != [4311, 4312, 4313]:
            raise RuntimeError("each PCA group must map to three canonical SAE seeds")
        for key in (
            "model", "model_safe", "layer", "relative_depth", "d_hidden",
            "activation_path", "records_path",
        ):
            if len({row[key] for row in group}) != 1:
                raise RuntimeError(f"PCA group invariant differs for {key}")
    return groups


def normalization_from_train(
    acts: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(acts[train_idx], dtype=np.float32)
    mean = np.mean(train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    normalized = ((train - mean) / scale).astype(np.float32)
    return normalized, mean, scale


def checkpoint_normalization_error(
    checkpoint: Path, mean: np.ndarray, scale: np.ndarray
) -> dict[str, float]:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not payload.get("final", False):
        raise RuntimeError(f"normalization reference is not a final checkpoint: {checkpoint}")
    state = payload["model"]
    saved_mean = np.asarray(state["mu"], dtype=np.float32)
    saved_scale = np.asarray(state["sigma"], dtype=np.float32)
    return {
        "max_abs_mean_error": float(np.max(np.abs(saved_mean - mean))),
        "max_abs_scale_error": float(np.max(np.abs(saved_scale - scale))),
    }


def normalized_rows(
    acts: np.ndarray, indices: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return ((np.asarray(acts[indices], dtype=np.float32) - mean) / scale).astype(np.float32)


def main() -> None:
    args = parse_args()
    groups = pca_groups(args.manifest)
    if not 0 <= args.group_index < len(groups):
        raise IndexError(f"group-index outside 0..{len(groups) - 1}")
    group = groups[args.group_index]
    row = group[0]
    name = f"group_{args.group_index:03d}_{row['model_safe']}_layer{int(row['layer']):02d}"
    output = args.output_root / name
    metrics_path = output / "pca_concept_metrics.csv"
    model_path = output / "pca768_model.npz"
    summary_path = output / "summary.json"
    if summary_path.exists() and metrics_path.exists() and model_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == PROTOCOL:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    if len(acts) != len(records) or acts.shape[1] != 768:
        raise RuntimeError(f"unexpected activation shape: {acts.shape}")
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
    semantic_idx = train_idx
    if args.complete_case_concepts:
        complete = np.all(np.isfinite(concepts), axis=1)
        semantic_idx = semantic_idx[complete[semantic_idx]]
        validation_idx = validation_idx[complete[validation_idx]]
        test_idx = test_idx[complete[test_idx]]
    if args.semantic_train_limit > 0 and len(semantic_idx) > args.semantic_train_limit:
        rng = np.random.default_rng(SEMANTIC_SEED)
        semantic_idx = np.sort(
            rng.choice(semantic_idx, size=args.semantic_train_limit, replace=False)
        )
    family_by_concept = {
        item["concept_id"]: item["family"]
        for item in read_csv(args.concept_registry)
        if item.get("main") == "yes"
    }
    if len(concept_names) != args.expected_concepts or any(
        name not in family_by_concept for name in concept_names
    ):
        raise RuntimeError(
            f"expected registered {args.expected_concepts}-concept waveform panel"
        )

    x_train, mean, scale = normalization_from_train(acts, train_idx)
    normalization_audit = checkpoint_normalization_error(
        resolved(row["checkpoint"]), mean, scale
    )
    if max(normalization_audit.values()) > 1e-7:
        raise RuntimeError(f"PCA/SAE normalization mismatch: {normalization_audit}")

    pca = PCA(n_components=768, svd_solver="full")
    pca.fit(x_train)
    x_semantic = pca.transform(normalized_rows(acts, semantic_idx, mean, scale)).astype(np.float32)
    x_validation = pca.transform(normalized_rows(acts, validation_idx, mean, scale)).astype(np.float32)
    x_test = pca.transform(normalized_rows(acts, test_idx, mean, scale)).astype(np.float32)
    y_semantic = concepts[semantic_idx]
    train_corr = feature_concept_correlations(x_semantic, y_semantic)
    validation_corr = feature_concept_correlations(x_validation, concepts[validation_idx])
    test_corr = feature_concept_correlations(x_test, concepts[test_idx])
    validation_rows, validation_summary = selected_concept_metrics(
        train_corr, validation_corr, concept_names
    )
    test_rows, test_summary = selected_concept_metrics(train_corr, test_corr, concept_names)

    output_rows = []
    for split_name, split_rows in (("validation", validation_rows), ("test", test_rows)):
        for item in split_rows:
            output_rows.append(
                {
                    "protocol": PROTOCOL,
                    "group_index": args.group_index,
                    "model": row["model"],
                    "model_safe": row["model_safe"],
                    "layer": int(row["layer"]),
                    "relative_depth": float(row["relative_depth"]),
                    "split": split_name,
                    "concept": item["concept"],
                    "family": family_by_concept[str(item["concept"])],
                    "method": "pca_full_768",
                    "selected_feature": item["selected_feature"],
                    "train_correlation": item["train_correlation"],
                    "eval_correlation": item["eval_correlation"],
                    "sign_aligned_eval_correlation": item["sign_aligned_eval_correlation"],
                    "abs_eval_correlation": item["abs_eval_correlation"],
                    "sign_match": item["sign_match"],
                    "covered_020": int(float(item["abs_eval_correlation"]) >= 0.20),
                }
            )

    gram = np.asarray(pca.components_, dtype=np.float64) @ np.asarray(
        pca.components_, dtype=np.float64
    ).T
    orthonormal_error = float(np.max(np.abs(gram - np.eye(768))))
    test_normalized = normalized_rows(acts, test_idx, mean, scale)
    reconstructed = pca.inverse_transform(x_test).astype(np.float32)
    denominator = float(np.square(test_normalized - test_normalized.mean(0)).sum())
    reconstruction_r2 = float(
        1.0 - np.square(test_normalized - reconstructed).sum() / denominator
    )
    atomic_csv(metrics_path, output_rows)
    atomic_npz(
        model_path,
        activation_mean=mean,
        activation_scale=scale,
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        components=np.asarray(pca.components_, dtype=np.float32),
        explained_variance=np.asarray(pca.explained_variance_, dtype=np.float32),
        explained_variance_ratio=np.asarray(pca.explained_variance_ratio_, dtype=np.float32),
    )
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "group_index": args.group_index,
        "model": row["model"],
        "model_safe": row["model_safe"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
        "pca_components": 768,
        "fit_split": "complete train",
        "selection_split": "fixed 4096-record semantic train subset",
        "evaluation_split": "patient-disjoint validation and test",
        "n_train": len(train_idx),
        "n_semantic_train": len(semantic_idx),
        "n_validation": len(validation_idx),
        "n_test": len(test_idx),
        "n_concepts": len(concept_names),
        "normalization_reference": str(resolved(row["checkpoint"])),
        "normalization_audit": normalization_audit,
        "orthonormal_max_abs_error": orthonormal_error,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "test_full_rank_reconstruction_r2": reconstruction_r2,
        "validation": validation_summary,
        "test": test_summary,
        "concept_metrics": str(metrics_path),
        "pca_model": str(model_path),
    }
    if orthonormal_error > 1e-5 or reconstruction_r2 < 0.99999:
        raise RuntimeError(
            f"PCA numerical audit failed: orth={orthonormal_error}, R2={reconstruction_r2}"
        )
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
