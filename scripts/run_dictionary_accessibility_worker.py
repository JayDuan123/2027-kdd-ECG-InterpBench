#!/usr/bin/env python
"""Run one model-depth cell of the held-out dictionary accessibility benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.accessibility_calibration import feature_concept_correlations  # noqa: E402
from benchmark_v1.dictionary_accessibility import (  # noqa: E402
    SelectionProfile,
    concept_centric_profile,
    feature_centric_profile,
    matched_feature_subsets,
    nonconstant_feature_mask,
    tie_aware_auc_matrix,
)
from benchmark_v1.multiscale_sae import read_csv, standardized_concepts  # noqa: E402
from scripts.run_accessibility_baselines_v2_worker import (  # noqa: E402
    baseline_groups,
    checkpoint_normalization_audit,
)
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npz,
    encode_random_dictionary,
    encode_sae,
    load_sae,
    normalized_dense,
    resolved,
)


PROTOCOL = "dictionary_accessibility_e8_v1"
WAVEFORM_THRESHOLD = 0.20
DIAGNOSIS_THRESHOLD = 0.70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument("--random-replicates", type=int, default=20)
    parser.add_argument("--random-seed-base", type=int, default=930000)
    parser.add_argument("--matched-budget", type=int, default=768)
    parser.add_argument("--budget-replicates", type=int, default=20)
    parser.add_argument("--budget-seed-base", type=int, default=940000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=ROOT / "results/manifest/tasks_matrix.csv",
    )
    parser.add_argument(
        "--concept-registry", type=Path, default=ROOT / "configs/concepts.csv"
    )
    parser.add_argument(
        "--task-registry", type=Path, default=ROOT / "configs/tasks.csv"
    )
    parser.add_argument("--complete-case-concepts", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/dictionary_accessibility_e8_v1/workers",
    )
    return parser.parse_args()


def aligned_binary_targets(
    record_ids: list[str], rows: list[dict[str, str]]
) -> tuple[np.ndarray, list[str]]:
    if not rows:
        raise ValueError("task matrix is empty")
    names = [name for name in rows[0] if name != "ecg_id"]
    by_id = {str(row["ecg_id"]): row for row in rows}
    values = np.empty((len(record_ids), len(names)), dtype=np.float32)
    for index, ecg_id in enumerate(record_ids):
        row = by_id.get(str(ecg_id))
        if row is None:
            raise KeyError(f"task row missing for ecg_id={ecg_id}")
        values[index] = [float(row[name]) for name in names]
    if not np.isin(values, [0.0, 1.0]).all():
        raise ValueError("task matrix contains non-binary values")
    return values, names


def record_hash(record_ids: list[str]) -> str:
    payload = "\n".join(map(str, record_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def descriptive_scores(profile: SelectionProfile, center: float) -> np.ndarray:
    if center == 0.0:
        return np.abs(profile.test_value)
    return center + np.abs(profile.test_value - center)


def feature_summary_row(
    identity: dict[str, Any],
    profile: SelectionProfile,
    live: np.ndarray,
    *,
    target_type: str,
    metric: str,
    center: float,
    threshold: float,
    method: str,
    dictionary_width: int,
    candidate_features: np.ndarray | None = None,
    replicate_kind: str,
    replicate: int,
    sae_seed: int | None = None,
    random_seed: int | None = None,
    budget_replicate: int | None = None,
) -> dict[str, Any]:
    candidates = (
        np.arange(len(profile.test_value), dtype=np.int64)
        if candidate_features is None
        else np.asarray(candidate_features, dtype=np.int64)
    )
    oriented = profile.test_oriented_value[candidates]
    descriptive = descriptive_scores(profile, center)[candidates]
    live_candidates = np.asarray(live[candidates], dtype=bool)
    live_values = oriented[live_candidates]
    if len(live_values) == 0:
        raise RuntimeError(f"no live features for {identity} {method} {target_type}")
    return {
        **identity,
        "target_type": target_type,
        "metric": metric,
        "method": method,
        "dictionary_width": int(dictionary_width),
        "candidate_budget": int(len(candidates)),
        "replicate_kind": replicate_kind,
        "replicate": int(replicate),
        "sae_seed": "" if sae_seed is None else int(sae_seed),
        "random_seed": "" if random_seed is None else int(random_seed),
        "budget_replicate": "" if budget_replicate is None else int(budget_replicate),
        "n_live": int(live_candidates.sum()),
        "live_fraction": float(live_candidates.mean()),
        "selected_target_count": int(np.unique(profile.selected_index[candidates]).size),
        "mean_test_oriented_score": float(np.mean(oriented)),
        "median_test_oriented_score": float(np.median(oriented)),
        "q90_test_oriented_score": float(np.quantile(oriented, 0.90)),
        "q95_test_oriented_score": float(np.quantile(oriented, 0.95)),
        "max_test_oriented_score": float(np.max(oriented)),
        "mean_test_descriptive_score": float(np.mean(descriptive)),
        "n_above_primary": int(np.sum(oriented >= threshold)),
        "fraction_above_primary": float(np.mean(oriented >= threshold)),
        "n_live_above_primary": int(np.sum(live_values >= threshold)),
        "live_fraction_above_primary": float(np.mean(live_values >= threshold)),
        "primary_threshold": float(threshold),
    }


def target_rows(
    identity: dict[str, Any],
    profile: SelectionProfile,
    target_names: list[str],
    family_by_target: dict[str, str],
    *,
    target_type: str,
    metric: str,
    center: float,
    threshold: float,
    method: str,
    dictionary_width: int,
    candidate_budget: int,
    replicate_kind: str,
    replicate: int,
    sae_seed: int | None = None,
    random_seed: int | None = None,
    budget_replicate: int | None = None,
) -> list[dict[str, Any]]:
    descriptive = descriptive_scores(profile, center)
    return [
        {
            **identity,
            "target_type": target_type,
            "target": target,
            "family": family_by_target[target],
            "metric": metric,
            "method": method,
            "dictionary_width": int(dictionary_width),
            "candidate_budget": int(candidate_budget),
            "replicate_kind": replicate_kind,
            "replicate": int(replicate),
            "sae_seed": "" if sae_seed is None else int(sae_seed),
            "random_seed": "" if random_seed is None else int(random_seed),
            "budget_replicate": "" if budget_replicate is None else int(budget_replicate),
            "selected_feature": int(profile.selected_index[index]),
            "train_value": float(profile.train_value[index]),
            "test_value": float(profile.test_value[index]),
            "test_oriented_score": float(profile.test_oriented_value[index]),
            "test_descriptive_score": float(descriptive[index]),
            "covered_primary": int(profile.test_oriented_value[index] >= threshold),
            "primary_threshold": float(threshold),
        }
        for index, target in enumerate(target_names)
    ]


def associations(
    train_features,
    test_features,
    train_concepts: np.ndarray,
    test_concepts: np.ndarray,
    train_tasks: np.ndarray,
    test_tasks: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray, float, float, str]]:
    return {
        "waveform": (
            feature_concept_correlations(train_features, train_concepts),
            feature_concept_correlations(test_features, test_concepts),
            0.0,
            WAVEFORM_THRESHOLD,
            "pearson_r",
        ),
        "diagnosis": (
            tie_aware_auc_matrix(train_features, train_tasks),
            tie_aware_auc_matrix(test_features, test_tasks),
            0.5,
            DIAGNOSIS_THRESHOLD,
            "auroc",
        ),
    }


def add_representation(
    *,
    identity: dict[str, Any],
    method: str,
    dictionary_width: int,
    replicate_kind: str,
    replicate: int,
    association_matrices: dict[str, tuple[np.ndarray, np.ndarray, float, float, str]],
    live: np.ndarray,
    target_names: dict[str, list[str]],
    family_by_target: dict[str, dict[str, str]],
    feature_rows: list[dict[str, Any]],
    concept_rows: list[dict[str, Any]],
    raw_profiles: dict[str, np.ndarray],
    raw_prefix: str,
    matched_subsets: list[np.ndarray] | None = None,
    matched_method: str | None = None,
    budget_replicate_offset: int = 0,
    sae_seed: int | None = None,
    random_seed: int | None = None,
) -> None:
    for target_type, (train_values, test_values, center, threshold, metric) in association_matrices.items():
        feature_profile = feature_centric_profile(train_values, test_values, center=center)
        concept_profile = concept_centric_profile(train_values, test_values, center=center)
        raw_profiles[f"{raw_prefix}_{target_type}_feature_score"] = (
            feature_profile.test_oriented_value
        )
        raw_profiles[f"{raw_prefix}_{target_type}_selected_target"] = (
            feature_profile.selected_index
        )
        feature_rows.append(
            feature_summary_row(
                identity,
                feature_profile,
                live,
                target_type=target_type,
                metric=metric,
                center=center,
                threshold=threshold,
                method=method,
                dictionary_width=dictionary_width,
                replicate_kind=replicate_kind,
                replicate=replicate,
                sae_seed=sae_seed,
                random_seed=random_seed,
            )
        )
        concept_rows.extend(
            target_rows(
                identity,
                concept_profile,
                target_names[target_type],
                family_by_target[target_type],
                target_type=target_type,
                metric=metric,
                center=center,
                threshold=threshold,
                method=method,
                dictionary_width=dictionary_width,
                candidate_budget=dictionary_width,
                replicate_kind=replicate_kind,
                replicate=replicate,
                sae_seed=sae_seed,
                random_seed=random_seed,
            )
        )
        if matched_subsets is None or matched_method is None:
            continue
        for local_budget_replicate, subset in enumerate(matched_subsets):
            budget_replicate = budget_replicate_offset + local_budget_replicate
            matched_profile = concept_centric_profile(
                train_values,
                test_values,
                center=center,
                candidate_features=subset,
            )
            feature_rows.append(
                feature_summary_row(
                    identity,
                    feature_profile,
                    live,
                    target_type=target_type,
                    metric=metric,
                    center=center,
                    threshold=threshold,
                    method=matched_method,
                    dictionary_width=dictionary_width,
                    candidate_features=subset,
                    replicate_kind="budget",
                    replicate=budget_replicate,
                    sae_seed=sae_seed,
                    random_seed=random_seed,
                    budget_replicate=budget_replicate,
                )
            )
            concept_rows.extend(
                target_rows(
                    identity,
                    matched_profile,
                    target_names[target_type],
                    family_by_target[target_type],
                    target_type=target_type,
                    metric=metric,
                    center=center,
                    threshold=threshold,
                    method=matched_method,
                    dictionary_width=dictionary_width,
                    candidate_budget=len(subset),
                    replicate_kind="budget",
                    replicate=budget_replicate,
                    sae_seed=sae_seed,
                    random_seed=random_seed,
                    budget_replicate=budget_replicate,
                )
            )


def main() -> None:
    args = parse_args()
    groups = baseline_groups(args.manifest, args.expansion)
    if not 0 <= args.group_index < len(groups):
        raise IndexError(f"group index outside 0..{len(groups) - 1}")
    rows = groups[args.group_index]
    row = rows[0]
    if args.matched_budget != int(row["d_hidden"]):
        raise ValueError("matched budget must equal the native hidden width")
    if args.random_replicates < 1 or args.budget_replicates < 1:
        raise ValueError("replicate counts must be positive")

    cell_name = f"group_{args.group_index:03d}_{row['model_safe']}_layer{int(row['layer']):02d}"
    cell_root = args.output_root / cell_name
    feature_path = cell_root / "feature_profiles.csv"
    target_path = cell_root / "target_profiles.csv"
    raw_path = cell_root / "feature_score_arrays.npz"
    summary_path = cell_root / "summary.json"
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if (
            existing.get("status") == "complete"
            and existing.get("protocol") == PROTOCOL
            and feature_path.exists()
            and target_path.exists()
            and raw_path.exists()
        ):
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    normalization_audit = checkpoint_normalization_audit(rows)
    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    if len(acts) != len(records):
        raise RuntimeError("activation and record counts differ")
    record_ids = [record["ecg_id"] for record in records]
    splits = np.asarray([record["split"] for record in records])
    train_idx = np.flatnonzero(splits == "train")
    test_idx = np.flatnonzero(splits == "test")
    concepts, concept_names, _, _ = standardized_concepts(
        record_ids,
        read_csv(args.concepts),
        splits == "train",
        preserve_missing=args.complete_case_concepts,
    )
    tasks, task_names = aligned_binary_targets(record_ids, read_csv(args.tasks))
    if args.complete_case_concepts:
        complete = np.all(np.isfinite(concepts), axis=1)
        train_idx = train_idx[complete[train_idx]]
        test_idx = test_idx[complete[test_idx]]
    semantic_train_idx = train_idx
    if args.semantic_train_limit and len(train_idx) > args.semantic_train_limit:
        semantic_train_idx = np.sort(
            np.random.default_rng(20260714).choice(
                train_idx, size=args.semantic_train_limit, replace=False
            )
        )
    concept_families = {
        registry["concept_id"]: registry["family"]
        for registry in read_csv(args.concept_registry)
        if registry.get("main") == "yes"
    }
    task_families = {
        registry["task_id"]: registry["task_family"]
        for registry in read_csv(args.task_registry)
        if registry.get("main") in {"yes", "conditional"}
    }
    target_names = {"waveform": concept_names, "diagnosis": task_names}
    family_by_target = {"waveform": concept_families, "diagnosis": task_families}
    train_concepts, test_concepts = concepts[semantic_train_idx], concepts[test_idx]
    train_tasks, test_tasks = tasks[semantic_train_idx], tasks[test_idx]
    for name, matrix in (("train", train_tasks), ("test", test_tasks)):
        positive = matrix.sum(axis=0)
        if np.any(positive < 10) or np.any(len(matrix) - positive < 10):
            raise RuntimeError(f"insufficient binary task support in {name}")

    identity = {
        "group_index": args.group_index,
        "model": row["model"],
        "model_safe": row["model_safe"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
    }
    feature_rows: list[dict[str, Any]] = []
    concept_rows: list[dict[str, Any]] = []
    raw_profiles: dict[str, np.ndarray] = {}
    budget_subsets = matched_feature_subsets(
        int(row["N"]),
        args.matched_budget,
        args.budget_replicates,
        args.budget_seed_base,
    )

    normalization_model, checkpoint_path = load_sae(row, args.device)
    dense_train = normalized_dense(normalization_model, acts, semantic_train_idx)
    dense_test = normalized_dense(normalization_model, acts, test_idx)
    dense_associations = associations(
        dense_train,
        dense_test,
        train_concepts,
        test_concepts,
        train_tasks,
        test_tasks,
    )
    dense_live = nonconstant_feature_mask(dense_train) & nonconstant_feature_mask(dense_test)
    add_representation(
        identity=identity,
        method="dense_native_768",
        dictionary_width=dense_train.shape[1],
        replicate_kind="native",
        replicate=0,
        association_matrices=dense_associations,
        live=dense_live,
        target_names=target_names,
        family_by_target=family_by_target,
        feature_rows=feature_rows,
        concept_rows=concept_rows,
        raw_profiles=raw_profiles,
        raw_prefix="dense",
    )
    raw_profiles["dense_live"] = dense_live

    sae_live_arrays = []
    for seed_index, seed_row in enumerate(rows):
        model, _ = load_sae(seed_row, args.device)
        z_train = encode_sae(model, acts, semantic_train_idx, args.batch_size, args.device)
        z_test = encode_sae(model, acts, test_idx, args.batch_size, args.device)
        sae_associations = associations(
            z_train,
            z_test,
            train_concepts,
            test_concepts,
            train_tasks,
            test_tasks,
        )
        live = nonconstant_feature_mask(z_train) & nonconstant_feature_mask(z_test)
        sae_live_arrays.append(live)
        add_representation(
            identity=identity,
            method="sae_full_6144",
            dictionary_width=int(seed_row["N"]),
            replicate_kind="sae_seed",
            replicate=seed_index,
            association_matrices=sae_associations,
            live=live,
            target_names=target_names,
            family_by_target=family_by_target,
            feature_rows=feature_rows,
            concept_rows=concept_rows,
            raw_profiles=raw_profiles,
            raw_prefix=f"sae_seed{int(seed_row['seed'])}",
            matched_subsets=budget_subsets,
            matched_method="sae_matched_768",
            sae_seed=int(seed_row["seed"]),
        )
        del z_train, z_test, sae_associations, model
    raw_profiles["sae_live"] = np.stack(sae_live_arrays)

    random_live_arrays = []
    random_seeds = []
    for replicate in range(args.random_replicates):
        random_seed = args.random_seed_base + replicate
        random_seeds.append(random_seed)
        z_train = encode_random_dictionary(
            normalization_model,
            acts,
            semantic_train_idx,
            args.batch_size,
            args.device,
            random_seed,
        )
        z_test = encode_random_dictionary(
            normalization_model,
            acts,
            test_idx,
            args.batch_size,
            args.device,
            random_seed,
        )
        random_associations = associations(
            z_train,
            z_test,
            train_concepts,
            test_concepts,
            train_tasks,
            test_tasks,
        )
        live = nonconstant_feature_mask(z_train) & nonconstant_feature_mask(z_test)
        random_live_arrays.append(live)
        subset = [budget_subsets[replicate % len(budget_subsets)]]
        add_representation(
            identity=identity,
            method="random_full_6144",
            dictionary_width=int(row["N"]),
            replicate_kind="random_seed",
            replicate=replicate,
            association_matrices=random_associations,
            live=live,
            target_names=target_names,
            family_by_target=family_by_target,
            feature_rows=feature_rows,
            concept_rows=concept_rows,
            raw_profiles=raw_profiles,
            raw_prefix=f"random_seed{random_seed}",
            matched_subsets=subset,
            matched_method="random_matched_768",
            budget_replicate_offset=replicate,
            random_seed=random_seed,
        )
        del z_train, z_test, random_associations
    raw_profiles["random_live"] = np.stack(random_live_arrays)

    atomic_csv(feature_path, feature_rows)
    atomic_csv(target_path, concept_rows)
    atomic_npz(
        raw_path,
        waveform_targets=np.asarray(concept_names),
        diagnosis_targets=np.asarray(task_names),
        test_ecg_ids=np.asarray([record_ids[index] for index in test_idx]),
        random_seeds=np.asarray(random_seeds, dtype=np.int64),
        budget_subsets=np.stack(budget_subsets),
        **raw_profiles,
    )
    payload = {
        "status": "complete",
        "protocol": PROTOCOL,
        **identity,
        "expansion_E": int(row["expansion_E"]),
        "N": int(row["N"]),
        "k": int(row["k"]),
        "native_width": int(row["d_hidden"]),
        "sae_seeds": [int(seed_row["seed"]) for seed_row in rows],
        "random_seeds": random_seeds,
        "budget_seeds": [args.budget_seed_base + value for value in range(args.budget_replicates)],
        "random_replicates": args.random_replicates,
        "budget_replicates": args.budget_replicates,
        "normalization_checkpoint": str(checkpoint_path),
        "normalization_audit": normalization_audit,
        "n_waveform_targets": len(concept_names),
        "n_diagnosis_targets": len(task_names),
        "n_train": len(train_idx),
        "n_semantic_train": len(semantic_train_idx),
        "n_test": len(test_idx),
        "test_record_sha256": record_hash([record_ids[index] for index in test_idx]),
        "feature_profile_rows": len(feature_rows),
        "target_profile_rows": len(concept_rows),
        "feature_profiles": str(feature_path),
        "target_profiles": str(target_path),
        "feature_score_arrays": str(raw_path),
        "selection_policy": "target and direction selected on train; test used once for frozen evaluation",
        "pooling_policy": "identical precomputed record-level activations for dense, SAE, and random",
        "claim_boundary": "dictionary association and accessibility, not mechanism or causal use",
    }
    atomic_json(summary_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
