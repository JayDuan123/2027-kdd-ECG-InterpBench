#!/usr/bin/env python
"""Fit and evaluate one model/cohort/seed fair-comparison task."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_common import (  # noqa: E402
    bootstrap_steering_metrics,
    group_bootstrap_weights,
    interval_and_p,
)
from scripts.method_comparison_common import (  # noqa: E402
    BASE,
    COMMON_K,
    COMMON_RANK,
    LABEL_BUDGETS,
    METHOD_METRICS,
    METHODS,
    RATE_DISTORTION_K,
    RECONSTRUCTIVE_METHODS,
    direction_delta,
    direction_control_logit_deltas,
    draw_random_component_groups,
    hard_topk,
    logit_delta,
    norm_match,
    norm_matched_logit_delta,
    random_component_deltas,
    random_direction_deltas,
    random_unit_directions,
    component_control_logit_deltas,
    reconstruction_metrics,
    selected_component_delta,
    stable_seed,
    stable_subset,
    write_json,
)
from scripts.method_comparison_models import (  # noqa: E402
    encode_decode_sae,
    fit_or_load_ica,
    fit_or_load_pca,
    fit_or_load_random_basis,
    fit_or_load_semi_nmf,
    semi_nmf_transform,
    train_or_load_common_sae,
)
from scripts.run_external_sae_steering_task import (  # noqa: E402
    load_activations,
    threshold_at_specificity,
)


SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "steering_summary", ROOT / "scripts" / "summarize_steering_benchmark.py"
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=BASE / "manifest.csv")
    parser.add_argument("--out", type=Path, default=BASE / "workers")
    parser.add_argument("--rank", type=int, default=COMMON_RANK)
    parser.add_argument("--k", type=int, default=COMMON_K)
    parser.add_argument("--max-train", type=int, default=8192)
    parser.add_argument("--max-validation", type=int, default=4096)
    parser.add_argument("--sae-steps", type=int, default=2000)
    parser.add_argument("--sae-batch", type=int, default=512)
    parser.add_argument("--sae-lr", type=float, default=3e-4)
    parser.add_argument("--semi-nmf-iterations", type=int, default=80)
    parser.add_argument("--semi-nmf-transform-iterations", type=int, default=50)
    parser.add_argument("--ica-max-iter", type=int, default=600)
    parser.add_argument("--ica-tolerance", type=float, default=1e-4)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force-evaluation", action="store_true")
    parser.add_argument("--refit-ica", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def stratified_subset(
    indices: np.ndarray,
    identifiers: np.ndarray,
    label_map: dict[str, np.ndarray],
    limit: int,
    *key: object,
) -> np.ndarray:
    """Deterministically retain both classes for every target before filling."""
    indices = np.asarray(indices, dtype=int)
    if limit <= 0 or len(indices) <= limit:
        return indices
    per_class = max(2, min(256, limit // max(2 * len(label_map), 1)))
    reserved: set[int] = set()
    for target, labels in label_map.items():
        values = np.asarray(labels, dtype=float)
        finite = indices[np.isfinite(values[indices])]
        unique = np.unique(values[finite])
        if len(unique) < 2:
            raise RuntimeError(f"Target {target} has fewer than two classes in the available split")
        if len(unique) <= 2:
            strata = [finite[values[finite] == value] for value in (unique.min(), unique.max())]
        else:
            low, high = np.quantile(values[finite], [0.25, 0.75])
            strata = [finite[values[finite] <= low], finite[values[finite] >= high]]
        for stratum_index, stratum in enumerate(strata):
            if not len(stratum):
                raise RuntimeError(f"Target {target} has an empty required stratum")
            selected = stable_subset(
                stratum,
                identifiers,
                min(per_class, len(stratum)),
                *key,
                target,
                stratum_index,
            )
            reserved.update(map(int, selected))
    remaining_budget = max(0, limit - len(reserved))
    available = np.asarray([index for index in indices if int(index) not in reserved], dtype=int)
    filler = (
        np.empty(0, dtype=int)
        if remaining_budget == 0
        else stable_subset(available, identifiers, remaining_budget, *key, "filler")
    )
    return np.asarray(sorted(reserved.union(map(int, filler))), dtype=int)


def method_point(data: dict[str, np.ndarray], result: dict) -> dict[str, float]:
    point = SUMMARY.one_stats(data, result, [])
    return {metric: float(point[metric]) for metric in METHOD_METRICS}


def decoder_offset_for_sae(model) -> tuple[np.ndarray, np.ndarray]:
    decoder = (
        model.W_dec.detach().cpu().numpy().T
        * model.sigma.detach().cpu().numpy()[None, :]
    ).astype(np.float32)
    offset = (
        model.b_dec.detach().cpu().numpy() * model.sigma.detach().cpu().numpy()
        + model.mu.detach().cpu().numpy()
    ).astype(np.float32)
    return decoder, offset


def sparse_reconstruction(
    codes: np.ndarray,
    decoder: np.ndarray,
    offset: np.ndarray,
    k: int,
    positive_only: bool,
) -> np.ndarray:
    return hard_topk(codes, k, positive_only=positive_only) @ decoder + offset[None, :]


def fit_sparse_probe(
    path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    if path.exists():
        payload = joblib.load(path)
        return np.asarray(payload["direction"], dtype=np.float32), payload["metadata"]
    valid_train = np.isfinite(y_train)
    valid_validation = np.isfinite(y_validation)
    best = None
    for c_value in (0.01, 0.1, 1.0):
        model = LogisticRegression(
            penalty="l1",
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
        )
        model.fit(x_train[valid_train], y_train[valid_train].astype(int))
        score = model.decision_function(x_validation[valid_validation])
        auc = float(roc_auc_score(y_validation[valid_validation].astype(int), score))
        candidate = (auc, -c_value, model)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    model = best[2]
    direction = np.asarray(model.coef_).reshape(-1).astype(np.float32)
    metadata = {
        "validation_auroc": float(best[0]),
        "C": float(model.C),
        "nonzero_coefficients": int(np.count_nonzero(direction)),
        "dimension": len(direction),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    joblib.dump({"direction": direction, "metadata": metadata}, temporary)
    temporary.replace(path)
    return direction, metadata


def existing_sae_state(row: pd.Series, scaler, x_test_raw: np.ndarray) -> dict:
    import torch

    checkpoint = Path(row.existing_sae_checkpoint)
    cache = Path(row.existing_sae_cache)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["model"]
    z_test = np.load(cache / "zte.npy", mmap_mode="r")
    centroid = np.asarray(np.load(cache / "centroid.npy", mmap_mode="r"), dtype=np.float32)
    if len(z_test) != len(x_test_raw):
        raise RuntimeError(f"Existing SAE test-code mismatch: {len(z_test)} != {len(x_test_raw)}")
    w_dec = state["W_dec"].detach().cpu().numpy().astype(np.float32)
    b_dec = state["b_dec"].detach().cpu().numpy().astype(np.float32)
    mu = state["mu"].detach().cpu().numpy().astype(np.float32)
    sigma = state["sigma"].detach().cpu().numpy().astype(np.float32)
    decoder = (w_dec.T * (sigma / scaler.scale_)[None, :]).astype(np.float32)
    reconstruction = np.empty_like(x_test_raw, dtype=np.float32)
    sparse = np.empty_like(x_test_raw, dtype=np.float32)
    active_total = 0
    for start in range(0, len(z_test), 512):
        codes = np.asarray(z_test[start : start + 512], dtype=np.float32)
        normalized = codes @ w_dec.T + b_dec[None, :]
        raw = normalized * sigma[None, :] + mu[None, :]
        reconstruction[start : start + len(codes)] = scaler.transform(raw).astype(np.float32)
        sparse_codes = hard_topk(codes, COMMON_K, positive_only=True)
        sparse_normalized = sparse_codes @ w_dec.T + b_dec[None, :]
        sparse_raw = sparse_normalized * sigma[None, :] + mu[None, :]
        sparse[start : start + len(codes)] = scaler.transform(sparse_raw).astype(np.float32)
        active_total += int(np.count_nonzero(codes))
    return {
        "checkpoint": checkpoint,
        "cache": cache,
        "codes": z_test,
        "centroid": centroid,
        "decoder": decoder,
        "reconstruction": reconstruction,
        "sparse_reconstruction": sparse,
        "active_mean": active_total / len(z_test),
        "config": saved["config"],
    }


def existing_target_deltas(
    existing: dict,
    result: dict,
    n_random: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    codes = existing["codes"]
    centroid = existing["centroid"]
    decoder = existing["decoder"]

    def one(indices: list[int]) -> np.ndarray:
        selected = np.asarray(indices, dtype=int)
        dz = centroid[selected][None, :] - np.asarray(codes[:, selected], dtype=np.float32)
        return (dz @ decoder[selected]).astype(np.float32)

    selected_delta = one(result["selected_atoms"]["top5"])
    random = [one(group) for group in result["random_groups"][:n_random]]
    return selected_delta, random


def build_common_data(bundle: dict, labels: dict, indices: np.ndarray, base: np.ndarray, thresholds: np.ndarray) -> dict:
    names = list(bundle["targets"])
    kinds = []
    means = []
    standard_deviations = []
    for name in names:
        values = np.asarray(labels[name], dtype=float)
        finite = values[np.isfinite(values)]
        binary = len(np.unique(finite)) <= 2
        kinds.append("binary" if binary else "continuous")
        means.append(np.nan if binary else float(np.nanmean(values)))
        standard_deviations.append(np.nan if binary else float(np.nanstd(values)))
    return {
        "patient_ids": np.asarray(bundle.get("group_ids", bundle["record_ids"]))[indices].astype("U64"),
        "target_names": np.asarray(names, dtype="U64"),
        "target_types": np.asarray(kinds, dtype="U16"),
        "labels": np.column_stack([labels[name][indices] for name in names]).astype(np.float32),
        "baseline_logits": np.asarray(base, dtype=np.float32),
        "thresholds_95spec": np.asarray(thresholds, dtype=np.float32),
        "continuous_target_means": np.asarray(means, dtype=np.float32),
        "continuous_target_stds": np.asarray(standard_deviations, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.task_index < 0 or args.task_index >= len(manifest):
        raise ValueError(f"task-index must be in 0..{len(manifest) - 1}")
    row = manifest.iloc[args.task_index]
    worker = args.out / f"task_{args.task_index:03d}_{row.model_suffix}_{row.cohort}_seed{int(row.seed)}"
    complete = worker / "complete.json"
    if complete.exists() and not args.force_evaluation:
        print(f"already complete: {complete}")
        return
    if args.rank != int(row.dimension) and args.rank > int(row.dimension):
        raise RuntimeError(f"rank={args.rank} exceeds d={row.dimension}")
    preflight = {
        "task_index": args.task_index,
        "model": row.model,
        "model_suffix": row.model_suffix,
        "cohort": row.cohort,
        "seed": int(row.seed),
        "dimension": int(row.dimension),
        "targets": int(row.targets),
        "rank": args.rank,
        "k": args.k,
        "head_exists": Path(row.head_path).exists(),
        "checkpoint_exists": Path(row.existing_sae_checkpoint).exists(),
        "cache_complete": (Path(row.existing_sae_cache) / "complete.json").exists(),
    }
    if not all(preflight[key] for key in ("head_exists", "checkpoint_exists", "cache_complete")):
        raise RuntimeError(f"Preflight failed: {preflight}")
    if args.preflight_only:
        print(preflight)
        return

    worker.mkdir(parents=True, exist_ok=True)
    fit_root = worker / "fits"
    fit_root.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(Path(row.head_path))
    names = list(bundle["targets"])
    labels = {name: np.asarray(bundle["heads"][name]["labels"], dtype=float) for name in names}
    split = np.asarray(bundle["split"]).astype(str)
    record_ids = np.asarray(bundle["record_ids"]).astype(str)
    activations, activation_ids = load_activations(Path(row.activation_root))
    if not np.array_equal(record_ids, activation_ids.astype(str)):
        raise RuntimeError("Activation/head record order mismatch")
    train_all = np.flatnonzero(split == "train")
    validation_all = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    fit_train = stratified_subset(
        train_all,
        record_ids,
        labels,
        args.max_train,
        "method-comparison-fit",
        row.model_suffix,
        row.cohort,
        int(row.seed),
    )
    fit_validation = stratified_subset(
        validation_all,
        record_ids,
        labels,
        args.max_validation,
        "method-comparison-validation",
        row.model_suffix,
        row.cohort,
        int(row.seed),
    )
    scaler = bundle["scaler"]
    x_train = scaler.transform(np.asarray(activations[fit_train], dtype=np.float32)).astype(np.float32)
    x_validation = scaler.transform(np.asarray(activations[fit_validation], dtype=np.float32)).astype(np.float32)
    x_test_raw = np.asarray(activations[test_indices], dtype=np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)
    del activations
    gc.collect()

    device = args.device
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA task requested but torch.cuda.is_available() is false")

    seed = int(row.seed)
    representations: dict[str, dict] = {}
    reconstruction_rows = []
    fit_diagnostics: dict[str, dict] = {}

    sae, sae_payload = train_or_load_common_sae(
        fit_root / "sae_common64.pt",
        x_train,
        x_train.shape[1],
        args.rank,
        args.k,
        seed,
        args.sae_steps,
        args.sae_batch,
        args.sae_lr,
        device,
    )
    sae_train_codes, _ = encode_decode_sae(sae, x_train, device, args.sae_batch)
    sae_test_codes, sae_test_reconstruction = encode_decode_sae(sae, x_test, device, args.sae_batch)
    sae_decoder, sae_offset = decoder_offset_for_sae(sae)
    representations["sae_common64"] = {
        "train_codes": sae_train_codes,
        "test_codes": sae_test_codes,
        "decoder": sae_decoder,
        "positive_only": True,
        "offset": sae_offset,
    }
    fit_diagnostics["sae_common64"] = {
        "steps": int(sae_payload["steps"]),
        "best_training_loss": float(sae_payload["best_training_loss"]),
    }
    reconstruction_by_method = {
        "sae_common64": (
            sae_test_reconstruction,
            sparse_reconstruction(sae_test_codes, sae_decoder, sae_offset, args.k, True),
            float(np.count_nonzero(sae_test_codes) / len(sae_test_codes)),
        )
    }

    pca = fit_or_load_pca(fit_root / "pca64.joblib", x_train, args.rank, seed)
    pca_train = pca.transform(x_train).astype(np.float32)
    pca_test = pca.transform(x_test).astype(np.float32)
    pca_decoder = np.asarray(pca.components_, dtype=np.float32)
    representations["pca64"] = {
        "train_codes": pca_train,
        "test_codes": pca_test,
        "decoder": pca_decoder,
        "positive_only": False,
        "offset": np.asarray(pca.mean_, dtype=np.float32),
    }
    reconstruction_by_method["pca64"] = (
        pca.inverse_transform(pca_test).astype(np.float32),
        sparse_reconstruction(pca_test, pca_decoder, np.asarray(pca.mean_), args.k, False),
        float(args.rank),
    )
    fit_diagnostics["pca64"] = {"explained_variance_sum": float(pca.explained_variance_ratio_.sum())}

    ica, ica_converged = fit_or_load_ica(
        fit_root / "ica64.joblib",
        x_train,
        args.rank,
        seed,
        args.ica_max_iter,
        args.ica_tolerance,
        force=args.refit_ica,
    )
    ica_train = ica.transform(x_train).astype(np.float32)
    ica_test = ica.transform(x_test).astype(np.float32)
    ica_decoder = np.asarray(ica.mixing_.T, dtype=np.float32)
    representations["ica64"] = {
        "train_codes": ica_train,
        "test_codes": ica_test,
        "decoder": ica_decoder,
        "positive_only": False,
        "offset": np.asarray(ica.mean_, dtype=np.float32),
    }
    reconstruction_by_method["ica64"] = (
        ica.inverse_transform(ica_test).astype(np.float32),
        sparse_reconstruction(ica_test, ica_decoder, np.asarray(ica.mean_), args.k, False),
        float(args.rank),
    )
    fit_diagnostics["ica64"] = {
        "converged": bool(ica_converged),
        "iterations": int(ica.n_iter_),
        "max_iterations": args.ica_max_iter,
    }

    semi_payload, semi_diagnostics = fit_or_load_semi_nmf(
        fit_root / "semi_nmf64.npz",
        x_train,
        args.rank,
        seed,
        args.semi_nmf_iterations,
        device,
    )
    semi_train = semi_nmf_transform(
        x_train, semi_payload, args.semi_nmf_transform_iterations, device
    )
    semi_test = semi_nmf_transform(
        x_test, semi_payload, args.semi_nmf_transform_iterations, device
    )
    semi_decoder = np.asarray(semi_payload["decoder"], dtype=np.float32)
    semi_mean = np.asarray(semi_payload["mean"], dtype=np.float32)
    representations["semi_nmf64"] = {
        "train_codes": semi_train,
        "test_codes": semi_test,
        "decoder": semi_decoder,
        "positive_only": True,
        "offset": semi_mean,
    }
    reconstruction_by_method["semi_nmf64"] = (
        semi_test @ semi_decoder + semi_mean[None, :],
        sparse_reconstruction(semi_test, semi_decoder, semi_mean, args.k, True),
        float(args.rank),
    )
    fit_diagnostics["semi_nmf64"] = semi_diagnostics

    random_payload = fit_or_load_random_basis(
        fit_root / "random_basis64.npz",
        x_train.shape[1],
        args.rank,
        stable_seed("random-basis", row.model_suffix, row.cohort, seed),
    )
    random_mean = x_train.mean(axis=0).astype(np.float32)
    random_basis = np.asarray(random_payload["basis"], dtype=np.float32)
    random_train = (x_train - random_mean) @ random_basis
    random_test = (x_test - random_mean) @ random_basis
    random_decoder = random_basis.T
    representations["random_basis64"] = {
        "train_codes": random_train,
        "test_codes": random_test,
        "decoder": random_decoder,
        "positive_only": False,
        "offset": random_mean,
    }
    reconstruction_by_method["random_basis64"] = (
        random_test @ random_decoder + random_mean[None, :],
        sparse_reconstruction(random_test, random_decoder, random_mean, args.k, False),
        float(args.rank),
    )
    fit_diagnostics["random_basis64"] = {"orthogonality_max_abs_error": float(
        np.abs(random_basis.T @ random_basis - np.eye(args.rank)).max()
    )}

    rate_distortion_rows = []
    for method in RECONSTRUCTIVE_METHODS:
        dense, sparse, active_mean = reconstruction_by_method[method]
        dense_metrics = reconstruction_metrics(x_test, dense)
        sparse_metrics = reconstruction_metrics(x_test, sparse)
        reconstruction_rows.append(
            {
                "task_index": args.task_index,
                "model": row.model,
                "model_suffix": row.model_suffix,
                "cohort": row.cohort,
                "seed": seed,
                "method": method,
                "rank": args.rank,
                "activation_budget_k": args.k,
                "mean_active_coefficients": active_mean,
                **{f"dense_{key}": value for key, value in dense_metrics.items()},
                **{f"topk_{key}": value for key, value in sparse_metrics.items()},
            }
        )
        representation = representations[method]
        allowed_budgets = [
            budget
            for budget in RATE_DISTORTION_K
            if budget <= args.rank
            and (method != "sae_common64" or budget <= args.k)
        ]
        for code_budget in allowed_budgets:
            rate_reconstruction = sparse_reconstruction(
                representation["test_codes"],
                representation["decoder"],
                representation["offset"],
                code_budget,
                representation["positive_only"],
            )
            rate_metrics = reconstruction_metrics(x_test, rate_reconstruction)
            rate_distortion_rows.append(
                {
                    "task_index": args.task_index,
                    "model": row.model,
                    "model_suffix": row.model_suffix,
                    "cohort": row.cohort,
                    "seed": seed,
                    "method": method,
                    "rank": args.rank,
                    "code_budget_k": code_budget,
                    **rate_metrics,
                }
            )
        del dense, sparse
    del reconstruction_by_method

    existing = existing_sae_state(row, scaler, x_test_raw)
    existing_dense = reconstruction_metrics(x_test, existing["reconstruction"])
    existing_sparse = reconstruction_metrics(x_test, existing["sparse_reconstruction"])
    reconstruction_rows.append(
        {
            "task_index": args.task_index,
            "model": row.model,
            "model_suffix": row.model_suffix,
            "cohort": row.cohort,
            "seed": seed,
            "method": "sae_existing_8d",
            "rank": int(existing["config"]["n_features"]),
            "activation_budget_k": int(existing["config"]["k"]),
            "mean_active_coefficients": float(existing["active_mean"]),
            **{f"dense_{key}": value for key, value in existing_dense.items()},
            **{f"topk_{key}": value for key, value in existing_sparse.items()},
        }
    )
    del existing["reconstruction"], existing["sparse_reconstruction"]

    coefficients = np.vstack(
        [np.asarray(bundle["heads"][name]["clf"].coef_).reshape(-1) for name in names]
    ).astype(np.float32)
    base_validation = np.column_stack(
        [bundle["heads"][name]["clf"].decision_function(x_validation) for name in names]
    ).astype(np.float32)
    base_test = np.column_stack(
        [bundle["heads"][name]["clf"].decision_function(x_test) for name in names]
    ).astype(np.float32)
    thresholds = np.asarray(
        [
            threshold_at_specificity(
                labels[name][fit_validation].astype(int), base_validation[:, index]
            )
            for index, name in enumerate(names)
        ],
        dtype=np.float32,
    )
    common = build_common_data(bundle, labels, test_indices, base_test, thresholds)
    method_rows = []
    contrast_rows = []
    selected_rows = []
    label_budget_rows = []

    for target in names:
        target_index = names.index(target)
        y_train = labels[target][fit_train]
        y_validation = labels[target][fit_validation]
        raw_by_method: dict[str, np.ndarray] = {}
        random_specs_by_method: dict[str, dict] = {}
        fallback_by_method: dict[str, np.ndarray] = {}
        selected_by_method: dict[str, list[int] | None] = {}

        for method, representation in representations.items():
            raw, selected = selected_component_delta(
                representation["train_codes"],
                representation["test_codes"],
                representation["decoder"],
                y_train,
                args.k,
            )
            raw_by_method[method] = raw
            selected_by_method[method] = selected.tolist()
            fallback_by_method[method] = representation["decoder"][selected[0]]
            random_specs_by_method[method] = {
                "kind": "component",
                "groups": draw_random_component_groups(
                    representation["train_codes"].shape[1],
                    selected,
                    args.n_random,
                    stable_seed(
                        "component-random",
                        row.model_suffix,
                        row.cohort,
                        seed,
                        target,
                        method,
                    ),
                    args.k,
                ),
                "train_codes": representation["train_codes"],
                "test_codes": representation["test_codes"],
                "decoder": representation["decoder"],
            }

        positive = np.isfinite(y_train) & (y_train == np.nanmax(y_train))
        negative = np.isfinite(y_train) & (y_train == np.nanmin(y_train))
        cav = x_train[positive].mean(axis=0) - x_train[negative].mean(axis=0)
        cav /= max(float(np.linalg.norm(cav)), 1e-12)
        raw_by_method["supervised_cav"] = direction_delta(x_train, x_test, cav)
        random_specs_by_method["supervised_cav"] = {
            "kind": "direction",
            "directions": random_unit_directions(
                x_train.shape[1],
                args.n_random,
                stable_seed(
                    "direction-random", row.model_suffix, row.cohort, seed, target, "cav"
                ),
            ),
        }
        fallback_by_method["supervised_cav"] = cav
        selected_by_method["supervised_cav"] = None

        probe, probe_metadata = fit_sparse_probe(
            fit_root / f"sparse_probe_{target}.joblib",
            x_train,
            y_train,
            x_validation,
            y_validation,
            stable_seed("sparse-probe", row.model_suffix, row.cohort, seed, target),
        )
        if float(np.linalg.norm(probe)) <= 1e-12:
            probe = cav.copy()
            probe_metadata["zero_direction_fallback_to_cav"] = True
        atomic_npz(
            fit_root / f"supervised_directions_{target}.npz",
            supervised_cav=np.asarray(cav, dtype=np.float32),
            sparse_probe=np.asarray(probe, dtype=np.float32),
        )
        raw_by_method["sparse_probe"] = direction_delta(x_train, x_test, probe)
        random_specs_by_method["sparse_probe"] = {
            "kind": "direction",
            "directions": random_unit_directions(
                x_train.shape[1],
                args.n_random,
                stable_seed(
                    "direction-random", row.model_suffix, row.cohort, seed, target, "probe"
                ),
            ),
        }
        fallback_by_method["sparse_probe"] = probe
        selected_by_method["sparse_probe"] = None

        result_path = (
            Path(row.head_path).parent
            / "steering"
            / "cohort_adapted_atom"
            / f"seed{seed}"
            / target
            / "result.json"
        )
        result = json.loads(result_path.read_text())
        existing_raw, _ = existing_target_deltas(existing, result, 0)
        existing_records_path = result_path.with_name("records.npz")
        with np.load(existing_records_path, allow_pickle=False) as loaded:
            existing_records = {key: loaded[key] for key in loaded.files}
        if not np.array_equal(
            existing_records["patient_ids"].astype(str), common["patient_ids"].astype(str)
        ):
            raise RuntimeError(f"Existing steering record mismatch: {result_path}")
        parity = float(
            np.max(
                np.abs(
                    logit_delta(existing_raw, coefficients)
                    - np.asarray(existing_records["top5_delta"], dtype=np.float32)
                )
            )
        )
        if parity > 2e-4:
            raise RuntimeError(f"Existing SAE parity failed for {result_path}: {parity}")
        raw_by_method["sae_existing_8d"] = existing_raw
        random_specs_by_method["sae_existing_8d"] = {
            "kind": "component_existing",
            "groups": [
                np.asarray(group, dtype=int)
                for group in result["random_groups"][: args.n_random]
            ],
            "test_codes": existing["codes"],
            "centroid": existing["centroid"],
            "decoder": existing["decoder"],
        }
        fallback_by_method["sae_existing_8d"] = existing["decoder"][
            int(result["selected_atoms"]["top5"][0])
        ]
        selected_by_method["sae_existing_8d"] = list(result["selected_atoms"]["top5"])

        result_stub = {
            "target": target,
            "focus_thresholds_train": {name: 1.0 for name in names},
        }
        weights, inverse = group_bootstrap_weights(
            common["patient_ids"],
            args.bootstrap,
            np.random.default_rng(
                stable_seed("record-bootstrap", row.model_suffix, row.cohort, seed, target)
            ),
        )
        regime_specs = {
            "common64_energy": ("sae_common64", list(METHODS)),
            "existing_sae_energy": (
                "sae_existing_8d",
                ["sae_existing_8d", *METHODS],
            ),
        }
        for regime, (reference, methods) in regime_specs.items():
            reference_norm = np.linalg.norm(raw_by_method[reference], axis=1)
            points = {}
            samples = {}
            for method in methods:
                if method == reference:
                    target_logit_delta = logit_delta(
                        raw_by_method[method], coefficients
                    )
                    l2_error = 0.0
                else:
                    target_logit_delta, l2_error = norm_matched_logit_delta(
                        raw_by_method[method],
                        reference_norm,
                        fallback_by_method[method],
                        coefficients,
                    )
                data = dict(common)
                data["top5_delta"] = target_logit_delta
                random_spec = random_specs_by_method[method]
                if random_spec["kind"] == "component":
                    data["random_top5_delta"] = component_control_logit_deltas(
                        random_spec["test_codes"],
                        random_spec["decoder"],
                        random_spec["groups"],
                        reference_norm,
                        fallback_by_method[method],
                        coefficients,
                        train_codes=random_spec["train_codes"],
                    )
                elif random_spec["kind"] == "component_existing":
                    data["random_top5_delta"] = component_control_logit_deltas(
                        random_spec["test_codes"],
                        random_spec["decoder"],
                        random_spec["groups"],
                        reference_norm,
                        fallback_by_method[method],
                        coefficients,
                        centroid=random_spec["centroid"],
                    )
                else:
                    data["random_top5_delta"] = direction_control_logit_deltas(
                        x_train,
                        x_test,
                        random_spec["directions"],
                        reference_norm,
                        fallback_by_method[method],
                        coefficients,
                    )
                points[method] = method_point(data, result_stub)
                samples[method] = bootstrap_steering_metrics(
                    data, result_stub, weights, inverse
                )
                method_row = {
                    "task_index": args.task_index,
                    "model": row.model,
                    "model_suffix": row.model_suffix,
                    "cohort": row.cohort,
                    "target": target,
                    "family": result["family"],
                    "seed": seed,
                    "regime": regime,
                    "method": method,
                    "reference": reference,
                    "rank": (
                        int(existing["config"]["n_features"])
                        if method == "sae_existing_8d"
                        else 1
                        if method in {"sparse_probe", "supervised_cav"}
                        else args.rank
                    ),
                    "activation_budget_k": args.k,
                    "bootstrap_samples": args.bootstrap,
                    "n_random": args.n_random,
                    "reference_l2_rms": float(np.sqrt(np.mean(reference_norm**2))),
                    "matched_l2_max_abs_error": l2_error,
                    "existing_sae_logit_parity_max_abs": parity,
                }
                for metric in METHOD_METRICS:
                    method_row[metric] = points[method][metric]
                    stats = interval_and_p(
                        samples[method][metric],
                        -1 if metric in {"otd_mean", "wbi"} else 1,
                    )
                    method_row[f"{metric}_ci_low"] = stats["ci_low"]
                    method_row[f"{metric}_ci_high"] = stats["ci_high"]
                method_rows.append(method_row)
                selected_rows.append(
                    {
                        "task_index": args.task_index,
                        "model": row.model,
                        "model_suffix": row.model_suffix,
                        "cohort": row.cohort,
                        "target": target,
                        "seed": seed,
                        "regime": regime,
                        "method": method,
                        "selected_components": ""
                        if selected_by_method[method] is None
                        else "|".join(map(str, selected_by_method[method])),
                        "probe_nonzero_coefficients": probe_metadata["nonzero_coefficients"]
                        if method == "sparse_probe"
                        else np.nan,
                        "probe_validation_auroc": probe_metadata["validation_auroc"]
                        if method == "sparse_probe"
                        else np.nan,
                    }
                )

            for method in methods:
                if method == reference:
                    continue
                contrast = {
                    "task_index": args.task_index,
                    "model": row.model,
                    "model_suffix": row.model_suffix,
                    "cohort": row.cohort,
                    "target": target,
                    "family": result["family"],
                    "seed": seed,
                    "regime": regime,
                    "method": method,
                    "reference": reference,
                    "bootstrap_samples": args.bootstrap,
                }
                for metric in METHOD_METRICS:
                    difference = samples[method][metric] - samples[reference][metric]
                    stats = interval_and_p(
                        difference,
                        -1 if metric in {"otd_mean", "wbi"} else 1,
                    )
                    contrast[f"delta_{metric}"] = points[method][metric] - points[reference][metric]
                    contrast[f"delta_{metric}_ci_low"] = stats["ci_low"]
                    contrast[f"delta_{metric}_ci_high"] = stats["ci_high"]
                    contrast[f"delta_{metric}_p_one_sided"] = stats["p_one_sided"]
                    contrast[f"delta_{metric}_p_two_sided"] = stats["p_two_sided"]
                contrast_rows.append(contrast)

        local_indices = np.arange(len(fit_train), dtype=int)
        local_identifiers = record_ids[fit_train]
        for label_budget in LABEL_BUDGETS:
            budget_indices = stratified_subset(
                local_indices,
                local_identifiers,
                {target: y_train},
                label_budget,
                "label-budget",
                row.model_suffix,
                row.cohort,
                seed,
                target,
            )
            budget_labels = y_train[budget_indices]
            budget_raw: dict[str, np.ndarray] = {}
            budget_fallback: dict[str, np.ndarray] = {}
            budget_selected: dict[str, list[int] | None] = {}
            for method, representation in representations.items():
                raw, selected = selected_component_delta(
                    representation["train_codes"][budget_indices],
                    representation["test_codes"],
                    representation["decoder"],
                    budget_labels,
                    args.k,
                )
                budget_raw[method] = raw
                budget_fallback[method] = representation["decoder"][selected[0]]
                budget_selected[method] = selected.tolist()
            budget_positive = np.isfinite(budget_labels) & (
                budget_labels == np.nanmax(budget_labels)
            )
            budget_negative = np.isfinite(budget_labels) & (
                budget_labels == np.nanmin(budget_labels)
            )
            budget_cav = (
                x_train[budget_indices][budget_positive].mean(axis=0)
                - x_train[budget_indices][budget_negative].mean(axis=0)
            )
            budget_cav /= max(float(np.linalg.norm(budget_cav)), 1e-12)
            budget_raw["supervised_cav"] = direction_delta(
                x_train[budget_indices], x_test, budget_cav
            )
            budget_fallback["supervised_cav"] = budget_cav
            budget_selected["supervised_cav"] = None
            budget_probe, budget_probe_metadata = fit_sparse_probe(
                fit_root / f"sparse_probe_{target}_n{label_budget}.joblib",
                x_train[budget_indices],
                budget_labels,
                x_validation,
                y_validation,
                stable_seed(
                    "sparse-probe-budget",
                    row.model_suffix,
                    row.cohort,
                    seed,
                    target,
                    label_budget,
                ),
            )
            if float(np.linalg.norm(budget_probe)) <= 1e-12:
                budget_probe = budget_cav.copy()
                budget_probe_metadata["zero_direction_fallback_to_cav"] = True
            budget_raw["sparse_probe"] = direction_delta(
                x_train[budget_indices], x_test, budget_probe
            )
            budget_fallback["sparse_probe"] = budget_probe
            budget_selected["sparse_probe"] = None
            reference_norm = np.linalg.norm(budget_raw["sae_common64"], axis=1)
            for method in METHODS:
                matched = (
                    budget_raw[method]
                    if method == "sae_common64"
                    else norm_match(
                        budget_raw[method], reference_norm, budget_fallback[method]
                    )
                )
                data = dict(common)
                data["top5_delta"] = logit_delta(matched, coefficients)
                # Label-budget outputs contain only raw method endpoints. A
                # one-group zero placeholder avoids recomputing unused random
                # excess metrics for every budget without changing any saved
                # STE/OTD/selectivity/WBI/behavior value.
                data["random_top5_delta"] = np.zeros(
                    (len(x_test), 1, len(names)), dtype=np.float32
                )
                point = method_point(data, result_stub)
                label_budget_rows.append(
                    {
                        "task_index": args.task_index,
                        "model": row.model,
                        "model_suffix": row.model_suffix,
                        "cohort": row.cohort,
                        "target": target,
                        "family": result["family"],
                        "seed": seed,
                        "label_budget_requested": label_budget,
                        "label_budget_actual": len(budget_indices),
                        "positive_labels": int(budget_positive.sum()),
                        "negative_labels": int(budget_negative.sum()),
                        "method": method,
                        "reference": "sae_common64",
                        "selected_components": ""
                        if budget_selected[method] is None
                        else "|".join(map(str, budget_selected[method])),
                        "probe_nonzero_coefficients": budget_probe_metadata[
                            "nonzero_coefficients"
                        ]
                        if method == "sparse_probe"
                        else np.nan,
                        "reference_l2_rms": float(
                            np.sqrt(np.mean(reference_norm**2))
                        ),
                        "matched_l2_max_abs_error": float(
                            np.max(
                                np.abs(
                                    np.linalg.norm(matched, axis=1) - reference_norm
                                )
                            )
                        ),
                        **point,
                    }
                )
        print(
            f"evaluated task={args.task_index} target={target} methods={len(method_rows)}",
            flush=True,
        )

    atomic_csv(pd.DataFrame(reconstruction_rows), worker / "reconstruction_metrics.csv")
    atomic_csv(pd.DataFrame(rate_distortion_rows), worker / "rate_distortion.csv")
    atomic_csv(pd.DataFrame(method_rows), worker / "method_seed_cells.csv")
    atomic_csv(pd.DataFrame(contrast_rows), worker / "paired_method_contrasts.csv")
    atomic_csv(pd.DataFrame(selected_rows), worker / "selected_directions.csv")
    atomic_csv(pd.DataFrame(label_budget_rows), worker / "label_budget_seed_cells.csv")
    expected_method_rows = len(names) * (len(METHODS) + len(METHODS) + 1)
    expected_contrast_rows = len(names) * ((len(METHODS) - 1) + len(METHODS))
    expected_label_budget_rows = len(names) * len(LABEL_BUDGETS) * len(METHODS)
    if (
        len(method_rows) != expected_method_rows
        or len(contrast_rows) != expected_contrast_rows
        or len(label_budget_rows) != expected_label_budget_rows
    ):
        raise RuntimeError(
            f"Output row audit failed: methods={len(method_rows)}/{expected_method_rows}, "
            f"contrasts={len(contrast_rows)}/{expected_contrast_rows}, "
            f"label_budget={len(label_budget_rows)}/{expected_label_budget_rows}"
        )
    metadata = {
        "schema_version": 1,
        **preflight,
        "status": "complete",
        "train_records_available": len(train_all),
        "train_records_fit": len(fit_train),
        "validation_records_available": len(validation_all),
        "validation_records_fit": len(fit_validation),
        "test_records_evaluated": len(test_indices),
        "method_rows": len(method_rows),
        "contrast_rows": len(contrast_rows),
        "reconstruction_rows": len(reconstruction_rows),
        "rate_distortion_rows": len(rate_distortion_rows),
        "label_budget_rows": len(label_budget_rows),
        "fit_diagnostics": fit_diagnostics,
        "waveforms_written": False,
        "record_level_activations_written": False,
        "data_files_modified": False,
    }
    write_json(worker / "fit_diagnostics.json", fit_diagnostics)
    write_json(complete, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
