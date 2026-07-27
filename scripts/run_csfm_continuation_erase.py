#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.adapters.csfm import CSFM_DEPTH, try_load_model
from benchmark_v1.config import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSFM continuation-based single-concept erasure.")
    parser.add_argument("--activation-index-dir", required=True, type=Path)
    parser.add_argument("--probe-features-dir", required=True, type=Path)
    parser.add_argument("--concepts-matrix", default=ROOT / "results" / "manifest" / "concepts_matrix.csv", type=Path)
    parser.add_argument("--tasks-matrix", default=ROOT / "results" / "manifest" / "tasks_matrix.csv", type=Path)
    parser.add_argument("--concept-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--concept-alpha", type=float, default=10.0)
    parser.add_argument("--task-alpha", type=float, default=1.0)
    parser.add_argument("--eraser-method", choices=["leace", "euclidean"], default="leace")
    parser.add_argument(
        "--leace-ridge",
        type=float,
        default=1e-4,
        help="Relative ridge added to Sigma_hh: ridge * mean(diag(Sigma_hh)).",
    )
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--max-test-shards", type=int, default=0, help="0 means all test shards.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_feature_path(probe_features_dir: Path, feature_name: str) -> Path:
    for row in read_csv(probe_features_dir / "features.csv"):
        if row["feature"] == feature_name:
            return ROOT / row["file"]
    raise ValueError(f"feature {feature_name!r} not found in {probe_features_dir / 'features.csv'}")


def matrix_column(records: list[dict[str, str]], matrix_path: Path, column: str):
    import numpy as np

    rows_by_id = {row["ecg_id"]: row for row in read_csv(matrix_path)}
    values = np.empty(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        values[i] = parse_float(rows_by_id[record["ecg_id"]][column])
    return values


def robust_scale_train(y_train):
    import numpy as np

    med = np.nanmedian(y_train)
    q25, q75 = np.nanpercentile(y_train, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanstd(y_train)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return float(med), float(scale)


def raw_direction_from_probe(x_train_raw, y_train, alpha: float):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    valid = np.isfinite(y_train)
    if valid.sum() < 100:
        raise ValueError("not enough finite concept targets in train split")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(x_train_raw[valid], dtype=np.float32))
    med, scale = robust_scale_train(y_train[valid])
    y_scaled = (y_train[valid] - med) / scale
    probe = Ridge(alpha=alpha)
    probe.fit(x_train, y_scaled)
    coef_std = np.asarray(probe.coef_, dtype=np.float32).reshape(-1)
    raw = coef_std / np.asarray(scaler.scale_, dtype=np.float32)
    norm = np.linalg.norm(raw)
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("concept direction has zero norm")
    return raw / norm


def fit_leace_eraser(x_train_raw, y_train, ridge: float, rng=None):
    import numpy as np

    valid = np.isfinite(y_train)
    if valid.sum() < 100:
        raise ValueError("not enough finite concept targets in train split")
    x = np.asarray(x_train_raw[valid], dtype=np.float64)
    med, scale = robust_scale_train(y_train[valid])
    y = (y_train[valid] - med) / scale
    y = y - np.mean(y)
    mu = x.mean(axis=0)
    centered = x - mu
    n, d = centered.shape
    cov = (centered.T @ centered) / max(n, 1)
    ridge_abs = float(ridge) * float(np.trace(cov) / max(d, 1))
    if not np.isfinite(ridge_abs) or ridge_abs <= 0:
        ridge_abs = float(ridge)
    cov_ridged = cov + ridge_abs * np.eye(d, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov_ridged)
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_cov = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    inv_sqrt_cov = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    if rng is None:
        whitened = centered @ inv_sqrt_cov
        direction = (whitened.T @ y) / max(n, 1)
    else:
        direction = rng.normal(size=d)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("LEACE direction has zero norm")
    u = direction / norm
    remove_matrix = inv_sqrt_cov @ np.outer(u, u) @ sqrt_cov
    return {
        "method": "leace",
        "mu": mu.astype(np.float32),
        "remove_matrix": remove_matrix.astype(np.float32),
        "ridge_abs": ridge_abs,
        "rank": 1,
    }


def fit_euclidean_eraser(x_train_raw, y_train, alpha: float, rng=None):
    import numpy as np

    if rng is None:
        direction = raw_direction_from_probe(x_train_raw, y_train, alpha=alpha)
    else:
        direction = rng.normal(size=x_train_raw.shape[1]).astype(np.float32)
        direction = direction / np.linalg.norm(direction)
    return {"method": "euclidean", "direction": np.asarray(direction, dtype=np.float32), "rank": 1}


def fit_eraser(x_train_raw, y_train, method: str, alpha: float, leace_ridge: float, rng=None):
    if method == "leace":
        return fit_leace_eraser(x_train_raw, y_train, ridge=leace_ridge, rng=rng)
    if method == "euclidean":
        return fit_euclidean_eraser(x_train_raw, y_train, alpha=alpha, rng=rng)
    raise ValueError(method)


def r2_score_np(y_true, y_pred) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def erase_features(x, eraser):
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if eraser["method"] == "euclidean":
        direction = np.asarray(eraser["direction"], dtype=np.float32)
        return x - (x @ direction)[:, None] * direction[None, :]
    if eraser["method"] == "leace":
        mu = np.asarray(eraser["mu"], dtype=np.float32)
        remove_matrix = np.asarray(eraser["remove_matrix"], dtype=np.float32)
        return x - ((x - mu) @ remove_matrix)
    raise ValueError(eraser["method"])


def residual_probe_audit(layer_feature, splits, concept_y, eraser, alpha: float):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    valid_train = train_idx[np.isfinite(concept_y[train_idx])]
    valid_test = test_idx[np.isfinite(concept_y[test_idx])]
    if len(valid_train) < 100 or len(valid_test) < 20:
        return {
            "original_probe_r2": None,
            "residual_probe_r2": None,
            "residual_probe_r2_drop": None,
            "residual_probe_threshold": None,
            "eraser_effective_flag": None,
        }

    x_train_raw = np.asarray(layer_feature[valid_train], dtype=np.float32)
    x_test_raw = np.asarray(layer_feature[valid_test], dtype=np.float32)
    med, scale = robust_scale_train(concept_y[valid_train])
    y_train = (concept_y[valid_train] - med) / scale
    y_test = (concept_y[valid_test] - med) / scale

    original_scaler = StandardScaler()
    x_train = original_scaler.fit_transform(x_train_raw)
    x_test = original_scaler.transform(x_test_raw)
    original_probe = Ridge(alpha=alpha)
    original_probe.fit(x_train, y_train)
    original_r2 = r2_score_np(y_test, original_probe.predict(x_test))

    erased_train_raw = erase_features(x_train_raw, eraser)
    erased_test_raw = erase_features(x_test_raw, eraser)
    residual_scaler = StandardScaler()
    erased_train = residual_scaler.fit_transform(erased_train_raw)
    erased_test = residual_scaler.transform(erased_test_raw)
    residual_probe = Ridge(alpha=alpha)
    residual_probe.fit(erased_train, y_train)
    residual_r2 = r2_score_np(y_test, residual_probe.predict(erased_test))
    threshold = max(0.02, 0.35 * max(original_r2, 0.04))
    return {
        "original_probe_r2": float(original_r2),
        "residual_probe_r2": float(residual_r2),
        "residual_probe_r2_drop": float(original_r2 - residual_r2),
        "residual_probe_threshold": float(threshold),
        "eraser_effective_flag": bool(residual_r2 < threshold),
    }


def erase_tokens(tokens, eraser):
    import numpy as np

    tokens = np.asarray(tokens, dtype=np.float32)
    if eraser["method"] == "euclidean":
        direction = np.asarray(eraser["direction"], dtype=np.float32)
        return tokens - (tokens @ direction)[:, :, None] * direction[None, None, :]
    if eraser["method"] == "leace":
        mu = np.asarray(eraser["mu"], dtype=np.float32)
        remove_matrix = np.asarray(eraser["remove_matrix"], dtype=np.float32)
        return tokens - ((tokens - mu) @ remove_matrix)
    raise ValueError(eraser["method"])


def bootstrap_adjusted_delta(y, erased, random, n_samples: int, seed: int):
    import numpy as np
    from sklearn.metrics import roc_auc_score

    if n_samples <= 0:
        return {
            "bootstrap_samples": 0,
            "bootstrap_valid_samples": 0,
            "delta_auroc_minus_random_ci_low": None,
            "delta_auroc_minus_random_ci_high": None,
            "delta_auroc_minus_random_p_one_sided": None,
        }
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int32)
    erased = np.asarray(erased, dtype=np.float64)
    random = np.asarray(random, dtype=np.float64)
    n = len(y)
    vals = []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(set(yb.tolist())) < 2:
            continue
        vals.append(float(roc_auc_score(yb, random[idx]) - roc_auc_score(yb, erased[idx])))
    if not vals:
        return {
            "bootstrap_samples": n_samples,
            "bootstrap_valid_samples": 0,
            "delta_auroc_minus_random_ci_low": None,
            "delta_auroc_minus_random_ci_high": None,
            "delta_auroc_minus_random_p_one_sided": None,
        }
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "bootstrap_samples": n_samples,
        "bootstrap_valid_samples": int(len(arr)),
        "delta_auroc_minus_random_ci_low": float(np.percentile(arr, 2.5)),
        "delta_auroc_minus_random_ci_high": float(np.percentile(arr, 97.5)),
        "delta_auroc_minus_random_p_one_sided": float((np.sum(arr <= 0.0) + 1.0) / (len(arr) + 1.0)),
    }


def continue_from_post_block(model, x, layer_idx: int, mask=None):
    for attn, ff in model.transformer.layers[layer_idx + 1 :]:
        x = attn(x, mask=mask) + x
        x = ff(x) + x
    x = model.transformer.norm(x)
    x = x.mean(dim=1) if model.pool == "mean" else x[:, 0]
    x = model.to_latent(x)
    return model.mlp_head(x)


def main() -> None:
    args = parse_args()
    if args.layer < 0 or args.layer >= CSFM_DEPTH:
        raise ValueError(f"layer must be in 0..{CSFM_DEPTH - 1}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import torch
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    records = read_csv(args.probe_features_dir / "records.csv")
    splits = np.array([row["split"] for row in records])
    train_idx = np.where(splits == "train")[0]
    feature_name = f"layer_{args.layer:02d}_mean"
    layer_feature = np.load(load_feature_path(args.probe_features_dir, feature_name), mmap_mode="r")
    pooled_feature = np.load(load_feature_path(args.probe_features_dir, "pooled"), mmap_mode="r")
    concept_y = matrix_column(records, args.concepts_matrix, args.concept_id)
    task_y = matrix_column(records, args.tasks_matrix, args.task_id)

    eraser = fit_eraser(
        layer_feature[train_idx],
        concept_y[train_idx],
        method=args.eraser_method,
        alpha=args.concept_alpha,
        leace_ridge=args.leace_ridge,
    )
    residual_probe = residual_probe_audit(
        layer_feature,
        splits,
        concept_y,
        eraser,
        alpha=args.concept_alpha,
    )
    random_eraser = fit_eraser(
        layer_feature[train_idx],
        concept_y[train_idx],
        method=args.eraser_method,
        alpha=args.concept_alpha,
        leace_ridge=args.leace_ridge,
        rng=rng,
    )

    valid_train = np.isfinite(task_y[train_idx])
    train_classes = set(task_y[train_idx][valid_train].astype(int).tolist())
    if len(train_classes) < 2:
        raise ValueError(f"task {args.task_id} has fewer than 2 classes in train split")
    task_scaler = StandardScaler()
    x_train = task_scaler.fit_transform(np.asarray(pooled_feature[train_idx][valid_train], dtype=np.float32))
    head = RidgeClassifier(alpha=args.task_alpha)
    head.fit(x_train, task_y[train_idx][valid_train].astype(int))

    model, status = try_load_model(device=args.device)
    if model is None:
        raise RuntimeError(status)
    model.eval()

    shard_rows = [
        row for row in read_csv(args.activation_index_dir / "shards.csv")
        if row["split"] == "test"
    ]
    if args.max_test_shards > 0:
        shard_rows = shard_rows[: args.max_test_shards]

    y_true: list[float] = []
    base_scores: list[float] = []
    erased_scores: list[float] = []
    random_scores: list[float] = []
    max_abs_reconstruction_diff = 0.0

    with torch.no_grad():
        for shard in shard_rows:
            shard_dir = ROOT / Path(shard["activation_metadata"]).parent
            record_ids = [row["ecg_id"] for row in read_csv(shard_dir / "record_ids.csv")]
            labels = matrix_column([{"ecg_id": ecg_id} for ecg_id in record_ids], args.tasks_matrix, args.task_id)
            valid = np.isfinite(labels)
            if not valid.any():
                continue
            tokens = np.load(shard_dir / f"layer_{args.layer:02d}.npy", mmap_mode="r")
            tokens_np = np.asarray(tokens, dtype=np.float32)
            base_tokens = torch.as_tensor(tokens_np, dtype=torch.float32, device=args.device)
            erased_tokens = torch.as_tensor(erase_tokens(tokens_np, eraser), dtype=torch.float32, device=args.device)
            random_tokens = torch.as_tensor(erase_tokens(tokens_np, random_eraser), dtype=torch.float32, device=args.device)

            base_pooled = continue_from_post_block(model, base_tokens, args.layer).detach().cpu().numpy().astype(np.float32)
            erased_pooled = continue_from_post_block(model, erased_tokens, args.layer).detach().cpu().numpy().astype(np.float32)
            random_pooled = continue_from_post_block(model, random_tokens, args.layer).detach().cpu().numpy().astype(np.float32)
            saved_pooled = np.load(shard_dir / "pooled.npy", mmap_mode="r")
            max_abs_reconstruction_diff = max(
                max_abs_reconstruction_diff,
                float(np.max(np.abs(base_pooled - np.asarray(saved_pooled, dtype=np.float32)))),
            )

            base_score = head.decision_function(task_scaler.transform(base_pooled[valid]))
            erased_score = head.decision_function(task_scaler.transform(erased_pooled[valid]))
            random_score = head.decision_function(task_scaler.transform(random_pooled[valid]))
            y_true.extend(labels[valid].astype(int).tolist())
            base_scores.extend(np.asarray(base_score).reshape(-1).tolist())
            erased_scores.extend(np.asarray(erased_score).reshape(-1).tolist())
            random_scores.extend(np.asarray(random_score).reshape(-1).tolist())

    y = np.asarray(y_true, dtype=np.int32)
    if len(set(y.tolist())) < 2:
        raise ValueError(f"task {args.task_id} has fewer than 2 classes in evaluated test records")
    base = np.asarray(base_scores)
    erased = np.asarray(erased_scores)
    random = np.asarray(random_scores)
    base_auroc = roc_auc_score(y, base)
    erased_auroc = roc_auc_score(y, erased)
    random_auroc = roc_auc_score(y, random)
    base_auprc = average_precision_score(y, base)
    erased_auprc = average_precision_score(y, erased)
    random_auprc = average_precision_score(y, random)
    bootstrap = bootstrap_adjusted_delta(
        y,
        erased,
        random,
        n_samples=args.bootstrap_samples,
        seed=args.seed + args.layer * 1009,
    )

    report = {
        "model_status": status,
        "concept_id": args.concept_id,
        "task_id": args.task_id,
        "layer": args.layer,
        "feature_name": feature_name,
        "n_test": int(len(y)),
        "positive_test": int(y.sum()),
        "shards_evaluated": len(shard_rows),
        "base_auroc": float(base_auroc),
        "erased_auroc": float(erased_auroc),
        "random_auroc": float(random_auroc),
        "delta_auroc": float(base_auroc - erased_auroc),
        "delta_auroc_minus_random": float((base_auroc - erased_auroc) - (base_auroc - random_auroc)),
        "base_auprc": float(base_auprc),
        "erased_auprc": float(erased_auprc),
        "random_auprc": float(random_auprc),
        "delta_auprc": float(base_auprc - erased_auprc),
        "max_abs_reconstruction_diff": max_abs_reconstruction_diff,
        "concept_alpha": args.concept_alpha,
        "task_alpha": args.task_alpha,
        "eraser_method": args.eraser_method,
        "eraser_rank": int(eraser.get("rank", 1)),
        "leace_ridge": args.leace_ridge if args.eraser_method == "leace" else None,
        "leace_ridge_abs": eraser.get("ridge_abs"),
        "seed": args.seed,
        **residual_probe,
        **bootstrap,
    }
    suffix = f"_smoke{args.max_test_shards}" if args.max_test_shards > 0 else ""
    out_path = args.out_dir / f"continuation_erase_{args.concept_id}_to_{args.task_id}_layer{args.layer:02d}{suffix}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
