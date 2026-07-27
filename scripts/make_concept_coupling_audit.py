#!/usr/bin/env python3
"""Audit concept co-erasure among LEACE-confirmed concepts.

For each confirmed (model, concept, task, layer) LEACE erasure cell, this script
reconstructs the LEACE eraser from train features and measures how much other
confirmed concepts remain linearly readable from the same erased layer features.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results" / "manifest"
PROBE_FEATURES = ROOT / "results" / "probe_features"
CLEANUP = ROOT / "results" / "analysis" / "model_comparison" / "cleanup_audit"

LEACE_RIDGE = 1e-4
PROBE_ALPHA = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit cross-concept damage after LEACE erasure.")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
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
    values = np.empty(len(records), dtype=np.float32)
    rows_by_id = {row["ecg_id"]: row for row in read_csv(matrix_path)}
    for i, record in enumerate(records):
        values[i] = parse_float(rows_by_id[record["ecg_id"]][column])
    return values


def robust_scale_train(y_train):
    med = np.nanmedian(y_train)
    q25, q75 = np.nanpercentile(y_train, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanstd(y_train)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return float(med), float(scale)


def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def fit_leace_eraser(x_train_raw, y_train, ridge: float = LEACE_RIDGE):
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
    whitened = centered @ inv_sqrt_cov
    direction = (whitened.T @ y) / max(n, 1)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("LEACE direction has zero norm")
    u = direction / norm
    remove_matrix = inv_sqrt_cov @ np.outer(u, u) @ sqrt_cov
    return {
        "mu": mu.astype(np.float32),
        "remove_matrix": remove_matrix.astype(np.float32),
        "ridge_abs": ridge_abs,
    }


def erase_features(x, eraser):
    x = np.asarray(x, dtype=np.float32)
    mu = np.asarray(eraser["mu"], dtype=np.float32)
    remove_matrix = np.asarray(eraser["remove_matrix"], dtype=np.float32)
    return x - ((x - mu) @ remove_matrix)


def residual_probe_for_target(layer_feature, splits, target_y, eraser):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    valid_train = train_idx[np.isfinite(target_y[train_idx])]
    valid_test = test_idx[np.isfinite(target_y[test_idx])]
    if len(valid_train) < 100 or len(valid_test) < 20:
        return None

    x_train_raw = np.asarray(layer_feature[valid_train], dtype=np.float32)
    x_test_raw = np.asarray(layer_feature[valid_test], dtype=np.float32)
    med, scale = robust_scale_train(target_y[valid_train])
    y_train = (target_y[valid_train] - med) / scale
    y_test = (target_y[valid_test] - med) / scale

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_test = scaler.transform(x_test_raw)
    probe = Ridge(alpha=PROBE_ALPHA)
    probe.fit(x_train, y_train)
    original_r2 = r2_score_np(y_test, probe.predict(x_test))

    erased_train_raw = erase_features(x_train_raw, eraser)
    erased_test_raw = erase_features(x_test_raw, eraser)
    erased_scaler = StandardScaler()
    erased_train = erased_scaler.fit_transform(erased_train_raw)
    erased_test = erased_scaler.transform(erased_test_raw)
    residual_probe = Ridge(alpha=PROBE_ALPHA)
    residual_probe.fit(erased_train, y_train)
    residual_r2 = r2_score_np(y_test, residual_probe.predict(erased_test))
    threshold = max(0.02, 0.35 * max(original_r2, 0.04))
    return {
        "target_original_r2": float(original_r2),
        "target_residual_r2": float(residual_r2),
        "target_r2_drop": float(original_r2 - residual_r2),
        "target_residual_threshold": float(threshold),
        "target_erased_effective": bool(residual_r2 < threshold),
    }


def confirmed_concepts_by_model(confirmed: pd.DataFrame) -> dict[str, list[str]]:
    out = {}
    for model, part in confirmed.groupby("model", sort=False):
        out[model] = sorted(part["concept_id"].unique())
    return out


def process_source_record(args_tuple):
    source, target_concepts, family_by_concept, corr_map = args_tuple
    probe_dir = PROBE_FEATURES / source["suffix"]
    records = read_csv(probe_dir / "records.csv")
    splits = np.array([row["split"] for row in records])
    train_idx = np.where(splits == "train")[0]
    layer_feature = np.load(load_feature_path(probe_dir, source["feature"]), mmap_mode="r")
    source_y = matrix_column(records, MANIFEST / "concepts_matrix.csv", source["concept_id"])
    eraser = fit_leace_eraser(layer_feature[train_idx], source_y[train_idx], ridge=LEACE_RIDGE)
    rows = []
    for target_concept in target_concepts:
        target_y = matrix_column(records, MANIFEST / "concepts_matrix.csv", target_concept)
        result = residual_probe_for_target(layer_feature, splits, target_y, eraser)
        if result is None:
            continue
        corr_val = corr_map.get((source["concept_id"], target_concept), np.nan)
        rows.append(
            {
                "model": source["model"],
                "suffix": source["suffix"],
                "source_concept": source["concept_id"],
                "source_family": source["family"],
                "source_task": source["task_id"],
                "source_canonical_task": source["canonical_task_id"],
                "source_layer": int(source["layer"]),
                "source_delta_auroc_minus_random": source["delta_auroc_minus_random"],
                "target_concept": target_concept,
                "target_family": family_by_concept.get(target_concept),
                "same_concept": source["concept_id"] == target_concept,
                "same_family": source["family"] == family_by_concept.get(target_concept),
                "source_target_spearman_r": corr_val,
                "source_target_abs_spearman_r": abs(corr_val) if np.isfinite(corr_val) else np.nan,
                "leace_ridge_abs": eraser["ridge_abs"],
                **result,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    concepts = pd.read_csv(ROOT / "configs" / "concepts.csv")
    family_by_concept = concepts.set_index("concept_id")["family"].to_dict()
    concept_matrix = pd.read_csv(MANIFEST / "concepts_matrix.csv")
    concept_values = concept_matrix[[c for c in concept_matrix.columns if c != "ecg_id"]].apply(
        pd.to_numeric, errors="coerce"
    )
    corr = concept_values.corr(method="spearman", min_periods=100)
    corr_map = {}
    for a in corr.index:
        for b in corr.columns:
            corr_map[(a, b)] = corr.loc[a, b]

    canonical = pd.read_csv(CLEANUP / "continuation_canonical_strict_fdr.csv")
    confirmed = canonical[canonical["canonical_confirmed"]].copy()
    confirmed = confirmed[confirmed["eraser_method"] == "leace"].copy()
    targets_by_model = confirmed_concepts_by_model(confirmed)

    jobs = [
        (source, targets_by_model[source["model"]], family_by_concept, corr_map)
        for source in confirmed.to_dict("records")
    ]
    rows = []
    if args.workers <= 1:
        for job in jobs:
            rows.extend(process_source_record(job))
    else:
        with Pool(processes=args.workers) as pool:
            for part in pool.imap_unordered(process_source_record, jobs):
                rows.extend(part)

    coupling = pd.DataFrame(rows)
    coupling.to_csv(CLEANUP / "concept_coupling_residual_matrix.csv", index=False)

    other = coupling[~coupling["same_concept"]].copy()
    summary = (
        other.groupby(["model", "source_concept", "source_family", "source_task", "source_layer"], sort=False)
        .agg(
            n_other_confirmed_concepts=("target_concept", "nunique"),
            n_other_erased_effective=("target_erased_effective", "sum"),
            max_other_r2_drop=("target_r2_drop", "max"),
            median_other_r2_drop=("target_r2_drop", "median"),
            max_abs_groundtruth_corr=("source_target_abs_spearman_r", "max"),
        )
        .reset_index()
    )
    summary["other_erased_fraction"] = (
        summary["n_other_erased_effective"] / summary["n_other_confirmed_concepts"].replace(0, np.nan)
    )
    summary.to_csv(CLEANUP / "concept_coupling_summary.csv", index=False)

    high = other.sort_values(["target_erased_effective", "target_r2_drop"], ascending=[False, False]).head(80)
    report = [
        "# Concept Coupling Audit",
        "",
        "Rows in `concept_coupling_residual_matrix.csv` measure how readable each target confirmed concept remains after applying a source concept's LEACE eraser at the source layer.",
        "",
        "## Source Summary",
        "",
        summary.sort_values(["other_erased_fraction", "max_other_r2_drop"], ascending=[False, False])
        .head(40)
        .fillna("")
        .to_markdown(index=False),
        "",
        "## Strongest Cross-Concept Damage",
        "",
        high[
            [
                "model",
                "source_concept",
                "target_concept",
                "source_family",
                "target_family",
                "source_target_spearman_r",
                "target_original_r2",
                "target_residual_r2",
                "target_r2_drop",
                "target_erased_effective",
            ]
        ]
        .fillna("")
        .to_markdown(index=False),
        "",
    ]
    (CLEANUP / "concept_coupling_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote concept coupling audit to {CLEANUP}")


if __name__ == "__main__":
    main()
