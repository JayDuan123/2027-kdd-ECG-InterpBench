#!/usr/bin/env python
"""Export real LEACE/CAV artifacts needed by the SAE extension.

This script does not run model inference or train an SAE. It recomputes, from
existing pooled probe features, the per-cell artifacts that were not persisted
by the LEACE benchmark run:

- LEACE removed subspace directions
- dense-probe CAV directions
- train-split activation normalisation stats

The SAE operates in per-dimension-normalised activation space, so this exporter
stores both raw-space and SAE-normalised directions. The SAE extension should use
`leace_u_sae_norm.npy` for A_geo and `cav_sae_norm.npy` for feature ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.config import ROOT


PILOT_KEYS = {
    ("CSFM", "st_amp_global", "mi_ischemia"),
    ("CSFM", "qrs_duration", "ptbxl_cd"),
    ("CSFM", "qrst_angle", "ptbxl_cd"),
    ("CSFM", "p_found", "af_rhythm"),
    ("HuBERT-ECG", "qrst_angle", "mi_ischemia"),
    ("HuBERT-ECG", "q_amp_precordial", "ptbxl_mi"),
    ("HuBERT-ECG", "hr_atrial", "af_rhythm"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CAV and LEACE subspace artifacts for SAE extension.")
    parser.add_argument(
        "--continuation-csv",
        type=Path,
        default=ROOT / "results" / "analysis" / "model_comparison" / "cleanup_audit" / "continuation_canonical_strict_fdr.csv",
    )
    parser.add_argument(
        "--concepts-matrix",
        type=Path,
        default=ROOT / "results" / "manifest" / "concepts_matrix.csv",
    )
    parser.add_argument(
        "--probe-features-root",
        type=Path,
        default=ROOT / "results" / "probe_features",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "sae_artifacts",
    )
    parser.add_argument("--concept-alpha", type=float, default=10.0)
    parser.add_argument("--leace-ridge", type=float, default=1e-4)
    parser.add_argument("--confirmed-only", action="store_true", default=True)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def robust_scale_train(y_train: np.ndarray) -> tuple[float, float]:
    med = np.nanmedian(y_train)
    q25, q75 = np.nanpercentile(y_train, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanstd(y_train)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return float(med), float(scale)


def load_feature_path(probe_features_dir: Path, feature_name: str) -> Path:
    for row in read_csv(probe_features_dir / "features.csv"):
        if row["feature"] == feature_name:
            return ROOT / row["file"]
    raise ValueError(f"feature {feature_name!r} not found in {probe_features_dir / 'features.csv'}")


def matrix_column(records: list[dict[str, str]], matrix_path: Path, column: str) -> np.ndarray:
    rows_by_id = {row["ecg_id"]: row for row in read_csv(matrix_path)}
    values = np.empty(len(records), dtype=np.float32)
    for i, record in enumerate(records):
        values[i] = parse_float(rows_by_id[record["ecg_id"]][column])
    return values


def fit_cav_directions(x_train_raw: np.ndarray, y_train_raw: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    valid = np.isfinite(y_train_raw)
    if int(valid.sum()) < 100:
        raise ValueError("not enough finite concept targets for CAV")
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(x_train_raw[valid], dtype=np.float32))
    med, scale = robust_scale_train(y_train_raw[valid])
    y_train = (y_train_raw[valid] - med) / scale
    probe = Ridge(alpha=alpha)
    probe.fit(x_train, y_train)
    cav_standardized = np.asarray(probe.coef_, dtype=np.float64).reshape(-1)
    std_norm = np.linalg.norm(cav_standardized)
    if not np.isfinite(std_norm) or std_norm <= 1e-12:
        raise ValueError("zero CAV standardized direction")
    cav_standardized = cav_standardized / std_norm

    raw_covector = np.asarray(probe.coef_, dtype=np.float64).reshape(-1) / np.asarray(scaler.scale_, dtype=np.float64)
    raw_norm = np.linalg.norm(raw_covector)
    if not np.isfinite(raw_norm) or raw_norm <= 1e-12:
        raise ValueError("zero CAV raw direction")
    raw_covector = raw_covector / raw_norm
    return {
        "cav_standardized": cav_standardized.astype(np.float32),
        "cav_raw_covector": raw_covector.astype(np.float32),
        "probe_scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "probe_scaler_scale": np.asarray(scaler.scale_, dtype=np.float32),
    }


def fit_leace_subspace(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    sae_sigma: np.ndarray,
    ridge: float,
) -> dict[str, object]:
    valid = np.isfinite(y_train_raw)
    if int(valid.sum()) < 100:
        raise ValueError("not enough finite concept targets for LEACE")
    x = np.asarray(x_train_raw[valid], dtype=np.float64)
    med, scale = robust_scale_train(y_train_raw[valid])
    y = (y_train_raw[valid] - med) / scale
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
        raise ValueError("zero LEACE whitened direction")
    u_whitened = direction / norm

    # Row-vector erasure subtracts scalar * (u^T Sigma^1/2). This is the raw
    # output subspace removed from activations.
    u_raw = sqrt_cov @ u_whitened
    u_raw = u_raw / max(float(np.linalg.norm(u_raw)), 1e-12)

    u_sae = u_raw / np.asarray(sae_sigma, dtype=np.float64).clip(min=1e-6)
    u_sae = u_sae / max(float(np.linalg.norm(u_sae)), 1e-12)

    return {
        "leace_u_whitened": u_whitened.astype(np.float32)[:, None],
        "leace_u_raw": u_raw.astype(np.float32)[:, None],
        "leace_u_sae_norm": u_sae.astype(np.float32)[:, None],
        "leace_mu": mu.astype(np.float32),
        "leace_ridge_abs": float(ridge_abs),
        "rank": 1,
    }


def selected_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.continuation_csv)
    if args.confirmed_only:
        rows = [row for row in rows if truthy(row.get("canonical_confirmed"))]
    if args.pilot_only:
        rows = [
            row
            for row in rows
            if (row["model"], row["concept_id"], row["canonical_task_id"]) in PILOT_KEYS
            or (row["model"], row["concept_id"], row["task_id"]) in PILOT_KEYS
        ]
    if args.limit:
        rows = rows[: args.limit]
    return rows


def export_one(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    suffix = row["suffix"]
    feature = row["feature"]
    concept = row["concept_id"]
    task = row["canonical_task_id"] or row["task_id"]
    layer = int(float(row["layer"]))
    probe_features_dir = args.probe_features_root / suffix
    records = read_csv(probe_features_dir / "records.csv")
    splits = np.array([record["split"] for record in records])
    train_idx = np.where(splits == "train")[0]
    feature_path = load_feature_path(probe_features_dir, feature)
    x = np.load(feature_path, mmap_mode="r")
    concept_y = matrix_column(records, args.concepts_matrix, concept)
    x_train_all = np.asarray(x[train_idx], dtype=np.float32)
    y_train_all = concept_y[train_idx]
    sae_mu = np.asarray(x_train_all.mean(axis=0), dtype=np.float32)
    sae_sigma = np.asarray(x_train_all.std(axis=0), dtype=np.float32)
    sae_sigma = np.maximum(sae_sigma, 1e-6)

    cav = fit_cav_directions(x_train_all, y_train_all, alpha=args.concept_alpha)
    # CAV as a normalised-space covector: raw score x_raw @ beta becomes
    # x_norm @ (sigma * beta) up to a constant.
    cav_sae = np.asarray(cav["cav_raw_covector"], dtype=np.float64) * sae_sigma.astype(np.float64)
    cav_sae = cav_sae / max(float(np.linalg.norm(cav_sae)), 1e-12)

    leace = fit_leace_subspace(x_train_all, y_train_all, sae_sigma=sae_sigma, ridge=args.leace_ridge)

    out_dir = (
        args.out_dir
        / safe_name(suffix)
        / f"{safe_name(concept)}__{safe_name(task)}__L{layer:02d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sae_mu.npy": sae_mu,
        "sae_sigma.npy": sae_sigma,
        "cav_sae_norm.npy": cav_sae.astype(np.float32),
        "cav_standardized.npy": cav["cav_standardized"],
        "cav_raw_covector.npy": cav["cav_raw_covector"],
        "probe_scaler_mean.npy": cav["probe_scaler_mean"],
        "probe_scaler_scale.npy": cav["probe_scaler_scale"],
        "leace_u_sae_norm.npy": leace["leace_u_sae_norm"],
        "leace_u_raw.npy": leace["leace_u_raw"],
        "leace_u_whitened.npy": leace["leace_u_whitened"],
        "leace_mu.npy": leace["leace_mu"],
    }
    for name, value in arrays.items():
        np.save(out_dir / name, value)

    meta = {
        "model": row["model"],
        "suffix": suffix,
        "concept_id": concept,
        "task_id": row["task_id"],
        "canonical_task_id": task,
        "feature": feature,
        "layer": layer,
        "feature_path": str(feature_path.relative_to(ROOT)),
        "artifact_dir": str(out_dir.relative_to(ROOT)),
        "n_train": int(len(train_idx)),
        "activation_dim": int(x.shape[1]),
        "concept_alpha": args.concept_alpha,
        "leace_ridge": args.leace_ridge,
        "leace_ridge_abs": leace["leace_ridge_abs"],
        "leace_rank": leace["rank"],
        "sae_direction_for_a_geo": "leace_u_sae_norm.npy",
        "sae_direction_for_feature_ranking": "cav_sae_norm.npy",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    args.continuation_csv = args.continuation_csv.resolve()
    args.concepts_matrix = args.concepts_matrix.resolve()
    args.probe_features_root = args.probe_features_root.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(args)
    manifest = []
    failures = []
    for row in rows:
        try:
            manifest.append(export_one(row, args))
        except Exception as exc:  # noqa: BLE001 - report per-cell failure.
            failures.append(
                {
                    "model": row.get("model"),
                    "suffix": row.get("suffix"),
                    "concept_id": row.get("concept_id"),
                    "task_id": row.get("canonical_task_id") or row.get("task_id"),
                    "layer": row.get("layer"),
                    "error": repr(exc),
                }
            )

    fields = [
        "model",
        "suffix",
        "concept_id",
        "task_id",
        "canonical_task_id",
        "feature",
        "layer",
        "feature_path",
        "artifact_dir",
        "n_train",
        "activation_dim",
        "concept_alpha",
        "leace_ridge",
        "leace_ridge_abs",
        "leace_rank",
        "sae_direction_for_a_geo",
        "sae_direction_for_feature_ranking",
    ]
    with (args.out_dir / "manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)
    with (args.out_dir / "failures.csv").open("w", newline="") as f:
        fail_fields = ["model", "suffix", "concept_id", "task_id", "layer", "error"]
        writer = csv.DictWriter(f, fieldnames=fail_fields)
        writer.writeheader()
        writer.writerows(failures)

    report = {
        "continuation_csv": str(args.continuation_csv),
        "out_dir": str(args.out_dir),
        "selected_cells": len(rows),
        "exported_cells": len(manifest),
        "failed_cells": len(failures),
        "pilot_only": args.pilot_only,
    }
    (args.out_dir / "export_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
