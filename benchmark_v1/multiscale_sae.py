"""Protocol and metrics for the multi-scale ECG-FM SAE benchmark."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


MODEL_SUFFIXES = {
    "CARDIAC-FM": "cardiac_fm_cu118_commons",
    "CSFM": "csfm_cu118_commons",
    "ECG-FM": "ecg_fm_cu118_commons",
    "ECG-JEPA": "ecg_jepa_cu118_commons",
    "HuBERT-ECG": "hubert_ecg_cu118_commons",
    "ST-MEM": "st_mem_cu118_commons",
}

DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_EXPANSIONS = (1, 4, 8, 16, 32)
DEFAULT_SEEDS = (4311, 4312, 4313)


@dataclass(frozen=True)
class LayerSpec:
    model: str
    suffix: str
    layer: int
    target_relative_depth: float
    actual_relative_depth: float
    n_layers: int
    d_hidden: int
    activation_path: Path
    records_path: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def relative_layer_indices(n_layers: int, depths: Sequence[float] = DEFAULT_DEPTHS) -> list[int]:
    """Map normalized depths to unique layers using deterministic half-up rounding."""
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    indices = []
    for depth in depths:
        if not 0.0 <= float(depth) <= 1.0:
            raise ValueError(f"relative depth outside [0, 1]: {depth}")
        index = int(math.floor(float(depth) * (n_layers - 1) + 0.5))
        if index not in indices:
            indices.append(index)
    return indices


def sparsity_for(arm: str, expansion: int, d_hidden: int) -> tuple[int, int]:
    """Return ``(N, k)`` for a preregistered relative-capacity protocol."""
    if expansion <= 0 or d_hidden <= 0:
        raise ValueError("expansion and d_hidden must be positive")
    n_features = int(expansion * d_hidden)
    if arm == "fixed_k_over_d":
        k = max(1, int(round(d_hidden / 8)))
    elif arm == "fixed_k_over_n":
        k = max(1, int(round(n_features / 64)))
    else:
        raise ValueError(f"unknown sparsity arm: {arm}")
    if k >= n_features:
        raise ValueError(f"invalid sparsity: k={k}, N={n_features}")
    return n_features, k


def canonical_config_hash(row: dict[str, object]) -> str:
    keys = (
        "model",
        "feature_suffix",
        "layer",
        "relative_depth",
        "actual_relative_depth",
        "n_layers",
        "d_hidden",
        "sparsity_arm",
        "expansion_E",
        "N",
        "k",
        "seed",
        "steps",
        "batch_size",
        "learning_rate",
    )
    payload = {key: row[key] for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def discover_layer_specs(
    root: Path,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    model_suffixes: dict[str, str] = MODEL_SUFFIXES,
) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for model, suffix in model_suffixes.items():
        feature_root = root / "results" / "probe_features" / suffix
        features_path = feature_root / "features.csv"
        records_path = feature_root / "records.csv"
        if not features_path.exists() or not records_path.exists():
            raise FileNotFoundError(f"missing probe features for {model}: {feature_root}")
        rows = [row for row in read_csv(features_path) if row["feature"].startswith("layer_")]
        rows.sort(key=lambda row: int(row["feature"].split("_")[1]))
        if not rows:
            raise RuntimeError(f"no layer features found for {model}")
        seen_layers: set[int] = set()
        for target_depth in depths:
            ordinal = relative_layer_indices(len(rows), (target_depth,))[0]
            if ordinal in seen_layers:
                continue
            seen_layers.add(ordinal)
            row = rows[ordinal]
            layer = int(row["feature"].split("_")[1])
            shape = ast.literal_eval(row["shape"])
            if len(shape) != 2:
                raise ValueError(f"expected matrix shape for {model} layer {layer}: {shape}")
            activation_path = root / row["file"]
            if not activation_path.exists():
                raise FileNotFoundError(activation_path)
            specs.append(
                LayerSpec(
                    model=model,
                    suffix=suffix,
                    layer=layer,
                    target_relative_depth=float(target_depth),
                    actual_relative_depth=layer / max(len(rows) - 1, 1),
                    n_layers=len(rows),
                    d_hidden=int(shape[1]),
                    activation_path=activation_path,
                    records_path=records_path,
                )
            )
    return specs


def build_manifest_rows(
    root: Path,
    output_root: Path,
    expansions: Iterable[int] = DEFAULT_EXPANSIONS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    depths: Sequence[float] = DEFAULT_DEPTHS,
    sparsity_arm: str = "fixed_k_over_d",
    steps: int = 8000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> list[dict[str, object]]:
    return build_manifest_rows_from_specs(
        discover_layer_specs(root, depths=depths),
        output_root,
        expansions=expansions,
        seeds=seeds,
        sparsity_arm=sparsity_arm,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )


def build_manifest_rows_from_specs(
    specs: Iterable[LayerSpec],
    output_root: Path,
    expansions: Iterable[int] = DEFAULT_EXPANSIONS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    sparsity_arm: str = "fixed_k_over_d",
    steps: int = 8000,
    batch_size: int = 256,
    learning_rate: float = 3e-4,
) -> list[dict[str, object]]:
    """Build the matched grid from an explicit, cohort-specific layer catalog."""
    rows: list[dict[str, object]] = []
    for spec in specs:
        safe_model = spec.model.lower().replace("-", "_")
        for expansion in expansions:
            n_features, k = sparsity_for(sparsity_arm, int(expansion), spec.d_hidden)
            for seed in seeds:
                cell_root = (
                    output_root
                    / "checkpoints"
                    / safe_model
                    / f"layer_{spec.layer:02d}"
                    / f"E{int(expansion)}"
                    / f"seed{int(seed)}"
                )
                row: dict[str, object] = {
                    "task_index": len(rows),
                    "model": spec.model,
                    "model_safe": safe_model,
                    "feature_suffix": spec.suffix,
                    "layer": spec.layer,
                    "relative_depth": round(spec.target_relative_depth, 8),
                    "actual_relative_depth": round(spec.actual_relative_depth, 8),
                    "n_layers": spec.n_layers,
                    "d_hidden": spec.d_hidden,
                    "sparsity_arm": sparsity_arm,
                    "expansion_E": int(expansion),
                    "N": n_features,
                    "k": k,
                    "k_over_d": k / spec.d_hidden,
                    "k_over_N": k / n_features,
                    "seed": int(seed),
                    "steps": int(steps),
                    "batch_size": int(batch_size),
                    "learning_rate": float(learning_rate),
                    "activation_path": str(spec.activation_path),
                    "records_path": str(spec.records_path),
                    "checkpoint": str(cell_root / f"batchtopk_N{n_features}_k{k}.pt"),
                    "metrics": str(cell_root / "metrics.json"),
                    "concept_metrics": str(cell_root / "concept_metrics.csv"),
                    "firing_rate": str(cell_root / "firing_rate.npy"),
                }
                row["config_hash"] = canonical_config_hash(row)
                rows.append(row)
    return rows


def standardized_concepts(
    record_ids: Sequence[str],
    concept_rows: Sequence[dict[str, str]],
    train_mask: np.ndarray,
    preserve_missing: bool = False,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Align concept values to activation rows and standardize with train-only stats."""
    if len(record_ids) != len(train_mask):
        raise ValueError("record_ids and train_mask have different lengths")
    if not concept_rows:
        raise ValueError("concept matrix is empty")
    names = [name for name in concept_rows[0] if name != "ecg_id"]
    by_id = {str(row["ecg_id"]): row for row in concept_rows}
    values = np.full((len(record_ids), len(names)), np.nan, dtype=np.float32)
    for index, ecg_id in enumerate(record_ids):
        row = by_id.get(str(ecg_id))
        if row is None:
            raise KeyError(f"concept row missing for ecg_id={ecg_id}")
        for concept_index, name in enumerate(names):
            try:
                values[index, concept_index] = float(row[name])
            except (TypeError, ValueError):
                pass
    train = values[train_mask]
    mean = np.nanmean(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    standardized = (values - mean) / scale
    if preserve_missing:
        standardized[~np.isfinite(values)] = np.nan
    else:
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=0.0, neginf=0.0)
    return standardized.astype(np.float32), names, mean.astype(np.float32), scale.astype(np.float32)


def correlation_from_sufficient_statistics(
    count: int,
    sum_z: np.ndarray,
    sum_z2: np.ndarray,
    sum_y: np.ndarray,
    sum_y2: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    """Compute feature-by-concept Pearson correlations from streaming sums."""
    if count <= 1:
        return np.zeros_like(cross, dtype=np.float32)
    covariance = cross - np.outer(sum_z, sum_y) / count
    var_z = np.maximum(sum_z2 - np.square(sum_z) / count, 0.0)
    var_y = np.maximum(sum_y2 - np.square(sum_y) / count, 0.0)
    denom = np.sqrt(np.outer(var_z, var_y))
    corr = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 1e-12)
    return np.clip(corr, -1.0, 1.0).astype(np.float32)


def selected_concept_metrics(
    train_corr: np.ndarray,
    eval_corr: np.ndarray,
    concept_names: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    """Select one feature per concept on train and evaluate it without reselection."""
    if train_corr.shape != eval_corr.shape:
        raise ValueError("train and evaluation correlation matrices must have equal shape")
    if train_corr.shape[1] != len(concept_names):
        raise ValueError("concept name count does not match correlation matrix")
    rows: list[dict[str, object]] = []
    signed_values = []
    absolute_values = []
    sign_matches = []
    for concept_index, concept in enumerate(concept_names):
        feature = int(np.argmax(np.abs(train_corr[:, concept_index])))
        train_value = float(train_corr[feature, concept_index])
        eval_value = float(eval_corr[feature, concept_index])
        signed = float(np.sign(train_value) * eval_value) if train_value != 0 else 0.0
        absolute = abs(eval_value)
        sign_match = float(train_value * eval_value > 0)
        rows.append(
            {
                "concept": concept,
                "selected_feature": feature,
                "train_correlation": train_value,
                "eval_correlation": eval_value,
                "sign_aligned_eval_correlation": signed,
                "abs_eval_correlation": absolute,
                "sign_match": bool(sign_match),
            }
        )
        signed_values.append(signed)
        absolute_values.append(absolute)
        sign_matches.append(sign_match)
    absolute_array = np.asarray(absolute_values, dtype=float)
    summary = {
        "mean_train_selected_abs_correlation": float(np.mean(absolute_array)),
        "median_train_selected_abs_correlation": float(np.median(absolute_array)),
        "mean_sign_aligned_correlation": float(np.mean(signed_values)),
        "sign_consistency_fraction": float(np.mean(sign_matches)),
        "coverage_abs_r_ge_0_10": float(np.mean(absolute_array >= 0.10)),
        "coverage_abs_r_ge_0_20": float(np.mean(absolute_array >= 0.20)),
    }
    return rows, summary
