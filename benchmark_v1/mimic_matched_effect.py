"""MIMIC-IV utilities for the final-layer matched-effect replication."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from benchmark_v1.accessibility_calibration import feature_concept_correlations
from benchmark_v1.matched_effect import RidgeReadout


PROTOCOL = "mimic_final_layer_matched_effect_v1"
PROTOCOL_100K = "mimic_final_layer_matched_effect_100k_v1"
SEEDS = (4311, 4312, 4313)
MODEL_SPECS = (
    ("CARDIAC-FM", "cardiac_fm_cu118_commons", 11, 12),
    ("CSFM", "csfm_cu118_commons", 5, 6),
    ("ECG-FM", "ecg_fm_cu118_commons", 11, 12),
    ("ECG-JEPA", "ecg_jepa_cu118_commons", 11, 12),
    ("HuBERT-ECG", "hubert_ecg_cu118_commons", 11, 12),
    ("ST-MEM", "st_mem_cu118_commons", 11, 12),
)
CONCEPT_SPECS = (
    ("heart_rate_bpm", "rate_rhythm"),
    ("qrs_duration_ms", "interval"),
    ("pr_interval_ms", "interval"),
    ("qt_like_ms", "interval"),
    ("r_amp_global_mv", "amplitude"),
    ("st_amp_global_mv", "st_t"),
    ("t_amp_global_mv", "st_t"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def safe_model_name(model: str) -> str:
    return model.lower().replace("-", "_")


def normalize_record_id(value: str) -> str:
    return str(value).split(":", 1)[-1]


def split_for_patient(patient_id: str) -> str:
    """Return the frozen external-head patient split used by the layer atlas."""
    digest = hashlib.sha256(f"external-head-v1:{patient_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "train" if bucket < 7 else "val" if bucket < 8 else "test"


def load_final_layer(
    benchmark_root: Path,
    model_suffix: str,
    final_layer: int,
    cohort: str = "mimic_f",
) -> tuple[list[dict[str, str]], np.ndarray]:
    """Load one indexed final layer while preserving shard record order."""
    index_root = (
        benchmark_root
        / "results/activations_external_full_v1/layer_atlas"
        / model_suffix
        / cohort
    )
    return load_indexed_layer(benchmark_root, index_root, final_layer)


def load_indexed_layer(
    benchmark_root: Path,
    index_root: Path,
    layer: int,
) -> tuple[list[dict[str, str]], np.ndarray]:
    """Load one layer from an arbitrary activation index in shard-offset order."""
    index_root = Path(index_root)
    shards = sorted(read_csv(index_root / "shards.csv"), key=lambda row: int(row["offset"]))
    if not shards:
        raise RuntimeError(f"no indexed shards in {index_root}")
    layer_name = f"layer_{layer:02d}.npy"
    records: list[dict[str, str]] = []
    arrays: list[np.ndarray] = []
    expected_offset = 0
    for shard in shards:
        if int(shard["offset"]) != expected_offset:
            raise RuntimeError(f"non-contiguous shard offsets in {index_root}")
        record_path = Path(shard["record_ids_file"])
        if not record_path.is_absolute():
            record_path = benchmark_root / record_path
        current_records = read_csv(record_path)
        pooled_path = Path(shard["pooled_file"])
        if not pooled_path.is_absolute():
            pooled_path = benchmark_root / pooled_path
        layer_path = pooled_path.parent / layer_name
        if layer_name not in str(shard["layer_files"]).split("|"):
            raise RuntimeError(f"{layer_name} is absent from {index_root}")
        values = np.asarray(np.load(layer_path, mmap_mode="r"), dtype=np.float32)
        if values.ndim != 2 or len(values) != len(current_records):
            raise RuntimeError(f"layer/record mismatch in {layer_path}")
        if not np.isfinite(values).all():
            raise RuntimeError(f"non-finite activations in {layer_path}")
        records.extend(current_records)
        arrays.append(values)
        expected_offset += len(current_records)
    activations = np.concatenate(arrays, axis=0)
    return records, activations


def aligned_concepts(
    record_ids: Sequence[str],
    manifest_rows: Sequence[dict[str, str]],
    train_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Align seven non-duplicated waveform concepts and retain missing labels."""
    train_mask = np.asarray(train_mask, dtype=bool)
    if len(record_ids) != len(train_mask):
        raise ValueError("record IDs and train mask do not align")
    by_id = {
        normalize_record_id(row["record_id"]): row
        for row in manifest_rows
        if row.get("status") == "ok"
    }
    names = [name for name, _ in CONCEPT_SPECS]
    values = np.full((len(record_ids), len(names)), np.nan, dtype=np.float64)
    for row_index, record_id in enumerate(record_ids):
        key = normalize_record_id(record_id)
        if key not in by_id:
            raise KeyError(f"missing MIMIC concept row for {record_id}")
        source = by_id[key]
        rr = _finite_float(source.get("rr_mean_ms"))
        for concept_index, name in enumerate(names):
            if name == "heart_rate_bpm":
                value = 60000.0 / rr if np.isfinite(rr) and rr > 0 else np.nan
            else:
                value = _finite_float(source.get(name))
            values[row_index, concept_index] = value
    means = np.nanmean(values[train_mask], axis=0)
    scales = np.nanstd(values[train_mask], axis=0)
    if not np.all(np.isfinite(means)):
        raise RuntimeError("one or more MIMIC concepts have no finite training values")
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    standardized = (values - means) / scales
    finite_counts = np.isfinite(values).sum(axis=0).astype(np.int64)
    return (
        standardized.astype(np.float32),
        names,
        means.astype(np.float32),
        scales.astype(np.float32),
        finite_counts,
    )


def masked_feature_concept_correlations(features, concepts: np.ndarray) -> np.ndarray:
    """Compute feature-concept correlations with a separate finite mask per target."""
    concepts = np.asarray(concepts, dtype=np.float64)
    if concepts.ndim != 2 or features.shape[0] != len(concepts):
        raise ValueError("feature and concept rows do not align")
    result = np.zeros((features.shape[1], concepts.shape[1]), dtype=np.float32)
    for target_index in range(concepts.shape[1]):
        valid = np.isfinite(concepts[:, target_index])
        if int(np.sum(valid)) < 3:
            raise ValueError(f"target {target_index} has fewer than three finite values")
        result[:, target_index] = feature_concept_correlations(
            features[valid], concepts[valid, target_index : target_index + 1]
        )[:, 0]
    return result


def fit_masked_ridge_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alphas: Sequence[float],
    min_train: int = 100,
    min_validation: int = 20,
    min_test: int = 20,
) -> tuple[RidgeReadout, np.ndarray]:
    """Fit one ridge per concept using only finite labels in each split."""
    from sklearn.linear_model import Ridge

    matrices = [
        np.asarray(value, dtype=np.float64)
        for value in (x_train, y_train, x_validation, y_validation, x_test, y_test)
    ]
    x_train, y_train, x_validation, y_validation, x_test, y_test = matrices
    if any(value.ndim != 2 for value in matrices):
        raise ValueError("all feature and target inputs must be matrices")
    if not (
        len(x_train) == len(y_train)
        and len(x_validation) == len(y_validation)
        and len(x_test) == len(y_test)
        and x_train.shape[1] == x_validation.shape[1] == x_test.shape[1]
        and y_train.shape[1] == y_validation.shape[1] == y_test.shape[1]
    ):
        raise ValueError("split dimensions do not align")
    alpha_values = tuple(float(value) for value in alphas)
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("alphas must be positive")

    n_features = x_train.shape[1]
    n_targets = y_train.shape[1]
    coefficients = np.full((n_features, n_targets), np.nan, dtype=np.float64)
    intercepts = np.full(n_targets, np.nan, dtype=np.float64)
    selected_alphas = np.full(n_targets, np.nan, dtype=np.float64)
    validation_correlations = np.full(n_targets, np.nan, dtype=np.float64)
    test_correlations = np.full(n_targets, np.nan, dtype=np.float64)
    counts = np.zeros((n_targets, 3), dtype=np.int64)
    for target_index in range(n_targets):
        masks = (
            np.isfinite(y_train[:, target_index]),
            np.isfinite(y_validation[:, target_index]),
            np.isfinite(y_test[:, target_index]),
        )
        counts[target_index] = [int(np.sum(mask)) for mask in masks]
        if np.any(counts[target_index] < (min_train, min_validation, min_test)):
            continue
        best: tuple[float, float, Ridge] | None = None
        for alpha in alpha_values:
            model = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
            model.fit(x_train[masks[0]], y_train[masks[0], target_index])
            prediction = model.predict(x_validation[masks[1]])
            score = abs(_pearson(y_validation[masks[1], target_index], prediction))
            if best is None or score > best[0]:
                best = (score, alpha, model)
        assert best is not None
        model = best[2]
        coefficients[:, target_index] = np.asarray(model.coef_, dtype=np.float64)
        intercepts[target_index] = float(model.intercept_)
        selected_alphas[target_index] = best[1]
        validation_correlations[target_index] = _pearson(
            y_validation[masks[1], target_index], model.predict(x_validation[masks[1]])
        )
        test_correlations[target_index] = _pearson(
            y_test[masks[2], target_index], model.predict(x_test[masks[2]])
        )
    if not np.all(np.isfinite(coefficients)):
        failed = np.flatnonzero(~np.all(np.isfinite(coefficients), axis=0)).tolist()
        raise RuntimeError(f"readout concepts failed finite-count gates: {failed}")
    return (
        RidgeReadout(
            coefficients=coefficients.astype(np.float32),
            intercepts=intercepts.astype(np.float32),
            selected_alphas=selected_alphas,
            validation_correlations=validation_correlations,
            test_correlations=test_correlations,
            feature_mean=x_train.mean(axis=0).astype(np.float32),
            feature_scale=np.where(x_train.std(axis=0) > 1e-8, x_train.std(axis=0), 1.0).astype(np.float32),
        ),
        counts,
    )


def _finite_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator > 1e-12 else 0.0
