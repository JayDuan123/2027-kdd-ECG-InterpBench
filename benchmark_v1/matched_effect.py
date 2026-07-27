"""Shared computations for validation-matched representation interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from benchmark_v1.accessibility_calibration import columnwise_pearson


@dataclass(frozen=True)
class RidgeReadout:
    """Multi-target ridge readout expressed in the input coordinate system."""

    coefficients: np.ndarray
    intercepts: np.ndarray
    selected_alphas: np.ndarray
    validation_correlations: np.ndarray
    test_correlations: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray


@dataclass(frozen=True)
class UnitIntervention:
    """Readout and representation changes produced by an alpha-one clamp."""

    validation_outputs: np.ndarray
    test_outputs: np.ndarray
    validation_l2: np.ndarray
    test_l2: np.ndarray
    selected_features: np.ndarray
    centroid: np.ndarray

    def validation_target_effect(self, target_index: int) -> float:
        return float(np.mean(self.validation_outputs[:, target_index]))


def _as_dense_columns(matrix, columns: np.ndarray) -> np.ndarray:
    selected = matrix[:, columns]
    try:
        from scipy import sparse

        if sparse.issparse(selected):
            selected = selected.toarray()
    except ImportError:
        pass
    return np.asarray(selected, dtype=np.float64)


def fit_multitarget_ridge_readout(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alphas: Sequence[float],
) -> RidgeReadout:
    """Fit ridge once per alpha and select alpha per target on validation only."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    y_validation = np.asarray(y_validation, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("training features and targets must be matrices")
    if not (
        x_train.shape[1] == x_validation.shape[1] == x_test.shape[1]
        and y_train.shape[1] == y_validation.shape[1] == y_test.shape[1]
    ):
        raise ValueError("split dimensions do not match")
    if not (
        len(x_train) == len(y_train)
        and len(x_validation) == len(y_validation)
        and len(x_test) == len(y_test)
    ):
        raise ValueError("feature and target rows do not match")
    alpha_values = np.asarray(tuple(float(value) for value in alphas), dtype=np.float64)
    if len(alpha_values) == 0 or np.any(alpha_values <= 0):
        raise ValueError("alphas must be positive")

    feature_mean = x_train.mean(axis=0)
    feature_scale = x_train.std(axis=0)
    feature_scale = np.where(feature_scale > 1e-8, feature_scale, 1.0)
    train = (x_train - feature_mean) / feature_scale
    validation = (x_validation - feature_mean) / feature_scale
    test = (x_test - feature_mean) / feature_scale
    target_mean = y_train.mean(axis=0)
    centered_targets = y_train - target_mean

    gram = train.T @ train
    cross = train.T @ centered_targets
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected = eigenvectors.T @ cross

    n_targets = y_train.shape[1]
    selected_scaled = np.empty((x_train.shape[1], n_targets), dtype=np.float64)
    selected_alphas = np.empty(n_targets, dtype=np.float64)
    best_scores = np.full(n_targets, -np.inf, dtype=np.float64)
    for alpha in alpha_values:
        coefficients = eigenvectors @ (projected / (eigenvalues[:, None] + alpha))
        predictions = validation @ coefficients + target_mean
        scores = np.abs(columnwise_pearson(y_validation, predictions))
        improved = scores > best_scores
        selected_scaled[:, improved] = coefficients[:, improved]
        selected_alphas[improved] = alpha
        best_scores[improved] = scores[improved]

    coefficients = selected_scaled / feature_scale[:, None]
    intercepts = target_mean - feature_mean @ coefficients
    validation_predictions = x_validation @ coefficients + intercepts
    test_predictions = x_test @ coefficients + intercepts
    return RidgeReadout(
        coefficients=coefficients.astype(np.float32),
        intercepts=intercepts.astype(np.float32),
        selected_alphas=selected_alphas,
        validation_correlations=columnwise_pearson(
            y_validation, validation_predictions
        ),
        test_correlations=columnwise_pearson(y_test, test_predictions),
        feature_mean=feature_mean.astype(np.float32),
        feature_scale=feature_scale.astype(np.float32),
    )


def centroid_unit_intervention(
    representation_train,
    representation_validation,
    representation_test,
    y_train: np.ndarray,
    ranking: np.ndarray,
    basis_to_dense: np.ndarray,
    readout_coefficients: np.ndarray,
    *,
    target_index: int,
    k: int,
    high_quantile: float = 0.75,
) -> UnitIntervention:
    """Clamp train-selected coordinates toward their high-target train centroid."""
    y_train = np.asarray(y_train, dtype=np.float64)
    ranking = np.asarray(ranking, dtype=np.int64)
    basis_to_dense = np.asarray(basis_to_dense, dtype=np.float64)
    readout_coefficients = np.asarray(readout_coefficients, dtype=np.float64)
    if not 0.5 <= high_quantile < 1.0:
        raise ValueError("high_quantile must be in [0.5, 1)")
    if y_train.ndim != 2 or not 0 <= target_index < y_train.shape[1]:
        raise ValueError("invalid target matrix or target index")
    if ranking.ndim != 2 or ranking.shape[1] != y_train.shape[1]:
        raise ValueError("ranking must provide features for every target")
    if k < 1 or k > ranking.shape[0]:
        raise ValueError("k exceeds the available ranking")
    if basis_to_dense.ndim != 2 or readout_coefficients.ndim != 2:
        raise ValueError("basis and readout coefficients must be matrices")
    if basis_to_dense.shape[1] != readout_coefficients.shape[0]:
        raise ValueError("basis and readout dimensions do not align")
    if representation_train.shape[1] != basis_to_dense.shape[0]:
        raise ValueError("representation width and basis rows do not align")
    if representation_train.shape[0] != len(y_train):
        raise ValueError("training representation and targets do not align")

    selected = np.asarray(ranking[:k, target_index], dtype=np.int64)
    target_values = y_train[:, target_index]
    finite_target = np.isfinite(target_values)
    if int(np.sum(finite_target)) < max(8, k):
        raise ValueError("target has too few finite training records")
    threshold = float(np.quantile(target_values[finite_target], high_quantile))
    high = finite_target & (target_values >= threshold)
    if int(np.sum(high)) < max(8, k):
        raise ValueError("high-target centroid has too few training records")
    train_selected = _as_dense_columns(representation_train, selected)
    centroid = train_selected[high].mean(axis=0)
    validation_selected = _as_dense_columns(representation_validation, selected)
    test_selected = _as_dense_columns(representation_test, selected)
    validation_change = centroid[None, :] - validation_selected
    test_change = centroid[None, :] - test_selected
    selected_basis = basis_to_dense[selected]
    feature_readout = selected_basis @ readout_coefficients
    validation_outputs = validation_change @ feature_readout
    test_outputs = test_change @ feature_readout
    validation_dense = validation_change @ selected_basis
    test_dense = test_change @ selected_basis
    return UnitIntervention(
        validation_outputs=validation_outputs.astype(np.float32),
        test_outputs=test_outputs.astype(np.float32),
        validation_l2=np.linalg.norm(validation_dense, axis=1).astype(np.float32),
        test_l2=np.linalg.norm(test_dense, axis=1).astype(np.float32),
        selected_features=selected,
        centroid=centroid.astype(np.float32),
    )


def common_validation_effect(
    interventions: Mapping[str, UnitIntervention],
    target_index: int,
    *,
    cap: float = 0.25,
    floor: float = 0.05,
    max_alpha: float = 1.0,
) -> tuple[float, dict[str, float], str]:
    """Choose a shared validation effect without using test outcomes."""
    if not interventions:
        raise ValueError("at least one intervention is required")
    if not (0 < floor <= cap and max_alpha > 0):
        raise ValueError("invalid effect floor, cap, or alpha bound")
    effects = {
        name: value.validation_target_effect(target_index)
        for name, value in interventions.items()
    }
    if any(not np.isfinite(value) or value <= 0 for value in effects.values()):
        return float("nan"), effects, "nonpositive_validation_effect"
    common = min(cap, *(max_alpha * value for value in effects.values()))
    if common < floor:
        return common, effects, "below_effect_floor"
    return common, effects, "eligible"


def calibrated_record_metrics(
    intervention: UnitIntervention,
    *,
    target_index: int,
    target_family: str,
    concept_families: Sequence[str],
    target_effect: float,
    unit_validation_effect: float,
    max_alpha: float = 1.0,
) -> tuple[np.ndarray, float, str]:
    """Return per-record target, off-target, and L2 metrics at matched effect."""
    families = np.asarray(tuple(concept_families), dtype=object)
    if len(families) != intervention.test_outputs.shape[1]:
        raise ValueError("concept families do not align with readout outputs")
    if not np.isfinite(target_effect) or target_effect <= 0:
        return np.empty((0, 5), dtype=np.float32), float("nan"), "invalid_target_effect"
    if not np.isfinite(unit_validation_effect) or unit_validation_effect <= 0:
        return np.empty((0, 5), dtype=np.float32), float("nan"), "nonpositive_validation_effect"
    alpha = target_effect / unit_validation_effect
    if alpha > max_alpha + 1e-10:
        return np.empty((0, 5), dtype=np.float32), alpha, "alpha_exceeds_bound"
    delta = np.asarray(intervention.test_outputs, dtype=np.float64) * alpha
    target = delta[:, target_index]
    other = np.ones(delta.shape[1], dtype=bool)
    other[target_index] = False
    cross_family = other & (families != target_family)
    if not np.any(cross_family):
        raise ValueError("target has no cross-family concepts")
    all_rms = np.sqrt(np.mean(np.square(delta[:, other]), axis=1))
    cross_rms = np.sqrt(np.mean(np.square(delta[:, cross_family]), axis=1))
    max_abs = np.max(np.abs(delta[:, other]), axis=1)
    l2 = np.asarray(intervention.test_l2, dtype=np.float64) * alpha
    return (
        np.column_stack((target, cross_rms, all_rms, max_abs, l2)).astype(np.float32),
        float(alpha),
        "eligible",
    )


def aggregate_patient_means(
    values: np.ndarray, patient_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate fixed record metrics to sorted patient means and counts."""
    values = np.asarray(values, dtype=np.float64)
    patients, inverse = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
    if values.ndim != 2 or len(values) != len(inverse):
        raise ValueError("record metrics and patient identifiers do not align")
    counts = np.zeros(len(patients), dtype=np.float64)
    sums = np.zeros((len(patients), values.shape[1]), dtype=np.float64)
    np.add.at(counts, inverse, 1.0)
    np.add.at(sums, inverse, values)
    means = sums / counts[:, None]
    return patients, means.astype(np.float32), counts.astype(np.float32)


def bootstrap_patient_metric_means(
    patient_means: np.ndarray,
    patient_counts: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Bootstrap record-weighted means from patient-level sufficient statistics."""
    patient_means = np.asarray(patient_means, dtype=np.float64)
    patient_counts = np.asarray(patient_counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if patient_means.ndim != 2 or patient_counts.shape != (len(patient_means),):
        raise ValueError("patient means and counts do not align")
    if weights.ndim != 2 or weights.shape[1] != len(patient_means):
        raise ValueError("bootstrap weights do not align with patients")
    sums = patient_means * patient_counts[:, None]
    denominator = weights @ patient_counts
    return (weights @ sums) / denominator[:, None]
