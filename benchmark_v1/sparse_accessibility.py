"""Shared utilities for matched-budget sparse accessibility curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from benchmark_v1.accessibility_calibration import columnwise_pearson


@dataclass(frozen=True)
class SparseCurveResult:
    """Validation-selected ridge curves for independently ranked targets."""

    ks: np.ndarray
    selected_features: np.ndarray
    selected_alphas: np.ndarray
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    validation_correlations: np.ndarray
    test_correlations: np.ndarray


class PatientClusterBootstrap:
    """Reusable patient-cluster bootstrap for fixed multivariate targets."""

    def __init__(
        self,
        patient_ids: Sequence[str],
        targets: np.ndarray,
        *,
        draws: int,
        seed: int,
    ) -> None:
        targets = np.asarray(targets, dtype=np.float64)
        if targets.ndim != 2 or len(patient_ids) != len(targets):
            raise ValueError("patient_ids and target rows must align")
        if draws < 1:
            raise ValueError("draws must be positive")
        patients, inverse = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
        weights = np.random.default_rng(seed).multinomial(
            len(patients),
            np.full(len(patients), 1.0 / len(patients)),
            size=draws,
        ).astype(np.float64)
        count = np.zeros(len(patients), dtype=np.float64)
        sum_y = np.zeros((len(patients), targets.shape[1]), dtype=np.float64)
        sum_y2 = np.zeros_like(sum_y)
        np.add.at(count, inverse, 1.0)
        np.add.at(sum_y, inverse, targets)
        np.add.at(sum_y2, inverse, targets * targets)
        self.inverse = inverse
        self.weights = weights
        self.n = weights @ count
        self.sum_y = weights @ sum_y
        self.sum_y2 = weights @ sum_y2
        self.n_records = len(targets)
        self.n_targets = targets.shape[1]
        self.n_patients = len(patients)
        self._targets = targets

    def correlations(self, predictions: np.ndarray) -> np.ndarray:
        values = np.asarray(predictions, dtype=np.float64)
        if values.shape != (self.n_records, self.n_targets):
            raise ValueError("prediction shape does not match bootstrap targets")
        n_patients = self.n_patients
        sum_x = np.zeros((n_patients, self.n_targets), dtype=np.float64)
        sum_x2 = np.zeros_like(sum_x)
        sum_xy = np.zeros_like(sum_x)
        targets = self._targets
        np.add.at(sum_x, self.inverse, values)
        np.add.at(sum_x2, self.inverse, values * values)
        np.add.at(sum_xy, self.inverse, values * targets)
        sx = self.weights @ sum_x
        sx2 = self.weights @ sum_x2
        sxy = self.weights @ sum_xy
        covariance = sxy - sx * self.sum_y / self.n[:, None]
        var_x = np.maximum(sx2 - sx * sx / self.n[:, None], 0.0)
        var_y = np.maximum(
            self.sum_y2 - self.sum_y * self.sum_y / self.n[:, None], 0.0
        )
        denominator = np.sqrt(var_x * var_y)
        return np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 1e-12,
        )


def deterministic_feature_subset(
    n_features: int, budget: int, seed: int
) -> np.ndarray:
    """Draw a deterministic feature subset without using target information."""
    if n_features < 1 or budget < 1 or budget > n_features:
        raise ValueError("feature budget must be within 1..n_features")
    if budget == n_features:
        return np.arange(n_features, dtype=np.int64)
    values = np.random.default_rng(seed).choice(n_features, budget, replace=False)
    return np.sort(values.astype(np.int64))


def deterministic_subset_from_candidates(
    candidate_features: Sequence[int], budget: int, seed: int
) -> np.ndarray:
    """Draw a deterministic subset from a target-independent candidate pool."""
    candidates = np.asarray(candidate_features, dtype=np.int64)
    if candidates.ndim != 1 or len(candidates) == 0:
        raise ValueError("candidate_features must be a nonempty vector")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate_features must be unique")
    if budget < 1 or budget > len(candidates):
        raise ValueError("feature budget must be within the candidate pool")
    if budget == len(candidates):
        return np.sort(candidates)
    selected = np.random.default_rng(seed).choice(candidates, budget, replace=False)
    return np.sort(selected.astype(np.int64))


def candidate_ranking(
    correlations: np.ndarray, candidate_features: Sequence[int] | None = None
) -> np.ndarray:
    """Rank candidates by absolute train correlation for every target."""
    values = np.asarray(correlations)
    if values.ndim != 2:
        raise ValueError("correlations must be feature-by-target")
    if candidate_features is None:
        candidates = np.arange(values.shape[0], dtype=np.int64)
    else:
        candidates = np.asarray(candidate_features, dtype=np.int64)
        if candidates.ndim != 1 or len(candidates) == 0:
            raise ValueError("candidate_features must be a nonempty vector")
        if len(np.unique(candidates)) != len(candidates):
            raise ValueError("candidate_features must be unique")
        if np.any(candidates < 0) or np.any(candidates >= values.shape[0]):
            raise ValueError("candidate feature outside correlation matrix")
    local = np.argsort(-np.abs(values[candidates]), axis=0, kind="stable")
    return candidates[local]


def _dense_columns(matrix, columns: np.ndarray) -> np.ndarray:
    selected = matrix[:, columns]
    try:
        from scipy import sparse

        if sparse.issparse(selected):
            selected = selected.toarray()
    except ImportError:
        pass
    return np.asarray(selected, dtype=np.float64)


def _pearson_one(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(columnwise_pearson(y_true[:, None], y_score[:, None])[0])


def fit_sparse_ridge_curve(
    x_train,
    y_train: np.ndarray,
    x_validation,
    y_validation: np.ndarray,
    x_test,
    y_test: np.ndarray,
    ranking: np.ndarray,
    *,
    ks: Sequence[int],
    alphas: Sequence[float],
) -> SparseCurveResult:
    """Fit target-specific top-k ridge probes with validation-only alpha choice."""
    y_train = np.asarray(y_train, dtype=np.float64)
    y_validation = np.asarray(y_validation, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64)
    ranking = np.asarray(ranking, dtype=np.int64)
    ks_array = np.asarray(tuple(int(value) for value in ks), dtype=np.int64)
    alpha_array = np.asarray(tuple(float(value) for value in alphas), dtype=np.float64)
    if y_train.ndim != 2 or y_validation.ndim != 2 or y_test.ndim != 2:
        raise ValueError("targets must be matrices")
    if not (y_train.shape[1] == y_validation.shape[1] == y_test.shape[1]):
        raise ValueError("target matrices must have equal column counts")
    if ranking.ndim != 2 or ranking.shape[1] != y_train.shape[1]:
        raise ValueError("ranking must provide candidates for every target")
    if len(ks_array) == 0 or np.any(ks_array < 1) or np.any(np.diff(ks_array) <= 0):
        raise ValueError("ks must be strictly increasing positive integers")
    if ks_array[-1] > ranking.shape[0]:
        raise ValueError("largest k exceeds candidate count")
    if len(alpha_array) == 0 or np.any(alpha_array < 0):
        raise ValueError("alphas must be nonnegative")
    for name, matrix, targets in (
        ("train", x_train, y_train),
        ("validation", x_validation, y_validation),
        ("test", x_test, y_test),
    ):
        if matrix.shape[0] != targets.shape[0]:
            raise ValueError(f"{name} features and targets have different rows")

    n_k = len(ks_array)
    n_targets = y_train.shape[1]
    validation_predictions = np.empty(
        (n_k, len(y_validation), n_targets), dtype=np.float32
    )
    test_predictions = np.empty((n_k, len(y_test), n_targets), dtype=np.float32)
    selected_alphas = np.empty((n_k, n_targets), dtype=np.float64)
    selected_features = np.empty((n_k, n_targets), dtype=object)

    max_k = int(ks_array[-1])
    for target in range(n_targets):
        columns = ranking[:max_k, target]
        train = _dense_columns(x_train, columns)
        validation = _dense_columns(x_validation, columns)
        test = _dense_columns(x_test, columns)
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale = np.where(scale > 1e-8, scale, 1.0)
        train = (train - mean) / scale
        validation = (validation - mean) / scale
        test = (test - mean) / scale
        target_train = y_train[:, target]
        intercept = float(target_train.mean())
        centered_target = target_train - intercept
        gram = train.T @ train
        cross = train.T @ centered_target

        for k_index, k_value in enumerate(ks_array):
            k = int(k_value)
            local_gram = gram[:k, :k]
            local_cross = cross[:k]
            best_score = -np.inf
            best_alpha = float(alpha_array[0])
            best_validation = None
            best_test = None
            for alpha in alpha_array:
                regularized = local_gram + float(alpha) * np.eye(k)
                try:
                    coefficients = np.linalg.solve(regularized, local_cross)
                except np.linalg.LinAlgError:
                    coefficients = np.linalg.lstsq(
                        regularized, local_cross, rcond=None
                    )[0]
                prediction_validation = validation[:, :k] @ coefficients + intercept
                validation_r = _pearson_one(
                    y_validation[:, target], prediction_validation
                )
                score = abs(validation_r)
                if score > best_score:
                    best_score = score
                    best_alpha = float(alpha)
                    best_validation = prediction_validation
                    best_test = test[:, :k] @ coefficients + intercept
            validation_predictions[k_index, :, target] = np.asarray(
                best_validation, dtype=np.float32
            )
            test_predictions[k_index, :, target] = np.asarray(
                best_test, dtype=np.float32
            )
            selected_alphas[k_index, target] = best_alpha
            selected_features[k_index, target] = tuple(
                int(value) for value in columns[:k]
            )

    validation_correlations = np.stack(
        [columnwise_pearson(y_validation, values) for values in validation_predictions]
    )
    test_correlations = np.stack(
        [columnwise_pearson(y_test, values) for values in test_predictions]
    )
    return SparseCurveResult(
        ks=ks_array,
        selected_features=selected_features,
        selected_alphas=selected_alphas,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        validation_correlations=validation_correlations,
        test_correlations=test_correlations,
    )


def normalized_log2_curve_auc(ks: Sequence[int], values: Sequence[float]) -> float:
    """Integrate a curve over log2(k) and normalize to the observed domain."""
    x = np.log2(np.asarray(tuple(ks), dtype=np.float64))
    y = np.asarray(tuple(values), dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("ks and values must be equal-length vectors with at least two points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("ks must be strictly increasing")
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values in original order."""
    values = np.asarray(tuple(pvalues), dtype=np.float64)
    if values.ndim != 1 or np.any(~np.isfinite(values)):
        raise ValueError("pvalues must be a finite vector")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result
