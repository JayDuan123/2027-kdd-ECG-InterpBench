"""Train-selected dictionary association metrics for dense and sparse features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SelectionProfile:
    """Frozen train selection evaluated on a second split."""

    selected_index: np.ndarray
    train_value: np.ndarray
    test_value: np.ndarray
    test_oriented_value: np.ndarray
    test_strength: np.ndarray


def _as_2d(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    return result


def _sparse_column_ranks(values: np.ndarray, n_implicit_zero: int) -> tuple[np.ndarray, float]:
    """Return average ranks for stored values and the rank of implicit zeros."""
    values = np.asarray(values, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    zero_at = np.searchsorted(unique, 0.0)
    if zero_at == len(unique) or unique[zero_at] != 0.0:
        unique = np.insert(unique, zero_at, 0.0)
        counts = np.insert(counts, zero_at, n_implicit_zero)
        inverse = inverse + (inverse >= zero_at)
    else:
        counts[zero_at] += n_implicit_zero
    cumulative = np.cumsum(counts)
    before = cumulative - counts
    average_ranks = before + (counts + 1.0) / 2.0
    return average_ranks[inverse], float(average_ranks[zero_at])


def tie_aware_auc_matrix(features, labels: np.ndarray) -> np.ndarray:
    """Compute feature-by-task AUROC with average ranks for tied scores."""
    from scipy import sparse
    from scipy.stats import rankdata

    labels = _as_2d(labels, "labels").astype(np.float64, copy=False)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must share a row dimension")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("labels must be binary")
    n = labels.shape[0]
    positive = labels.sum(axis=0)
    negative = n - positive
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError("every task must contain both classes")

    if sparse.issparse(features):
        matrix = features.tocsc(copy=True).astype(np.float64)
        matrix.eliminate_zeros()
        rank_sums = np.zeros((matrix.shape[1], labels.shape[1]), dtype=np.float64)
        for feature in range(matrix.shape[1]):
            start, end = matrix.indptr[feature : feature + 2]
            rows = matrix.indices[start:end]
            values = matrix.data[start:end]
            stored_ranks, zero_rank = _sparse_column_ranks(values, n - len(values))
            positive_stored = labels[rows].sum(axis=0) if len(rows) else 0.0
            rank_sums[feature] = zero_rank * (positive - positive_stored)
            if len(rows):
                rank_sums[feature] += stored_ranks @ labels[rows]
    else:
        matrix = _as_2d(np.asarray(features, dtype=np.float64), "features")
        ranks = rankdata(matrix, axis=0, method="average")
        rank_sums = ranks.T @ labels

    auc = (rank_sums - positive * (positive + 1.0) / 2.0) / (positive * negative)
    return np.clip(auc, 0.0, 1.0).astype(np.float32)


def nonconstant_feature_mask(features, tolerance: float = 1e-12) -> np.ndarray:
    """Identify feature columns with nonzero empirical variance."""
    from scipy import sparse

    if sparse.issparse(features):
        matrix = features.astype(np.float64, copy=False)
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_square = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
        variance = np.maximum(mean_square - np.square(mean), 0.0)
    else:
        variance = np.var(np.asarray(features, dtype=np.float64), axis=0)
    return np.asarray(variance > tolerance, dtype=bool)


def feature_centric_profile(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    center: float,
) -> SelectionProfile:
    """Select one target per feature on train and freeze it on test."""
    train_values = _as_2d(train_values, "train_values")
    test_values = _as_2d(test_values, "test_values")
    if train_values.shape != test_values.shape:
        raise ValueError("train and test association matrices must have equal shape")
    selected = np.argmax(np.abs(train_values - center), axis=1)
    rows = np.arange(train_values.shape[0])
    train = train_values[rows, selected]
    test = test_values[rows, selected]
    direction = np.where(train >= center, 1.0, -1.0)
    oriented = center + direction * (test - center)
    return SelectionProfile(
        selected_index=selected.astype(np.int32),
        train_value=np.asarray(train, dtype=np.float32),
        test_value=np.asarray(test, dtype=np.float32),
        test_oriented_value=np.asarray(oriented, dtype=np.float32),
        test_strength=np.asarray(np.abs(test - center), dtype=np.float32),
    )


def concept_centric_profile(
    train_values: np.ndarray,
    test_values: np.ndarray,
    *,
    center: float,
    candidate_features: Sequence[int] | None = None,
) -> SelectionProfile:
    """Select one feature per target on train and freeze it on test."""
    train_values = _as_2d(train_values, "train_values")
    test_values = _as_2d(test_values, "test_values")
    if train_values.shape != test_values.shape:
        raise ValueError("train and test association matrices must have equal shape")
    if candidate_features is None:
        candidates = np.arange(train_values.shape[0], dtype=np.int64)
    else:
        candidates = np.asarray(candidate_features, dtype=np.int64)
        if candidates.ndim != 1 or len(candidates) == 0:
            raise ValueError("candidate_features must be a nonempty vector")
        if np.any(candidates < 0) or np.any(candidates >= train_values.shape[0]):
            raise ValueError("candidate feature outside association matrix")
    local = np.argmax(np.abs(train_values[candidates] - center), axis=0)
    selected = candidates[local]
    columns = np.arange(train_values.shape[1])
    train = train_values[selected, columns]
    test = test_values[selected, columns]
    direction = np.where(train >= center, 1.0, -1.0)
    oriented = center + direction * (test - center)
    return SelectionProfile(
        selected_index=selected.astype(np.int32),
        train_value=np.asarray(train, dtype=np.float32),
        test_value=np.asarray(test, dtype=np.float32),
        test_oriented_value=np.asarray(oriented, dtype=np.float32),
        test_strength=np.asarray(np.abs(test - center), dtype=np.float32),
    )


def matched_feature_subsets(
    n_features: int,
    budget: int,
    replicates: int,
    seed_base: int,
) -> list[np.ndarray]:
    """Generate deterministic, independently sampled matched-budget subsets."""
    if not 0 < budget <= n_features:
        raise ValueError("budget must be in 1..n_features")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    return [
        np.sort(
            np.random.default_rng(seed_base + replicate).choice(
                n_features, size=budget, replace=False
            )
        ).astype(np.int32)
        for replicate in range(replicates)
    ]
