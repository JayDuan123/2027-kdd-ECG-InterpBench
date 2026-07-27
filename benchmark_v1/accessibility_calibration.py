"""Shared metrics for the E=8 clinical-accessibility calibration ladder."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from benchmark_v1.multiscale_sae import correlation_from_sufficient_statistics


def columnwise_pearson(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """Return one finite Pearson correlation per output column."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_score.ndim == 1:
        y_score = y_score[:, None]
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch: targets={y_true.shape}, scores={y_score.shape}")
    true_centered = y_true - y_true.mean(axis=0, keepdims=True)
    score_centered = y_score - y_score.mean(axis=0, keepdims=True)
    numerator = np.sum(true_centered * score_centered, axis=0)
    denominator = np.sqrt(
        np.sum(np.square(true_centered), axis=0)
        * np.sum(np.square(score_centered), axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def feature_concept_correlations(features, concepts: np.ndarray) -> np.ndarray:
    """Compute feature-by-concept correlations for dense or CSR features."""
    concepts = np.asarray(concepts, dtype=np.float64)
    if concepts.ndim != 2 or features.shape[0] != concepts.shape[0]:
        raise ValueError("feature and concept matrices must share a row dimension")
    try:
        from scipy import sparse

        is_sparse = sparse.issparse(features)
    except ImportError:
        is_sparse = False
    if is_sparse:
        matrix = features.astype(np.float64, copy=False)
        sum_z = np.asarray(matrix.sum(axis=0)).ravel()
        sum_z2 = np.asarray(matrix.multiply(matrix).sum(axis=0)).ravel()
        cross = np.asarray(matrix.T @ concepts)
    else:
        matrix = np.asarray(features, dtype=np.float64)
        sum_z = matrix.sum(axis=0)
        sum_z2 = np.square(matrix).sum(axis=0)
        cross = matrix.T @ concepts
    return correlation_from_sufficient_statistics(
        len(concepts),
        sum_z,
        sum_z2,
        concepts.sum(axis=0),
        np.square(concepts).sum(axis=0),
        cross,
    )


def ranked_feature_indices(correlations: np.ndarray) -> np.ndarray:
    """Rank every feature per concept by decreasing absolute train correlation."""
    correlations = np.asarray(correlations)
    if correlations.ndim != 2:
        raise ValueError("correlations must be feature-by-concept")
    return np.argsort(-np.abs(correlations), axis=0, kind="stable")


def canonical_single_atom_features(
    ranking: np.ndarray, canonical_features: Sequence[int]
) -> tuple[np.ndarray, int]:
    """Use the original train-selected atoms and report numerical tie changes."""
    ranking = np.asarray(ranking)
    canonical = np.asarray(canonical_features, dtype=np.int64)
    if ranking.ndim != 2 or canonical.shape != (ranking.shape[1],):
        raise ValueError("canonical features must provide one index per concept")
    if np.any(canonical < 0) or np.any(canonical >= ranking.shape[0]):
        raise ValueError("canonical feature index outside the SAE dictionary")
    return canonical, int(np.sum(ranking[0] != canonical))


def selected_coordinate_predictions(features, selected: Sequence[int]) -> np.ndarray:
    """Gather one independently selected coordinate for each concept."""
    selected = np.asarray(selected, dtype=np.int64)
    if selected.ndim != 1:
        raise ValueError("selected coordinates must be one-dimensional")
    if np.any(selected < 0) or np.any(selected >= features.shape[1]):
        raise ValueError("selected coordinate outside feature matrix")
    gathered = features[:, selected]
    try:
        from scipy import sparse

        if sparse.issparse(gathered):
            gathered = gathered.toarray()
    except ImportError:
        pass
    return np.asarray(gathered, dtype=np.float32)


def fit_ridge_predictions(
    x_train,
    y_train: np.ndarray,
    x_validation,
    x_test,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the fixed-alpha ridge protocol and return validation/test predictions."""
    from scipy import sparse
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    is_sparse = sparse.issparse(x_train)
    scaler = StandardScaler(with_mean=not is_sparse)
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)
    model = Ridge(
        alpha=float(alpha),
        fit_intercept=True,
        solver="lsqr",
        max_iter=1000,
        tol=1e-4,
    )
    model.fit(x_train_scaled, np.asarray(y_train, dtype=np.float64))
    return model.predict(x_validation_scaled), model.predict(x_test_scaled)


def safe_ratio(numerator: Sequence[float], denominator: Sequence[float]) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError("ratio operands must have equal shape")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-12,
    )
