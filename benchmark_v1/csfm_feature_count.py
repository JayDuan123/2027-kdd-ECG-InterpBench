"""Utilities for auditing CSFM L6 concept-associated feature counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmark_v1.dictionary_accessibility import (
    feature_centric_profile,
    tie_aware_auc_matrix,
)


@dataclass(frozen=True)
class FrozenFeatureResult:
    """Per-feature train selection and frozen test evaluation."""

    selected_target: np.ndarray
    train_auc: np.ndarray
    test_auc: np.ndarray
    train_oriented_auc: np.ndarray
    test_oriented_auc: np.ndarray
    direction: np.ndarray


def validate_original_scores(
    scores: np.ndarray,
    *,
    expected_width: int,
    expected_count: int,
    threshold: float,
) -> np.ndarray:
    """Validate an authoritative original score vector and its published count."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (expected_width,):
        raise ValueError(
            f"expected score vector width {expected_width}, got {values.shape}"
        )
    count = int(np.count_nonzero(values > threshold))
    if count != expected_count:
        raise ValueError(f"expected original count {expected_count}, got {count}")
    return values


def frozen_feature_profile(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> FrozenFeatureResult:
    """Select target and sign on train, then evaluate them unchanged on test."""
    train_auc = tie_aware_auc_matrix(train_features, train_labels)
    test_auc = tie_aware_auc_matrix(test_features, test_labels)
    profile = feature_centric_profile(train_auc, test_auc, center=0.5)
    train_oriented = 0.5 + np.abs(profile.train_value - 0.5)
    direction = np.where(profile.train_value >= 0.5, 1, -1).astype(np.int8)
    return FrozenFeatureResult(
        selected_target=profile.selected_index,
        train_auc=profile.train_value,
        test_auc=profile.test_value,
        train_oriented_auc=np.asarray(train_oriented, dtype=np.float32),
        test_oriented_auc=profile.test_oriented_value,
        direction=direction,
    )


def feature_count_summary(scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    """Return raw count and width-normalized fraction using a strict threshold."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a nonempty vector")
    count = int(np.count_nonzero(values > threshold))
    return {
        "width": int(values.size),
        "associated_count": count,
        "associated_fraction": count / int(values.size),
        "max_score": float(np.max(values)),
    }
