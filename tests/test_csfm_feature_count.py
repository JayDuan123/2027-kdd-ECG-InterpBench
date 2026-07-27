from __future__ import annotations

import numpy as np

from benchmark_v1.csfm_feature_count import (
    feature_count_summary,
    frozen_feature_profile,
    validate_original_scores,
)


def test_summary_uses_strict_threshold_and_width_normalization() -> None:
    summary = feature_count_summary(np.asarray([0.70, 0.71, 0.90, 0.20]), 0.70)
    assert summary["width"] == 4
    assert summary["associated_count"] == 2
    assert summary["associated_fraction"] == 0.5


def test_original_validation_checks_width_and_count() -> None:
    scores = validate_original_scores(
        np.asarray([0.6, 0.8, 0.7]),
        expected_width=3,
        expected_count=1,
        threshold=0.7,
    )
    np.testing.assert_allclose(scores, [0.6, 0.8, 0.7])


def test_frozen_profile_does_not_refold_test_direction() -> None:
    train_x = np.asarray([[0.0], [0.1], [0.8], [1.0]])
    test_x = np.asarray([[0.9], [0.8], [0.1], [0.0]])
    labels = np.asarray([[0], [0], [1], [1]])
    result = frozen_feature_profile(train_x, labels, test_x, labels)
    assert result.selected_target[0] == 0
    assert result.direction[0] == 1
    assert result.train_oriented_auc[0] == 1.0
    assert result.test_oriented_auc[0] == 0.0
