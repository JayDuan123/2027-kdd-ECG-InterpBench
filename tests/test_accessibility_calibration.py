from __future__ import annotations

import numpy as np
from scipy import sparse

from benchmark_v1.accessibility_calibration import (
    canonical_single_atom_features,
    columnwise_pearson,
    feature_concept_correlations,
    fit_ridge_predictions,
    ranked_feature_indices,
    safe_ratio,
    selected_coordinate_predictions,
)


def test_columnwise_pearson_handles_sign_and_constant_scores() -> None:
    targets = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    scores = np.asarray([[0.0, 1.0], [-1.0, 1.0], [-2.0, 1.0], [-3.0, 1.0]])
    np.testing.assert_allclose(columnwise_pearson(targets, scores), [-1.0, 0.0])


def test_sparse_feature_concept_correlations_match_numpy() -> None:
    features = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0], [3.0, 0.0, 1.0]]
    )
    concepts = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    dense = feature_concept_correlations(features, concepts)
    csr = feature_concept_correlations(sparse.csr_matrix(features), concepts)
    np.testing.assert_allclose(csr, dense, atol=1e-7)


def test_ranking_is_train_only_and_stable_on_ties() -> None:
    correlations = np.asarray([[0.8, 0.1], [-0.8, 0.5], [0.2, -0.7]])
    ranking = ranked_feature_indices(correlations)
    np.testing.assert_array_equal(ranking[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(ranking[:, 1], [2, 1, 0])


def test_canonical_single_atom_preserves_original_train_selection() -> None:
    ranking = np.asarray([[0, 2], [1, 1], [2, 0]])
    selected, mismatch = canonical_single_atom_features(ranking, [1, 2])
    np.testing.assert_array_equal(selected, [1, 2])
    assert mismatch == 1


def test_selected_coordinate_predictions_match_dense_and_sparse() -> None:
    features = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    expected = np.asarray([[3.0, 1.0], [6.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        selected_coordinate_predictions(features, [2, 0]), expected
    )
    np.testing.assert_array_equal(
        selected_coordinate_predictions(sparse.csr_matrix(features), [2, 0]),
        expected,
    )


def test_fixed_ridge_predictions_recover_linear_targets() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(120, 5))
    y = (2.0 * x[:, 0] - x[:, 1])[:, None]
    validation, test = fit_ridge_predictions(x[:80], y[:80], x[80:100], x[100:], 0.01)
    assert columnwise_pearson(y[80:100], validation)[0] > 0.999
    assert columnwise_pearson(y[100:], test)[0] > 0.999


def test_safe_ratio_preserves_undefined_denominators() -> None:
    ratio = safe_ratio([0.5, 0.2], [1.0, 0.0])
    assert ratio[0] == 0.5
    assert np.isnan(ratio[1])
