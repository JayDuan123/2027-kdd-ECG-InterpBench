from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

from benchmark_v1.dictionary_accessibility import (
    concept_centric_profile,
    feature_centric_profile,
    matched_feature_subsets,
    nonconstant_feature_mask,
    tie_aware_auc_matrix,
)


def test_tie_aware_auc_matches_sklearn_for_dense_and_sparse() -> None:
    features = np.asarray(
        [
            [0.0, -2.0, 1.0],
            [0.0, -1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [3.0, 4.0, 2.0],
            [3.0, 4.0, 2.0],
        ]
    )
    labels = np.asarray(
        [[0, 1], [0, 0], [1, 1], [0, 0], [1, 1], [1, 0]], dtype=float
    )
    expected = np.asarray(
        [
            [roc_auc_score(labels[:, task], features[:, feature]) for task in range(2)]
            for feature in range(3)
        ]
    )
    np.testing.assert_allclose(tie_aware_auc_matrix(features, labels), expected)
    np.testing.assert_allclose(
        tie_aware_auc_matrix(sparse.csr_matrix(features), labels), expected
    )


def test_feature_centric_selection_freezes_train_target_and_direction() -> None:
    train = np.asarray([[0.8, -0.9], [0.4, 0.1]])
    test = np.asarray([[0.7, 0.6], [-0.5, 0.2]])
    profile = feature_centric_profile(train, test, center=0.0)
    np.testing.assert_array_equal(profile.selected_index, [1, 0])
    np.testing.assert_allclose(profile.test_value, [0.6, -0.5])
    np.testing.assert_allclose(profile.test_oriented_value, [-0.6, -0.5])
    np.testing.assert_allclose(profile.test_strength, [0.6, 0.5])


def test_auc_selection_uses_train_orientation_on_test() -> None:
    train = np.asarray([[0.2], [0.7], [0.9]])
    test = np.asarray([[0.8], [0.6], [0.4]])
    profile = concept_centric_profile(train, test, center=0.5)
    assert profile.selected_index[0] == 2
    np.testing.assert_allclose(profile.test_oriented_value, [0.4])


def test_concept_centric_selection_respects_candidate_budget() -> None:
    train = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, -0.95]])
    test = np.asarray([[0.8, 0.2], [0.1, 0.7], [0.6, -0.9]])
    full = concept_centric_profile(train, test, center=0.0)
    matched = concept_centric_profile(
        train, test, center=0.0, candidate_features=[0, 1]
    )
    np.testing.assert_array_equal(full.selected_index, [0, 2])
    np.testing.assert_array_equal(matched.selected_index, [0, 1])


def test_nonconstant_mask_matches_dense_and_sparse() -> None:
    values = np.asarray([[0.0, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 3.0, 0.0]])
    expected = np.asarray([False, True, True])
    np.testing.assert_array_equal(nonconstant_feature_mask(values), expected)
    np.testing.assert_array_equal(
        nonconstant_feature_mask(sparse.csr_matrix(values)), expected
    )


def test_matched_subsets_are_deterministic_and_unique() -> None:
    first = matched_feature_subsets(12, 5, 3, 100)
    second = matched_feature_subsets(12, 5, 3, 100)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)
        assert len(np.unique(left)) == 5
    assert not np.array_equal(first[0], first[1])
