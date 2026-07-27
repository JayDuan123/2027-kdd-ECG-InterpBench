from __future__ import annotations

import numpy as np

from benchmark_v1.multiscale_sae import (
    correlation_from_sufficient_statistics,
    relative_layer_indices,
    selected_concept_metrics,
    sparsity_for,
)


def test_relative_layer_indices_use_standardized_depths() -> None:
    assert relative_layer_indices(6) == [0, 1, 3, 4, 5]
    assert relative_layer_indices(12) == [0, 3, 6, 8, 11]


def test_sparsity_arms_are_factored() -> None:
    assert sparsity_for("fixed_k_over_d", 1, 768) == (768, 96)
    assert sparsity_for("fixed_k_over_d", 32, 768) == (24576, 96)
    assert sparsity_for("fixed_k_over_n", 1, 768) == (768, 12)
    assert sparsity_for("fixed_k_over_n", 32, 768) == (24576, 384)


def test_streaming_correlation_matches_numpy() -> None:
    z = np.asarray([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, -1.0]], dtype=np.float64)
    y = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    actual = correlation_from_sufficient_statistics(
        len(z),
        z.sum(axis=0),
        np.square(z).sum(axis=0),
        y.sum(axis=0),
        np.square(y).sum(axis=0),
        z.T @ y,
    )
    expected = np.asarray([np.corrcoef(z[:, index], y[:, 0])[0, 1] for index in range(2)])
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-6)


def test_concept_feature_selection_is_train_only() -> None:
    train = np.asarray([[0.9], [0.2]], dtype=np.float32)
    evaluation = np.asarray([[0.4], [0.99]], dtype=np.float32)
    rows, summary = selected_concept_metrics(train, evaluation, ["qrs_duration"])
    assert rows[0]["selected_feature"] == 0
    assert rows[0]["eval_correlation"] == float(evaluation[0, 0])
    assert summary["mean_train_selected_abs_correlation"] == float(evaluation[0, 0])
