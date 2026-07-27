import unittest

import numpy as np

from benchmark_v1.accessibility_calibration import feature_concept_correlations
from benchmark_v1.sparse_accessibility import (
    PatientClusterBootstrap,
    candidate_ranking,
    deterministic_feature_subset,
    deterministic_subset_from_candidates,
    fit_sparse_ridge_curve,
    normalized_log2_curve_auc,
)


class SparseAccessibilityTest(unittest.TestCase):
    def test_deterministic_subset_is_unique_and_target_independent(self) -> None:
        first = deterministic_feature_subset(100, 20, 17)
        second = deterministic_feature_subset(100, 20, 17)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(len(np.unique(first)), 20)
        self.assertTrue(np.all(np.diff(first) > 0))

    def test_candidate_ranking_respects_budget(self) -> None:
        correlations = np.asarray(
            [[0.1, 0.9], [0.8, 0.2], [0.7, 0.6], [0.3, 0.4]], dtype=float
        )
        ranking = candidate_ranking(correlations, [0, 2, 3])
        self.assertEqual(ranking[:, 0].tolist(), [2, 3, 0])
        self.assertEqual(ranking[:, 1].tolist(), [0, 2, 3])

    def test_deterministic_subset_respects_explicit_live_pool(self) -> None:
        live = np.asarray([1, 4, 8, 11, 15, 19])
        first = deterministic_subset_from_candidates(live, 4, 71)
        second = deterministic_subset_from_candidates(live, 4, 71)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(set(first).issubset(set(live)))
        with self.assertRaises(ValueError):
            deterministic_subset_from_candidates(live, 7, 71)

    def test_sparse_curve_uses_train_ranking_and_validation_alpha(self) -> None:
        rng = np.random.default_rng(42)
        x_train = rng.normal(size=(240, 8))
        x_validation = rng.normal(size=(120, 8))
        x_test = rng.normal(size=(120, 8))

        def targets(values: np.ndarray) -> np.ndarray:
            return np.column_stack(
                [
                    1.8 * values[:, 1] + 0.2 * values[:, 3],
                    -1.5 * values[:, 6] + 0.3 * values[:, 2],
                ]
            )

        y_train = targets(x_train) + rng.normal(scale=0.05, size=(240, 2))
        y_validation = targets(x_validation) + rng.normal(scale=0.05, size=(120, 2))
        y_test = targets(x_test) + rng.normal(scale=0.05, size=(120, 2))
        correlations = feature_concept_correlations(x_train, y_train)
        ranking = candidate_ranking(correlations)
        result = fit_sparse_ridge_curve(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            ranking,
            ks=(1, 2, 4),
            alphas=(0.1, 1.0, 10.0),
        )
        self.assertEqual(result.test_predictions.shape, (3, 120, 2))
        self.assertTrue(np.all(np.abs(result.test_correlations[-1]) > 0.98))
        self.assertTrue(set(np.unique(result.selected_alphas)).issubset({0.1, 1.0, 10.0}))
        self.assertEqual(result.selected_features[0, 0], (1,))
        self.assertEqual(result.selected_features[0, 1], (6,))

    def test_patient_cluster_bootstrap_preserves_perfect_predictions(self) -> None:
        targets = np.arange(24, dtype=float).reshape(12, 2)
        patients = [f"p{index // 2}" for index in range(12)]
        bootstrap = PatientClusterBootstrap(patients, targets, draws=25, seed=5)
        correlations = bootstrap.correlations(targets)
        self.assertEqual(correlations.shape, (25, 2))
        self.assertTrue(np.allclose(correlations, 1.0))

    def test_log_curve_auc(self) -> None:
        self.assertAlmostEqual(
            normalized_log2_curve_auc([1, 2, 4], [0.2, 0.4, 0.6]), 0.4
        )


if __name__ == "__main__":
    unittest.main()
