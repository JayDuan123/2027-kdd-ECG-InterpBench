import unittest

import numpy as np

from benchmark_v1.accessibility_calibration import feature_concept_correlations
from benchmark_v1.matched_effect import (
    aggregate_patient_means,
    bootstrap_patient_metric_means,
    calibrated_record_metrics,
    centroid_unit_intervention,
    common_validation_effect,
    fit_multitarget_ridge_readout,
)
from benchmark_v1.sparse_accessibility import candidate_ranking


class MatchedEffectTest(unittest.TestCase):
    def test_ridge_selects_validation_alpha_without_test_selection(self) -> None:
        rng = np.random.default_rng(12)
        x_train = rng.normal(size=(320, 10))
        x_validation = rng.normal(size=(160, 10))
        x_test = rng.normal(size=(160, 10))

        def target(x):
            return np.column_stack((1.5 * x[:, 2] - x[:, 7], 0.8 * x[:, 4]))

        y_train = target(x_train) + rng.normal(scale=0.08, size=(320, 2))
        y_validation = target(x_validation) + rng.normal(scale=0.08, size=(160, 2))
        y_test = target(x_test) + rng.normal(scale=0.08, size=(160, 2))
        result = fit_multitarget_ridge_readout(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            y_test,
            alphas=(0.1, 1.0, 10.0),
        )
        self.assertEqual(result.coefficients.shape, (10, 2))
        self.assertTrue(np.all(np.abs(result.test_correlations) > 0.99))
        self.assertTrue(set(result.selected_alphas).issubset({0.1, 1.0, 10.0}))

    def test_centroid_intervention_and_effect_calibration(self) -> None:
        rng = np.random.default_rng(4)
        train = rng.normal(size=(300, 4))
        validation = rng.normal(size=(120, 4))
        test = rng.normal(size=(100, 4))
        y_train = np.column_stack((train[:, 0], train[:, 2]))
        correlations = feature_concept_correlations(train, y_train)
        ranking = candidate_ranking(correlations)
        coefficients = np.eye(4, 2)
        intervention = centroid_unit_intervention(
            train,
            validation,
            test,
            y_train,
            ranking,
            np.eye(4),
            coefficients,
            target_index=0,
            k=1,
        )
        self.assertEqual(intervention.selected_features.tolist(), [0])
        self.assertGreater(intervention.validation_target_effect(0), 0.5)
        common, effects, status = common_validation_effect(
            {"dense": intervention, "sae": intervention}, 0, cap=0.25, floor=0.05
        )
        self.assertEqual(status, "eligible")
        self.assertAlmostEqual(common, 0.25)
        metrics, alpha, status = calibrated_record_metrics(
            intervention,
            target_index=0,
            target_family="rate",
            concept_families=("rate", "morphology"),
            target_effect=common,
            unit_validation_effect=effects["dense"],
        )
        self.assertEqual(status, "eligible")
        self.assertLessEqual(alpha, 1.0)
        self.assertEqual(metrics.shape, (100, 5))
        self.assertTrue(np.allclose(metrics[:, 1:4], 0.0))

    def test_centroid_intervention_accepts_sparse_codes(self) -> None:
        from scipy import sparse

        rng = np.random.default_rng(14)
        dense = np.maximum(rng.normal(size=(180, 6)), 0.0)
        train = sparse.csr_matrix(dense)
        validation = sparse.csr_matrix(dense[:80])
        test = sparse.csr_matrix(dense[80:])
        y_train = np.column_stack((dense[:, 1], dense[:, 4]))
        ranking = candidate_ranking(feature_concept_correlations(train, y_train))
        intervention = centroid_unit_intervention(
            train,
            validation,
            test,
            y_train,
            ranking,
            np.eye(6),
            np.eye(6, 2),
            target_index=0,
            k=1,
        )
        self.assertEqual(intervention.validation_outputs.shape, (80, 2))
        self.assertEqual(intervention.test_outputs.shape, (100, 2))

    def test_centroid_intervention_ignores_missing_target_labels(self) -> None:
        rng = np.random.default_rng(21)
        train = rng.normal(size=(120, 4))
        validation = rng.normal(size=(40, 4))
        test = rng.normal(size=(50, 4))
        y_train = np.column_stack((train[:, 0], train[:, 2]))
        y_train[::3, 0] = np.nan
        finite = np.isfinite(y_train[:, 0])
        correlations = np.zeros((4, 2))
        correlations[:, 0] = feature_concept_correlations(
            train[finite], y_train[finite, :1]
        )[:, 0]
        correlations[:, 1] = feature_concept_correlations(train, y_train[:, 1:2])[:, 0]
        intervention = centroid_unit_intervention(
            train,
            validation,
            test,
            y_train,
            candidate_ranking(correlations),
            np.eye(4),
            np.eye(4, 2),
            target_index=0,
            k=1,
        )
        self.assertTrue(np.isfinite(intervention.centroid).all())
        self.assertEqual(intervention.test_outputs.shape, (50, 2))

    def test_nonpositive_effect_is_ineligible(self) -> None:
        rng = np.random.default_rng(9)
        values = rng.normal(size=(20, 3)).astype(np.float32)
        intervention = type("Intervention", (), {})()
        intervention.validation_outputs = values
        intervention.test_outputs = values
        intervention.validation_l2 = np.ones(20, dtype=np.float32)
        intervention.test_l2 = np.ones(20, dtype=np.float32)
        intervention.validation_target_effect = lambda target: -0.1
        common, _, status = common_validation_effect({"sae": intervention}, 0)
        self.assertTrue(np.isnan(common))
        self.assertEqual(status, "nonpositive_validation_effect")

    def test_patient_bootstrap_preserves_record_weighting(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        patients, means, counts = aggregate_patient_means(values, ["a", "a", "b"])
        self.assertEqual(patients.tolist(), ["a", "b"])
        self.assertTrue(np.allclose(means, [[2.0, 3.0], [10.0, 20.0]]))
        weights = np.asarray([[1, 1], [2, 0]], dtype=float)
        boot = bootstrap_patient_metric_means(means, counts, weights)
        self.assertTrue(np.allclose(boot[0], values.mean(axis=0)))
        self.assertTrue(np.allclose(boot[1], [2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
