import unittest

import numpy as np

from scripts.summarize_final_layer_feature_yield import (
    feature_yield_row,
    pearson_pvalues,
)


class FinalLayerFeatureYieldTest(unittest.TestCase):
    def test_pearson_pvalues_are_two_sided_and_monotone(self) -> None:
        values = pearson_pvalues(np.asarray([0.0, 0.2, -0.5]), 1000)
        self.assertAlmostEqual(values[0], 1.0)
        self.assertLess(values[2], values[1])
        self.assertLess(values[1], values[0])

    def test_feature_yield_requires_direction_threshold_fdr_and_live(self) -> None:
        row = feature_yield_row(
            model="synthetic",
            method="sae",
            replicate_kind="seed",
            replicate=0,
            scores=np.asarray([0.50, 0.30, 0.01, -0.40, 0.60]),
            selected_targets=np.asarray([0, 1, 2, 3, 3]),
            live=np.asarray([True, True, True, True, False]),
            n_targets=4,
            n_test=1000,
            correlation_threshold=0.20,
            fdr_threshold=0.05,
        )
        self.assertEqual(row["qualified_feature_count"], 2)
        self.assertEqual(row["covered_target_count"], 2)
        self.assertAlmostEqual(row["target_coverage"], 0.5)
        self.assertAlmostEqual(row["qualified_per_1000_candidates"], 400.0)


if __name__ == "__main__":
    unittest.main()
