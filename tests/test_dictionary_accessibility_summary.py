from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.summarize_dictionary_accessibility import summarize_depth


def test_depth_summary_averages_dictionary_replicates_before_comparison() -> None:
    feature = pd.DataFrame(
        [
            {
                "model": "M",
                "relative_depth": 0.0,
                "target_type": "waveform",
                "metric": "pearson_r",
                "method": "sae_matched_768",
                "dictionary_width": 6144,
                "candidate_budget": 768,
                "replicate": replicate,
                "n_live": 700,
                "median_test_oriented_score": value,
                "q95_test_oriented_score": value + 0.1,
                "max_test_oriented_score": value + 0.2,
                "n_above_primary": 100 + replicate,
                "fraction_above_primary": value,
                "live_fraction_above_primary": value + 0.01,
                "primary_threshold": 0.2,
            }
            for replicate, value in enumerate((0.2, 0.4))
        ]
    )
    target = pd.DataFrame(
        [
            {
                "model": "M",
                "relative_depth": 0.0,
                "target_type": "waveform",
                "metric": "pearson_r",
                "method": "sae_matched_768",
                "target": target_name,
                "test_oriented_score": value,
                "covered_primary": int(value >= 0.2),
            }
            for target_name, value in (("a", 0.1), ("a", 0.3), ("b", 0.5), ("b", 0.7))
        ]
    )
    result = summarize_depth(feature, target)
    assert len(result) == 1
    np.testing.assert_allclose(result.loc[0, "mean_feature_median"], 0.3)
    np.testing.assert_allclose(result.loc[0, "mean_high_feature_count"], 100.5)
    np.testing.assert_allclose(result.loc[0, "mean_best_target_score"], 0.4)
    np.testing.assert_allclose(result.loc[0, "target_coverage"], 0.75)
    assert result.loc[0, "targets"] == 2
