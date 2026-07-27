from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.summarize_accessibility_baselines_v2 import (
    paired_summary,
    random_seed_intervals,
)


def test_paired_summary_averages_sae_seeds_before_dense_comparison() -> None:
    rows = []
    for seed, value in [(1, 0.3), (2, 0.5), (3, 0.4)]:
        rows.append(
            {
                "model": "M",
                "relative_depth": 0.0,
                "seed": seed,
                "concept": "c",
                "family": "f",
                "method": "sae_single",
                "test_abs_r": value,
            }
        )
    v1 = pd.DataFrame(rows)
    dense = pd.DataFrame(
        [
            {
                "model": "M",
                "relative_depth": 0.0,
                "concept": "c",
                "family": "f",
                "test_abs_r": 0.25,
                "covered_020": 1,
            }
        ]
    )
    random = pd.DataFrame(
        [
            {
                "model": "M",
                "relative_depth": 0.0,
                "concept": "c",
                "family": "f",
                "test_abs_r": value,
                "covered_020": int(value >= 0.20),
            }
            for value in (0.1, 0.3)
        ]
    )
    paired, model = paired_summary(v1, dense, random)
    np.testing.assert_allclose(paired.sae_single_abs_r, [0.4])
    np.testing.assert_allclose(paired.sae_minus_dense_single, [0.15])
    np.testing.assert_allclose(paired.sae_minus_random_single, [0.2])
    np.testing.assert_allclose(paired.sae_coverage_probability, [1.0])
    assert model.loc[0, "positive_sae_minus_dense_depths"] == 1


def test_random_seed_intervals_use_replicate_level_model_estimates() -> None:
    frame = pd.DataFrame(
        [
            {
                "model": "M",
                "random_replicate": replicate,
                "test_abs_r": value,
                "covered_020": int(value >= 0.20),
            }
            for replicate, value in enumerate((0.1, 0.2, 0.3, 0.4))
        ]
    )
    result = random_seed_intervals(frame)
    assert result.loc[0, "random_replicates"] == 4
    np.testing.assert_allclose(result.loc[0, "mean_test_abs_r"], 0.25)
    assert result.loc[0, "mean_test_abs_r_q025"] < 0.2
    assert result.loc[0, "mean_test_abs_r_q975"] > 0.3
