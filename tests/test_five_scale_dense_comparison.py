from __future__ import annotations

import numpy as np

from benchmark_v1.five_scale_dense_comparison import (
    clustered_mean_bootstrap,
    threshold_coverage,
    validate_complete_factorial,
)


def test_cluster_bootstrap_equal_weights_clusters() -> None:
    values = np.asarray([0.0, 2.0, 10.0, 10.0])
    clusters = np.asarray(["a", "a", "b", "b"])
    result = clustered_mean_bootstrap(values, clusters, replicates=5000, seed=7)
    assert result.mean == 5.5
    assert result.lower <= result.mean <= result.upper


def test_factorial_validation_rejects_missing_cell() -> None:
    complete = {("a", 1), ("a", 2), ("b", 1), ("b", 2)}
    validate_complete_factorial(complete, (("a", "b"), (1, 2)))
    try:
        validate_complete_factorial(complete - {("b", 2)}, (("a", "b"), (1, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("missing factorial cell was not rejected")


def test_coverage_is_absolute_and_inclusive() -> None:
    assert threshold_coverage(np.asarray([-0.3, 0.2, 0.19]), 0.2) == 2 / 3
