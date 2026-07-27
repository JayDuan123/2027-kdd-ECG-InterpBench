"""Statistics for paired multi-scale SAE and fixed-dense comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapEstimate:
    mean: float
    lower: float
    upper: float


def clustered_mean_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    *,
    replicates: int = 20_000,
    seed: int = 20_260_714,
) -> BootstrapEstimate:
    """Bootstrap a mean by resampling complete, equally weighted clusters."""
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    if values.ndim != 1 or clusters.ndim != 1 or values.shape != clusters.shape:
        raise ValueError("values and clusters must be equal-length vectors")
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("values must be nonempty and finite")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    unique = np.unique(clusters)
    if unique.size < 2:
        raise ValueError("at least two clusters are required")
    cluster_means = np.asarray(
        [values[clusters == cluster].mean() for cluster in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(replicates, len(unique)))
    bootstrap_means = cluster_means[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return BootstrapEstimate(
        mean=float(cluster_means.mean()),
        lower=float(lower),
        upper=float(upper),
    )


def validate_complete_factorial(
    observed: set[tuple[object, ...]],
    levels: tuple[tuple[object, ...], ...],
) -> None:
    """Require exactly one observation for every requested factorial cell."""
    from itertools import product

    expected = set(product(*levels))
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            f"factorial grid mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def threshold_coverage(values: np.ndarray, threshold: float = 0.20) -> float:
    """Fraction of absolute association scores meeting an inclusive threshold."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty vector")
    return float(np.mean(np.abs(values) >= threshold))
