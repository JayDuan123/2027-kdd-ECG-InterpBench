from __future__ import annotations

import numpy as np

from scripts.run_multiscale_sae_patient_bootstrap import (
    aggregate_by_patient,
    bootstrap,
    bootstrap_design_hash,
    load_bootstrap_checkpoint,
)
from scripts.summarize_multiscale_sae_patient_bootstrap import log_auc_weights


def synthetic_stats() -> dict[str, np.ndarray]:
    inverse = np.asarray([0, 0, 1, 2, 2], dtype=np.int64)
    z = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    y = np.asarray([[0, 1], [1, 2], [2, 2], [3, 4], [4, 4]], dtype=float)
    return aggregate_by_patient(
        inverse,
        np.arange(5, dtype=float) + 1,
        np.arange(5, dtype=float) + 2,
        z,
        y,
    )


def test_patient_bootstrap_preserves_clustered_records() -> None:
    stats = synthetic_stats()
    np.testing.assert_array_equal(stats["count"], [2, 1, 2])
    observed, distributions = bootstrap(stats, 25, 7, 8, "cpu")
    assert distributions["recon_R2"].shape == (25,)
    assert distributions["concept_correlation"].shape == (25, 2)
    assert np.isfinite(observed["semantic_alignment"])


def test_patient_bootstrap_draws_are_reproducible() -> None:
    stats = synthetic_stats()
    _, first = bootstrap(stats, 17, 29, 5, "cpu")
    _, second = bootstrap(stats, 17, 29, 7, "cpu")
    for key in first:
        np.testing.assert_allclose(first[key], second[key], atol=0, rtol=0)


def test_common_scale_auc_weights_are_normalized() -> None:
    weights = log_auc_weights()
    assert tuple(weights) == (1, 4, 8, 16, 32)
    assert abs(sum(weights.values()) - 1.0) < 1e-12
    assert all(value > 0 for value in weights.values())


def test_bootstrap_design_hash_binds_patient_clusters_and_draws() -> None:
    reference = bootstrap_design_hash("a" * 64, 2862, 2000, 20260714)
    assert reference == bootstrap_design_hash("a" * 64, 2862, 2000, 20260714)
    assert reference != bootstrap_design_hash("b" * 64, 2862, 2000, 20260714)
    assert reference != bootstrap_design_hash("a" * 64, 2862, 2001, 20260714)


def test_patient_bootstrap_resumes_exactly_from_atomic_checkpoint(tmp_path) -> None:
    stats = synthetic_stats()
    checkpoint = tmp_path / "bootstrap.progress.npz"
    identity = {"task_index": 17, "config_hash": "abc"}

    def interrupt_after_two_chunks(completed: int) -> None:
        if completed >= 10:
            raise RuntimeError("simulated interruption")

    try:
        bootstrap(
            stats,
            23,
            29,
            5,
            "cpu",
            checkpoint_path=checkpoint,
            checkpoint_identity=identity,
            after_checkpoint=interrupt_after_two_chunks,
        )
    except RuntimeError as error:
        assert str(error) == "simulated interruption"
    else:
        raise AssertionError("expected simulated interruption")

    completed, _, partial = load_bootstrap_checkpoint(
        checkpoint,
        {
            "bootstrap_samples": 23,
            "bootstrap_seed": 29,
            "n_patients": 3,
            "identity": identity,
        },
    )
    assert completed == 10
    assert all(len(parts[0]) == 10 for parts in partial.values())

    observed, resumed = bootstrap(
        stats,
        23,
        29,
        7,
        "cpu",
        checkpoint_path=checkpoint,
        checkpoint_identity=identity,
    )
    expected_observed, expected = bootstrap(stats, 23, 29, 4, "cpu")
    for key in expected:
        np.testing.assert_allclose(resumed[key], expected[key], atol=0, rtol=0)
    for key in expected_observed:
        np.testing.assert_allclose(
            observed[key], expected_observed[key], atol=0, rtol=0
        )


def test_patient_bootstrap_rejects_stale_checkpoint_identity(tmp_path) -> None:
    stats = synthetic_stats()
    checkpoint = tmp_path / "bootstrap.progress.npz"
    bootstrap(
        stats,
        5,
        29,
        5,
        "cpu",
        checkpoint_path=checkpoint,
        checkpoint_identity={"config_hash": "first"},
    )
    try:
        bootstrap(
            stats,
            5,
            29,
            5,
            "cpu",
            checkpoint_path=checkpoint,
            checkpoint_identity={"config_hash": "second"},
        )
    except RuntimeError as error:
        assert "identity mismatch" in str(error)
    else:
        raise AssertionError("expected stale checkpoint rejection")
