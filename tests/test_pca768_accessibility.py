from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def test_full_rank_pca_is_orthonormal_and_invertible() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(80, 12)).astype(np.float32)
    pca = PCA(n_components=12, svd_solver="full").fit(values)
    transformed = pca.transform(values)
    reconstructed = pca.inverse_transform(transformed)
    np.testing.assert_allclose(reconstructed, values, atol=2e-5)
    np.testing.assert_allclose(pca.components_ @ pca.components_.T, np.eye(12), atol=2e-5)


def test_manifest_groups_cover_thirty_model_depth_cells() -> None:
    namespace = runpy.run_path("scripts/run_pca768_accessibility_worker.py")
    root = Path(__file__).resolve().parents[1]
    groups = namespace["pca_groups"](
        root / "results/multiscale_sae_v1/training_manifest.csv"
    )
    assert len(groups) == 30
    assert all(len(group) == 3 for group in groups)


def test_train_normalization_has_zero_mean_and_unit_scale() -> None:
    namespace = runpy.run_path("scripts/run_pca768_accessibility_worker.py")
    values = np.asarray([[1.0, 5.0], [3.0, 7.0], [5.0, 9.0]], dtype=np.float32)
    normalized, mean, scale = namespace["normalization_from_train"](
        values, np.asarray([0, 1, 2])
    )
    np.testing.assert_allclose(mean, [3.0, 7.0])
    np.testing.assert_allclose(normalized.mean(0), 0.0, atol=1e-6)
    np.testing.assert_allclose(normalized.std(0), 1.0, atol=1e-6)
    assert np.all(scale > 0)
