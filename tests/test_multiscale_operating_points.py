from __future__ import annotations

import pandas as pd

from scripts.select_multiscale_sae_operating_points import with_validated_sparsity_arm


def expect_runtime_error(callback) -> None:
    try:
        callback()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError")


def test_missing_surface_arm_is_recovered_from_primary_manifest() -> None:
    surface = pd.DataFrame({"model": ["ECG-FM", "ECG-JEPA"]})
    manifest = pd.DataFrame({"sparsity_arm": ["fixed_k_over_d"] * 2})
    result = with_validated_sparsity_arm(surface, manifest)
    assert set(result.sparsity_arm) == {"fixed_k_over_d"}
    assert "sparsity_arm" not in surface


def test_existing_surface_arm_is_validated() -> None:
    surface = pd.DataFrame({"sparsity_arm": ["fixed_k_over_d"] * 2})
    manifest = pd.DataFrame({"sparsity_arm": ["fixed_k_over_d"] * 2})
    result = with_validated_sparsity_arm(surface, manifest)
    assert set(result.sparsity_arm) == {"fixed_k_over_d"}


def test_conflicting_manifest_arms_are_rejected() -> None:
    surface = pd.DataFrame({"model": ["ECG-FM"]})
    manifest = pd.DataFrame(
        {"sparsity_arm": ["fixed_k_over_d", "fixed_k_over_n"]}
    )
    expect_runtime_error(lambda: with_validated_sparsity_arm(surface, manifest))


def test_surface_manifest_disagreement_is_rejected() -> None:
    surface = pd.DataFrame({"sparsity_arm": ["fixed_k_over_n"]})
    manifest = pd.DataFrame({"sparsity_arm": ["fixed_k_over_d"]})
    expect_runtime_error(lambda: with_validated_sparsity_arm(surface, manifest))
