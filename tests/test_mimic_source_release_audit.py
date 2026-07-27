from __future__ import annotations

from pathlib import Path

from scripts.audit_mimic_source_release import resolve_seed_checkpoint


def test_resolve_seed_checkpoint_accepts_cohort_specific_width(tmp_path: Path) -> None:
    checkpoint = tmp_path / "batchtopk_N4096_k64.pt"
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_seed_checkpoint(tmp_path) == checkpoint


def test_resolve_seed_checkpoint_accepts_standard_width(tmp_path: Path) -> None:
    checkpoint = tmp_path / "batchtopk_N6144_k96.pt"
    checkpoint.write_bytes(b"checkpoint")

    assert resolve_seed_checkpoint(tmp_path) == checkpoint


def test_resolve_seed_checkpoint_rejects_missing_checkpoint(tmp_path: Path) -> None:
    try:
        resolve_seed_checkpoint(tmp_path)
    except RuntimeError as error:
        assert "found 0" in str(error)
    else:
        raise AssertionError("expected a missing-checkpoint rejection")


def test_resolve_seed_checkpoint_rejects_ambiguous_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "batchtopk_N4096_k64.pt").write_bytes(b"checkpoint")
    (tmp_path / "batchtopk_N6144_k96.pt").write_bytes(b"checkpoint")

    try:
        resolve_seed_checkpoint(tmp_path)
    except RuntimeError as error:
        assert "found 2" in str(error)
    else:
        raise AssertionError("expected an ambiguous-checkpoint rejection")
