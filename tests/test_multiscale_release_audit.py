from __future__ import annotations

from pathlib import Path

from scripts.audit_multiscale_sae_release import resolve_release_path


def test_relative_release_path_is_resolved_from_benchmark_root(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "audit.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n")
    resolved, relative = resolve_release_path(Path("results/audit.json"), tmp_path)
    assert resolved == artifact.resolve()
    assert relative == Path("results/audit.json")


def test_absolute_release_path_has_the_same_identity(tmp_path: Path) -> None:
    artifact = tmp_path / "results" / "audit.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n")
    resolved, relative = resolve_release_path(artifact, tmp_path)
    assert resolved == artifact.resolve()
    assert relative == Path("results/audit.json")


def test_release_path_outside_root_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    try:
        resolve_release_path(outside, tmp_path)
    except RuntimeError as error:
        assert "outside benchmark root" in str(error)
    else:
        raise AssertionError("expected an out-of-root release artifact rejection")
