from __future__ import annotations

from pathlib import Path

import pandas as pd

from benchmark_v1.multiscale_sae import read_csv
from scripts.analyze_multiscale_sae_inference import validate_matched_scale_cells
from scripts.audit_multiscale_sae import (
    matched_scale_manifest_audit,
    record_manifest_alignment_audit,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest_frame() -> tuple[list[dict[str, str]], pd.DataFrame]:
    rows = read_csv(ROOT / "results/multiscale_sae_v1/training_manifest.csv")
    frame = pd.DataFrame(
        {
            "model": [row["model"] for row in rows],
            "relative_depth": [float(row["relative_depth"]) for row in rows],
            "expansion_E": [int(row["expansion_E"]) for row in rows],
            "seed": [int(row["seed"]) for row in rows],
        }
    )
    return rows, frame


def test_frozen_manifest_is_complete_matched_scale_grid() -> None:
    rows, frame = manifest_frame()
    blocks, grid_pass, issues = matched_scale_manifest_audit(rows)
    assert grid_pass
    assert not issues
    assert len(blocks) == 75
    assert {row["hidden_dimensions"] for row in blocks} == {"768"}
    assert all(";" not in row["absolute_dictionary_widths"] for row in blocks)
    assert all(";" not in row["active_budgets"] for row in blocks)
    assert validate_matched_scale_cells(frame) == 75


def test_inference_rejects_one_missing_fm_cell() -> None:
    _, frame = manifest_frame()
    try:
        validate_matched_scale_cells(frame.iloc[:-1].copy())
    except RuntimeError as error:
        assert "complete matched-scale FM grid" in str(error)
    else:
        raise AssertionError("incomplete matched-scale grid was accepted")


def test_inference_rejects_per_model_scale_substitution() -> None:
    _, frame = manifest_frame()
    altered = frame.copy()
    target = (
        (altered.model == "CARDIAC-FM")
        & (altered.relative_depth == 0.0)
        & (altered.expansion_E == 1)
        & (altered.seed == 4311)
    )
    altered.loc[target, "expansion_E"] = 32
    try:
        validate_matched_scale_cells(altered)
    except RuntimeError as error:
        assert "complete matched-scale FM grid" in str(error)
    else:
        raise AssertionError("per-model scale substitution was accepted")


def test_manifest_audit_rejects_absolute_width_mismatch() -> None:
    rows, _ = manifest_frame()
    altered = [dict(row) for row in rows]
    altered[0]["N"] = str(int(altered[0]["N"]) + 1)
    blocks, grid_pass, _ = matched_scale_manifest_audit(altered)
    assert not grid_pass
    failed = [row for row in blocks if row["manifest_status"] == "fail"]
    assert len(failed) == 1
    assert "absolute_dictionary_width_mismatch" in failed[0]["manifest_reasons"]


def test_all_models_use_the_same_ordered_record_manifest() -> None:
    rows, _ = manifest_frame()
    audit_rows, passed = record_manifest_alignment_audit(
        rows, tuple(sorted({row["model"] for row in rows}))
    )
    assert passed
    assert {row["record_count"] for row in audit_rows} == {21799}
    assert len({row["ordered_ecg_split_sha256"] for row in audit_rows}) == 1
