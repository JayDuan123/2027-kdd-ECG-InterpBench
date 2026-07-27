from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from benchmark_v1.mimic_source_benchmark import (
    complete_waveform_row,
    expected_extraction_commands,
    patient_split,
    selected_layers,
)
from benchmark_v1.multiscale_sae import (
    LayerSpec,
    build_manifest_rows_from_specs,
    standardized_concepts,
)


def test_mimic_depth_mapping_matches_frozen_relative_depths() -> None:
    assert selected_layers(12) == (0, 3, 6, 8, 11)
    assert selected_layers(6) == (0, 1, 3, 4, 5)
    assert expected_extraction_commands(100_000, 128) == 4692


def test_patient_split_is_deterministic() -> None:
    assert patient_split("123") == patient_split("123")
    assert patient_split("123") in {"train", "val", "test"}


def test_complete_waveform_gate_requires_all_seven_source_measurements() -> None:
    row = {
        "rr_mean_ms": "800",
        "qrs_duration_ms": "100",
        "pr_interval_ms": "160",
        "qt_like_ms": "410",
        "r_amp_global_mv": "1.2",
        "st_amp_global_mv": "0.1",
        "t_amp_global_mv": "0.3",
    }
    assert complete_waveform_row(row)
    row["pr_interval_ms"] = ""
    assert not complete_waveform_row(row)


def test_standardization_can_preserve_missing_targets() -> None:
    rows = [
        {"ecg_id": "1", "a": "1", "b": "2"},
        {"ecg_id": "2", "a": "2", "b": ""},
        {"ecg_id": "3", "a": "3", "b": "4"},
    ]
    values, names, _, _ = standardized_concepts(
        ["1", "2", "3"], rows, np.asarray([True, True, False]), preserve_missing=True
    )
    assert names == ["a", "b"]
    assert np.isnan(values[1, 1])
    imputed, _, _, _ = standardized_concepts(
        ["1", "2", "3"], rows, np.asarray([True, True, False])
    )
    assert imputed[1, 1] == 0.0


def test_explicit_catalog_builds_matched_grid() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        specs = []
        for model in ("A", "B"):
            activation = root / f"{model}.npy"
            records = root / f"{model}.csv"
            activation.touch()
            records.touch()
            specs.append(
                LayerSpec(
                    model=model,
                    suffix=model.lower(),
                    layer=5,
                    target_relative_depth=1.0,
                    actual_relative_depth=1.0,
                    n_layers=6,
                    d_hidden=768,
                    activation_path=activation,
                    records_path=records,
                )
            )
        rows = build_manifest_rows_from_specs(
            specs, root / "out", expansions=(1, 8), seeds=(1, 2), steps=10
        )
        assert len(rows) == 8
        assert {(row["model"], row["expansion_E"], row["seed"]) for row in rows} == {
            (model, expansion, seed)
            for model in ("A", "B")
            for expansion in (1, 8)
            for seed in (1, 2)
        }
