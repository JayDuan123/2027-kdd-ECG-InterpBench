"""Frozen configuration for the 100k MIMIC source-cohort benchmark."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

from benchmark_v1.mimic_matched_effect import CONCEPT_SPECS, MODEL_SPECS
from benchmark_v1.multiscale_sae import LayerSpec, relative_layer_indices


PROTOCOL = "mimic_source_benchmark_100k_v1"
SOURCE_MANIFEST = Path(
    "results/activations_external_full_v1/plan_mimic_100k/mimic_main_manifest.csv"
)
ICD_MATRIX = Path("results/multicohort/mimic_icd_label_matrix.csv")
ACTIVATION_ROOT = Path("results/activations_external_full_v1/mimic_source_100k_v1")
PLAN_ROOT = Path("results/mimic_source_benchmark_100k_v1/activation_plan")
RESULT_ROOT = Path("results/mimic_source_benchmark_100k_v1")

CANONICAL_MODEL = {
    "CARDIAC-FM": "cardiac_fm",
    "CSFM": "csfm",
    "ECG-FM": "ecg_fm",
    "ECG-JEPA": "ecg_jepa",
    "HuBERT-ECG": "hubert_ecg",
    "ST-MEM": "st_mem",
}

DIAGNOSIS_SPECS = (
    ("af_rhythm_icd", "rate_rhythm"),
    ("bbb_conduction_icd", "conduction"),
    ("qt_interval_icd", "repolarization"),
    ("mi_ischemia_icd", "ischemia"),
    ("hypertrophy_icd", "chamber"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def selected_layers(n_layers: int) -> tuple[int, ...]:
    return tuple(relative_layer_indices(n_layers))


def source_rows(path: Path, max_records: int = 0) -> list[dict[str, str]]:
    rows = [row for row in read_csv(path) if row.get("status") == "ok"]
    if max_records > 0:
        rows = rows[:max_records]
    return rows


def patient_split(patient_id: str) -> str:
    digest = hashlib.sha256(f"external-head-v1:{patient_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10
    return "train" if bucket < 7 else "val" if bucket < 8 else "test"


def layer_catalog(
    activation_root: Path,
    derived_root: Path,
    records_path: Path,
) -> list[LayerSpec]:
    specs: list[LayerSpec] = []
    for model, suffix, _final_layer, n_layers in MODEL_SPECS:
        for target_depth, layer in zip(
            (0.0, 0.25, 0.5, 0.75, 1.0), selected_layers(n_layers)
        ):
            activation_path = (
                derived_root / "activations" / suffix / f"layer_{layer:02d}.npy"
            )
            specs.append(
                LayerSpec(
                    model=model,
                    suffix=suffix,
                    layer=layer,
                    target_relative_depth=target_depth,
                    actual_relative_depth=layer / max(n_layers - 1, 1),
                    n_layers=n_layers,
                    d_hidden=768,
                    activation_path=activation_path,
                    records_path=records_path,
                )
            )
    return specs


def expected_extraction_commands(records: int, batch_size: int) -> int:
    return len(MODEL_SPECS) * math.ceil(records / batch_size)


def concept_names() -> tuple[str, ...]:
    return tuple(name for name, _family in CONCEPT_SPECS)


def diagnosis_names() -> tuple[str, ...]:
    return tuple(name for name, _family in DIAGNOSIS_SPECS)


def complete_waveform_row(row: dict[str, str]) -> bool:
    source_names = (
        "rr_mean_ms",
        "qrs_duration_ms",
        "pr_interval_ms",
        "qt_like_ms",
        "r_amp_global_mv",
        "st_amp_global_mv",
        "t_amp_global_mv",
    )
    for name in source_names:
        try:
            if not math.isfinite(float(row.get(name, ""))):
                return False
        except (TypeError, ValueError):
            return False
    return True
