#!/usr/bin/env python
"""Materialize aligned layer matrices and target panels for the MIMIC source atlas."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import MODEL_SPECS, load_indexed_layer, normalize_record_id  # noqa: E402
from benchmark_v1.mimic_source_benchmark import (  # noqa: E402
    ACTIVATION_ROOT,
    CONCEPT_SPECS,
    DIAGNOSIS_SPECS,
    ICD_MATRIX,
    PROTOCOL,
    RESULT_ROOT,
    SOURCE_MANIFEST,
    complete_waveform_row,
    patient_split,
    read_csv,
    selected_layers,
    source_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / SOURCE_MANIFEST)
    parser.add_argument("--icd-matrix", type=Path, default=ROOT / ICD_MATRIX)
    parser.add_argument("--activation-root", type=Path, default=ROOT / ACTIVATION_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / RESULT_ROOT)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def main() -> None:
    args = parse_args()
    source = source_rows(args.manifest, args.max_records)
    expected = args.max_records if args.max_records > 0 else 100_000
    if len(source) != expected:
        raise RuntimeError(f"expected {expected} source rows, found {len(source)}")
    record_ids = [normalize_record_id(row["record_id"]) for row in source]
    patients = [str(row["subject_id"]) for row in source]
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError("duplicate source record IDs")

    derived = args.out / "derived"
    records_path = derived / "records.csv"
    record_rows = [
        {
            "ecg_id": record_id,
            "patient_id": patient,
            "split": patient_split(patient),
            "waveform_complete_case": int(complete_waveform_row(row)),
        }
        for record_id, patient, row in zip(record_ids, patients, source)
    ]
    split_counts = {name: sum(row["split"] == name for row in record_rows) for name in ("train", "val", "test")}
    if min(split_counts.values()) == 0:
        raise RuntimeError(f"empty patient split: {split_counts}")
    atomic_csv(records_path, record_rows)

    concept_rows = []
    for record_id, row in zip(record_ids, source):
        rr = finite(row.get("rr_mean_ms"))
        values = {
            "heart_rate_bpm": 60000.0 / rr if np.isfinite(rr) and rr > 0 else "",
        }
        for name, _family in CONCEPT_SPECS:
            if name == "heart_rate_bpm":
                continue
            value = finite(row.get(name))
            values[name] = value if np.isfinite(value) else ""
        concept_rows.append({"ecg_id": record_id, **values})
    atomic_csv(derived / "concepts_matrix.csv", concept_rows)
    atomic_csv(
        derived / "concept_registry.csv",
        [{"concept_id": name, "family": family, "main": "yes"} for name, family in CONCEPT_SPECS],
    )

    icd_by_study = {str(row["study_id"]): row for row in read_csv(args.icd_matrix)}
    task_rows = []
    for record_id in record_ids:
        if record_id not in icd_by_study:
            raise KeyError(f"MIMIC ICD row missing for study_id={record_id}")
        source_task = icd_by_study[record_id]
        task_rows.append(
            {
                "ecg_id": record_id,
                **{name: int(float(source_task[name])) for name, _family in DIAGNOSIS_SPECS},
            }
        )
    atomic_csv(derived / "tasks_matrix.csv", task_rows)
    atomic_csv(
        derived / "task_registry.csv",
        [
            {"task_id": name, "task_family": family, "main": "yes"}
            for name, family in DIAGNOSIS_SPECS
        ],
    )

    catalog_rows = []
    for model, suffix, _final_layer, n_layers in MODEL_SPECS:
        index_root = args.activation_root / suffix / "mimic_f"
        for target_depth, layer in zip(
            (0.0, 0.25, 0.5, 0.75, 1.0), selected_layers(n_layers)
        ):
            indexed_records, activations = load_indexed_layer(ROOT, index_root, layer)
            indexed_ids = [normalize_record_id(row["ecg_id"]) for row in indexed_records]
            if indexed_ids != record_ids:
                raise RuntimeError(f"record order mismatch for {model} layer {layer}")
            if activations.shape != (len(record_ids), 768) or not np.isfinite(activations).all():
                raise RuntimeError(f"invalid activation matrix for {model} layer {layer}: {activations.shape}")
            activation_path = derived / "activations" / suffix / f"layer_{layer:02d}.npy"
            atomic_npy(activation_path, activations)
            catalog_rows.append(
                {
                    "model": model,
                    "feature_suffix": suffix,
                    "layer": layer,
                    "relative_depth": target_depth,
                    "actual_relative_depth": layer / max(n_layers - 1, 1),
                    "n_layers": n_layers,
                    "d_hidden": 768,
                    "activation_path": str(activation_path),
                    "records_path": str(records_path),
                }
            )
            del activations
    atomic_csv(derived / "layer_catalog.csv", catalog_rows)
    protocol = {
        "protocol": PROTOCOL,
        "status": "materialized",
        "records": len(record_ids),
        "patients": len(set(patients)),
        "split_counts": split_counts,
        "complete_waveform_records": sum(complete_waveform_row(row) for row in source),
        "models": 6,
        "model_depth_cells": len(catalog_rows),
        "waveform_concepts": len(CONCEPT_SPECS),
        "diagnosis_targets": len(DIAGNOSIS_SPECS),
        "analysis_missingness_policy": "target-specific masks for multiscale semantics; common complete-case panel for matched dictionary/sparse comparisons",
        "data_policy": "source waveform and label files are read-only; only derived benchmark artifacts are created",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
