#!/usr/bin/env python
"""Prepare aligned derived assets for the MIMIC final-layer replication."""

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

from benchmark_v1.mimic_matched_effect import (  # noqa: E402
    CONCEPT_SPECS,
    MODEL_SPECS,
    PROTOCOL,
    SEEDS,
    aligned_concepts,
    load_final_layer,
    normalize_record_id,
    read_csv,
    safe_model_name,
    split_for_patient,
)
from benchmark_v1.multiscale_sae import canonical_config_hash  # noqa: E402


DEFAULT_OUT = ROOT / "results/mimic_final_layer_matched_effect_v1"
DEFAULT_CONCEPT_MANIFEST = (
    ROOT
    / "results/activations_external_full_v1/plan_mimic_100k/mimic_layer_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--concept-manifest", type=Path, default=DEFAULT_CONCEPT_MANIFEST)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--smoke-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32))
    os.replace(tmp, path)


def training_row(
    task_index: int,
    model: str,
    suffix: str,
    layer: int,
    n_layers: int,
    activation_path: Path,
    records_path: Path,
    output_root: Path,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    safe = safe_model_name(model)
    cell_root = output_root / "checkpoints" / safe / f"layer_{layer:02d}" / "E8" / f"seed{seed}"
    row: dict[str, object] = {
        "task_index": task_index,
        "model": model,
        "model_safe": safe,
        "feature_suffix": suffix,
        "layer": layer,
        "relative_depth": 1.0,
        "actual_relative_depth": 1.0,
        "n_layers": n_layers,
        "d_hidden": 768,
        "sparsity_arm": "fixed_k_over_d",
        "expansion_E": 8,
        "N": 6144,
        "k": 96,
        "k_over_d": 0.125,
        "k_over_N": 0.015625,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "activation_path": str(activation_path),
        "records_path": str(records_path),
        "checkpoint": str(cell_root / "batchtopk_N6144_k96.pt"),
        "metrics": str(cell_root / "metrics.json"),
    }
    row["config_hash"] = canonical_config_hash(row)
    return row


def main() -> None:
    args = parse_args()
    concept_manifest = read_csv(args.concept_manifest)
    reference_ids: list[str] | None = None
    reference_subjects: list[str] | None = None
    activation_paths: dict[str, Path] = {}
    alignment_rows: list[dict[str, object]] = []

    for model, suffix, final_layer, _n_layers in MODEL_SPECS:
        records, activations = load_final_layer(ROOT, suffix, final_layer)
        if activations.shape != (4096, 768):
            raise RuntimeError(f"unexpected final-layer shape for {model}: {activations.shape}")
        ids = [normalize_record_id(row["ecg_id"]) for row in records]
        subjects = [str(row["subject_id"]) for row in records]
        if reference_ids is None:
            reference_ids = ids
            reference_subjects = subjects
        elif ids != reference_ids or subjects != reference_subjects:
            raise RuntimeError(f"record or patient order mismatch for {model}")
        safe = safe_model_name(model)
        output_path = args.out / "derived" / "activations" / safe / f"layer_{final_layer:02d}.npy"
        atomic_npy(output_path, activations)
        activation_paths[model] = output_path
        alignment_rows.append(
            {
                "model": model,
                "model_suffix": suffix,
                "final_layer_index": final_layer,
                "records": len(ids),
                "dimension": activations.shape[1],
                "record_order_matches_reference": True,
                "activation_path": str(output_path),
            }
        )

    assert reference_ids is not None and reference_subjects is not None
    splits = np.asarray([split_for_patient(value) for value in reference_subjects])
    records_path = args.out / "derived/records.csv"
    record_rows = [
        {
            "ecg_id": ecg_id,
            "patient_id": patient_id,
            "split": split,
        }
        for ecg_id, patient_id, split in zip(reference_ids, reference_subjects, splits)
    ]
    atomic_csv(records_path, record_rows)

    concepts, concept_names, concept_means, concept_scales, finite_counts = aligned_concepts(
        reference_ids, concept_manifest, splits == "train"
    )
    concept_path = args.out / "derived/concepts_standardized.csv"
    concept_rows: list[dict[str, object]] = []
    for row_index, ecg_id in enumerate(reference_ids):
        row: dict[str, object] = {"ecg_id": ecg_id}
        for concept_index, concept in enumerate(concept_names):
            value = concepts[row_index, concept_index]
            row[concept] = "" if not np.isfinite(value) else float(value)
        concept_rows.append(row)
    atomic_csv(concept_path, concept_rows)
    registry_path = args.out / "derived/concept_registry.csv"
    registry_rows = []
    for concept_index, (concept, family) in enumerate(CONCEPT_SPECS):
        registry_rows.append(
            {
                "concept_id": concept,
                "family": family,
                "train_mean_raw": float(concept_means[concept_index]),
                "train_sd_raw": float(concept_scales[concept_index]),
                "finite_records": int(finite_counts[concept_index]),
                "missing_records": int(len(reference_ids) - finite_counts[concept_index]),
            }
        )
    atomic_csv(registry_path, registry_rows)
    atomic_csv(args.out / "derived/alignment_audit.csv", alignment_rows)

    manifest_rows: list[dict[str, object]] = []
    for model, suffix, final_layer, n_layers in MODEL_SPECS:
        for seed in SEEDS:
            manifest_rows.append(
                training_row(
                    len(manifest_rows), model, suffix, final_layer, n_layers,
                    activation_paths[model], records_path, args.out, seed, args.steps,
                    args.batch_size, args.learning_rate,
                )
            )
    atomic_csv(args.out / "training_manifest.csv", manifest_rows)
    smoke_row = training_row(
        0,
        MODEL_SPECS[0][0],
        MODEL_SPECS[0][1],
        MODEL_SPECS[0][2],
        MODEL_SPECS[0][3],
        activation_paths[MODEL_SPECS[0][0]],
        records_path,
        args.out / "smoke",
        SEEDS[0],
        args.smoke_steps,
        args.batch_size,
        args.learning_rate,
    )
    atomic_csv(args.out / "smoke_training_manifest.csv", [smoke_row])

    split_counts = {name: int(np.sum(splits == name)) for name in ("train", "val", "test")}
    patient_counts = {
        name: len({patient for patient, split in zip(reference_subjects, splits) if split == name})
        for name in ("train", "val", "test")
    }
    protocol = {
        "status": "prepared",
        "protocol": PROTOCOL,
        "cohort": "MIMIC-IV-ECG",
        "models": [row[0] for row in MODEL_SPECS],
        "final_layers_zero_based": {row[0]: row[2] for row in MODEL_SPECS},
        "records": len(reference_ids),
        "patients": len(set(reference_subjects)),
        "record_split_counts": split_counts,
        "patient_split_counts": patient_counts,
        "split_rule": "SHA256 external-head-v1 patient split, 70/10/20 buckets",
        "concepts": concept_names,
        "concept_families": {name: family for name, family in CONCEPT_SPECS},
        "label_missingness": "finite labels only for readout fitting, feature selection, and centroids; no label imputation",
        "sae": {
            "architecture": "BatchTopK",
            "d": 768,
            "N": 6144,
            "k": 96,
            "expansion": 8,
            "seeds": list(SEEDS),
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "comparison": "Dense 768 coordinates versus target-independent SAE subsets of 768 coordinates",
        "data_policy": "derived outputs only; source MIMIC waveforms and indexed activation shards remain read-only",
        "training_cells": len(manifest_rows),
    }
    atomic_json(args.out / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
