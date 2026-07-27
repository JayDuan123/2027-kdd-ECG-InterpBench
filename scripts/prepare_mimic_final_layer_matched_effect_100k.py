#!/usr/bin/env python
"""Prepare aligned 100k MIMIC final-layer assets after extraction indexing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import (  # noqa: E402
    CONCEPT_SPECS,
    MODEL_SPECS,
    PROTOCOL_100K,
    SEEDS,
    aligned_concepts,
    load_final_layer,
    load_indexed_layer,
    normalize_record_id,
    read_csv,
    safe_model_name,
    split_for_patient,
)
from scripts.prepare_mimic_final_layer_matched_effect import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_npy,
    training_row,
)


DEFAULT_OUT = ROOT / "results/mimic_final_layer_matched_effect_100k_v1"
DEFAULT_MANIFEST = ROOT / "results/activations_external_full_v1/plan_mimic_100k/mimic_main_manifest.csv"
DEFAULT_EXTRACTED = ROOT / "results/activations_external_full_v1/final_layer_100k_v1"
DEFAULT_POOLED_INDEX = ROOT / "results/activations_external_full_v1/pooled"
DEFAULT_POOLED_ARRAYS = ROOT / "results/external_benchmark_v1"
REUSED_MODELS = {"ECG-FM", "HuBERT-ECG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--concept-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--extracted-root", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--pooled-index-root", type=Path, default=DEFAULT_POOLED_INDEX)
    parser.add_argument("--pooled-array-root", type=Path, default=DEFAULT_POOLED_ARRAYS)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-reuse-r2", type=float, default=0.99999)
    parser.add_argument("--max-reuse-normalized-rmse", type=float, default=0.005)
    return parser.parse_args()


def indexed_ids(index_root: Path) -> list[str]:
    return [normalize_record_id(row["ecg_id"]) for row in read_csv(index_root / "records.csv")]


def audit_reused_activation(
    model: str,
    suffix: str,
    layer: int,
    source_ids: list[str],
    pooled_index: Path,
    pooled_path: Path,
    min_r2: float,
    max_normalized_rmse: float,
) -> tuple[Path, dict[str, object]]:
    pooled_ids = indexed_ids(pooled_index)
    if pooled_ids != source_ids:
        raise RuntimeError(f"pooled record order mismatch for {model}")
    pooled = np.load(pooled_path, mmap_mode="r")
    if pooled.shape != (len(source_ids), 768) or not np.isfinite(pooled).all():
        raise RuntimeError(f"invalid pooled activation array for {model}: {pooled.shape}")
    old_records, old_values = load_final_layer(ROOT, suffix, layer)
    position = {record_id: index for index, record_id in enumerate(source_ids)}
    old_ids = [normalize_record_id(row["ecg_id"]) for row in old_records]
    if any(record_id not in position for record_id in old_ids):
        raise RuntimeError(f"4k audit records are not a subset of the 100k source for {model}")
    current = np.asarray(pooled[[position[record_id] for record_id in old_ids]], dtype=np.float64)
    reference = np.asarray(old_values, dtype=np.float64)
    difference = current - reference
    rmse = float(np.sqrt(np.mean(difference**2)))
    reference_sd = float(np.std(reference))
    normalized_rmse = rmse / max(reference_sd, 1e-12)
    r2 = float(1.0 - np.sum(difference**2) / max(np.sum((reference - reference.mean()) ** 2), 1e-12))
    passed = bool(r2 >= min_r2 and normalized_rmse <= max_normalized_rmse)
    audit = {
        "model": model,
        "source": "existing_100k_pooled_verified_as_final_layer",
        "records": len(source_ids),
        "overlap_records": len(old_ids),
        "r2_vs_4k_final_layer": r2,
        "rmse": rmse,
        "normalized_rmse": normalized_rmse,
        "max_abs": float(np.max(np.abs(difference))),
        "pass": passed,
        "activation_path": str(pooled_path),
        "index_root": str(pooled_index),
    }
    if not passed:
        raise RuntimeError(f"pooled/final-layer reuse gate failed for {model}: {audit}")
    return pooled_path, audit


def main() -> None:
    args = parse_args()
    source_rows = [row for row in read_csv(args.concept_manifest) if row.get("status") == "ok"]
    if len(source_rows) != 100_000:
        raise RuntimeError(f"expected 100000 source records, found {len(source_rows)}")
    source_ids = [normalize_record_id(row["record_id"]) for row in source_rows]
    subjects = [str(row["subject_id"]) for row in source_rows]
    if len(set(source_ids)) != len(source_ids):
        raise RuntimeError("duplicate source record IDs")

    activation_paths: dict[str, Path] = {}
    alignment_rows: list[dict[str, object]] = []
    reuse_rows: list[dict[str, object]] = []
    for model, suffix, layer, _n_layers in MODEL_SPECS:
        if model in REUSED_MODELS:
            pooled_index = args.pooled_index_root / suffix / "mimic_f"
            pooled_path = args.pooled_array_root / suffix / "mimic_f/cohort_adapted_sae/pooled_all.npy"
            activation_path, audit = audit_reused_activation(
                model,
                suffix,
                layer,
                source_ids,
                pooled_index,
                pooled_path,
                args.min_reuse_r2,
                args.max_reuse_normalized_rmse,
            )
            source = "reused_verified_pooled"
            reuse_rows.append(audit)
        else:
            index_root = args.extracted_root / suffix / "mimic_f"
            records, activations = load_indexed_layer(ROOT, index_root, layer)
            current_ids = [normalize_record_id(row["ecg_id"]) for row in records]
            if current_ids != source_ids or activations.shape != (len(source_ids), 768):
                raise RuntimeError(f"extracted record order or shape mismatch for {model}")
            activation_path = args.out / "derived/activations" / safe_model_name(model) / f"layer_{layer:02d}.npy"
            atomic_npy(activation_path, activations)
            source = "new_100k_final_layer_extraction"
        activation_paths[model] = activation_path
        alignment_rows.append(
            {
                "model": model,
                "model_suffix": suffix,
                "final_layer_index": layer,
                "records": len(source_ids),
                "dimension": 768,
                "record_order_matches_manifest": True,
                "activation_source": source,
                "activation_path": str(activation_path),
            }
        )
    atomic_json(
        args.out / "derived/reuse_equivalence_audit.json",
        {
            "status": "pass",
            "min_r2": args.min_reuse_r2,
            "max_normalized_rmse": args.max_reuse_normalized_rmse,
            "models": reuse_rows,
        },
    )

    splits = np.asarray([split_for_patient(value) for value in subjects])
    records_path = args.out / "derived/records.csv"
    atomic_csv(
        records_path,
        [
            {"ecg_id": record_id, "patient_id": patient_id, "split": split}
            for record_id, patient_id, split in zip(source_ids, subjects, splits)
        ],
    )
    concepts, concept_names, means, scales, finite_counts = aligned_concepts(
        source_ids, source_rows, splits == "train"
    )
    atomic_csv(
        args.out / "derived/concepts_standardized.csv",
        [
            {
                "ecg_id": record_id,
                **{
                    name: "" if not np.isfinite(concepts[row_index, column]) else float(concepts[row_index, column])
                    for column, name in enumerate(concept_names)
                },
            }
            for row_index, record_id in enumerate(source_ids)
        ],
    )
    atomic_csv(
        args.out / "derived/concept_registry.csv",
        [
            {
                "concept_id": name,
                "family": family,
                "train_mean_raw": float(means[index]),
                "train_sd_raw": float(scales[index]),
                "finite_records": int(finite_counts[index]),
                "missing_records": int(len(source_ids) - finite_counts[index]),
            }
            for index, (name, family) in enumerate(CONCEPT_SPECS)
        ],
    )
    atomic_csv(args.out / "derived/alignment_audit.csv", alignment_rows)

    training_rows = []
    for model, suffix, layer, n_layers in MODEL_SPECS:
        for seed in SEEDS:
            training_rows.append(
                training_row(
                    len(training_rows), model, suffix, layer, n_layers, activation_paths[model],
                    records_path, args.out, seed, args.steps, args.batch_size, args.learning_rate,
                )
            )
    atomic_csv(args.out / "training_manifest.csv", training_rows)
    split_counts = {name: int(np.sum(splits == name)) for name in ("train", "val", "test")}
    patient_counts = {
        name: len({patient for patient, split in zip(subjects, splits) if split == name})
        for name in ("train", "val", "test")
    }
    protocol = {
        "status": "prepared",
        "protocol": PROTOCOL_100K,
        "cohort": "MIMIC-IV-ECG",
        "records": len(source_ids),
        "patients": len(set(subjects)),
        "record_split_counts": split_counts,
        "patient_split_counts": patient_counts,
        "split_rule": "SHA256 external-head-v1 patient split, 70/10/20 buckets",
        "models": [row[0] for row in MODEL_SPECS],
        "final_layers_zero_based": {row[0]: row[2] for row in MODEL_SPECS},
        "activation_sources": {row["model"]: row["activation_source"] for row in alignment_rows},
        "concepts": concept_names,
        "concept_families": {name: family for name, family in CONCEPT_SPECS},
        "label_missingness": "finite labels only; no label imputation",
        "sae": {
            "architecture": "BatchTopK", "d": 768, "N": 6144, "k": 96, "expansion": 8,
            "seeds": list(SEEDS), "steps": args.steps, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        "comparison": "Dense 768 coordinates versus target-independent SAE subsets of 768 coordinates",
        "data_policy": "new derived outputs only; source MIMIC waveforms and existing activation shards remain read-only",
        "training_cells": len(training_rows),
    }
    atomic_json(args.out / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
