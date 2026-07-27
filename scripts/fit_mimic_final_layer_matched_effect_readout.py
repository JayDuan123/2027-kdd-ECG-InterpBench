#!/usr/bin/env python
"""Fit one masked-label final-layer readout for the MIMIC replication."""

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
    PROTOCOL,
    fit_masked_ridge_readout,
    read_csv,
    safe_model_name,
)
from scripts.run_accessibility_calibration_worker import atomic_json, atomic_npz  # noqa: E402


ALPHAS = (0.1, 1.0, 10.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/training_manifest.csv",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/derived/concepts_standardized.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/readouts",
    )
    parser.add_argument("--max-records-per-split", type=int, default=0)
    parser.add_argument("--protocol", default=PROTOCOL)
    return parser.parse_args()


def limited(indices: np.ndarray, limit: int) -> np.ndarray:
    return indices[:limit] if limit > 0 else indices


def aligned_concept_matrix(records: list[dict[str, str]], path: Path) -> tuple[np.ndarray, list[str]]:
    rows = read_csv(path)
    names = [name for name in rows[0] if name != "ecg_id"]
    expected = [name for name, _ in CONCEPT_SPECS]
    if names != expected:
        raise RuntimeError(f"concept order mismatch: {names} != {expected}")
    by_id = {row["ecg_id"]: row for row in rows}
    values = np.full((len(records), len(names)), np.nan, dtype=np.float32)
    for row_index, record in enumerate(records):
        source = by_id.get(record["ecg_id"])
        if source is None:
            raise KeyError(f"concept row missing for {record['ecg_id']}")
        for concept_index, name in enumerate(names):
            try:
                values[row_index, concept_index] = float(source[name])
            except (TypeError, ValueError):
                pass
    return values, names


def main() -> None:
    args = parse_args()
    if not 0 <= args.model_index < len(MODEL_SPECS):
        raise IndexError(f"model index outside 0..{len(MODEL_SPECS) - 1}")
    model, _suffix, layer, _n_layers = MODEL_SPECS[args.model_index]
    manifest_rows = [row for row in read_csv(args.manifest) if row["model"] == model]
    if not manifest_rows:
        raise RuntimeError(f"manifest has no rows for {model}")
    row = manifest_rows[0]
    output = args.output_root / f"model_{args.model_index:02d}_{safe_model_name(model)}_layer{layer:02d}"
    archive_path = output / "readout.npz"
    summary_path = output / "summary.json"
    if archive_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == args.protocol:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    activations = np.load(Path(row["activation_path"]), mmap_mode="r")
    records = read_csv(Path(row["records_path"]))
    if activations.shape != (len(records), 768):
        raise RuntimeError("activation and record dimensions do not align")
    concepts, concept_names = aligned_concept_matrix(records, args.concepts)
    splits = np.asarray([item["split"] for item in records])
    full_train = np.flatnonzero(splits == "train")
    indices = {
        "train": limited(full_train, args.max_records_per_split),
        "val": limited(np.flatnonzero(splits == "val"), args.max_records_per_split),
        "test": limited(np.flatnonzero(splits == "test"), args.max_records_per_split),
    }
    activation_mean = np.mean(np.asarray(activations[full_train]), axis=0, dtype=np.float64).astype(np.float32)
    activation_scale = np.std(np.asarray(activations[full_train]), axis=0, dtype=np.float64).astype(np.float32)
    activation_scale = np.where(activation_scale > 1e-6, activation_scale, 1.0).astype(np.float32)

    def normalized(selected: np.ndarray) -> np.ndarray:
        return (
            (np.asarray(activations[selected], dtype=np.float32) - activation_mean)
            / activation_scale
        ).astype(np.float32)

    fitted, label_counts = fit_masked_ridge_readout(
        normalized(indices["train"]),
        concepts[indices["train"]],
        normalized(indices["val"]),
        concepts[indices["val"]],
        normalized(indices["test"]),
        concepts[indices["test"]],
        alphas=ALPHAS,
    )
    families = np.asarray([family for _name, family in CONCEPT_SPECS])
    atomic_npz(
        archive_path,
        concept_names=np.asarray(concept_names),
        concept_families=families,
        activation_mean=activation_mean,
        activation_scale=activation_scale,
        coefficients=fitted.coefficients,
        intercepts=fitted.intercepts,
        selected_alphas=fitted.selected_alphas,
        validation_correlations=fitted.validation_correlations,
        test_correlations=fitted.test_correlations,
        finite_label_counts=label_counts,
    )
    summary = {
        "status": "complete",
        "protocol": args.protocol,
        "model_index": args.model_index,
        "model": model,
        "model_safe": safe_model_name(model),
        "layer": layer,
        "n_train": int(len(indices["train"])),
        "n_validation": int(len(indices["val"])),
        "n_test": int(len(indices["test"])),
        "n_concepts": len(concept_names),
        "concepts": concept_names,
        "alphas": list(ALPHAS),
        "mean_validation_abs_r": float(np.mean(np.abs(fitted.validation_correlations))),
        "mean_test_abs_r": float(np.mean(np.abs(fitted.test_correlations))),
        "finite_label_counts_train_val_test": label_counts.tolist(),
        "archive": str(archive_path),
        "selection": "ridge alpha selected per concept on validation only; finite labels only",
        "smoke": args.max_records_per_split > 0,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
