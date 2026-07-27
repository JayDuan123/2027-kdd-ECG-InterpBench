#!/usr/bin/env python
"""Fit one final-layer dense multi-concept readout for matched-effect tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.matched_effect import fit_multitarget_ridge_readout  # noqa: E402
from benchmark_v1.multiscale_sae import read_csv, standardized_concepts  # noqa: E402
from scripts.run_accessibility_calibration_worker import (  # noqa: E402
    atomic_json,
    atomic_npz,
    resolved,
)
from scripts.run_final_layer_sparse_accessibility_worker import (  # noqa: E402
    ALPHAS,
    checkpoint_normalization,
    final_groups,
    limited,
    normalize_rows,
    parse_csv_numbers,
)


PROTOCOL = "final_layer_matched_effect_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--expansion", type=int, default=8)
    parser.add_argument(
        "--alphas", type=lambda x: parse_csv_numbers(x, float), default=ALPHAS
    )
    parser.add_argument("--max-records-per-split", type=int, default=0)
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/readouts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = final_groups(args.manifest, args.expansion)
    if not 0 <= args.model_index < len(groups):
        raise IndexError(f"model index outside 0..{len(groups) - 1}")
    _, rows = groups[args.model_index]
    row = rows[0]
    output = args.output_root / (
        f"model_{args.model_index:02d}_{row['model_safe']}_layer{int(row['layer']):02d}"
    )
    archive_path = output / "readout.npz"
    summary_path = output / "summary.json"
    if archive_path.exists() and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and existing.get("protocol") == PROTOCOL:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    acts = np.load(resolved(row["activation_path"]), mmap_mode="r")
    records = read_csv(resolved(row["records_path"]))
    if len(acts) != len(records) or acts.shape[1] != int(row["d_hidden"]):
        raise RuntimeError("activation and record dimensions do not align")
    splits = np.asarray([item["split"] for item in records])
    full_train_mask = splits == "train"
    train_idx = limited(np.flatnonzero(full_train_mask), args.max_records_per_split)
    validation_idx = limited(
        np.flatnonzero(splits == "val"), args.max_records_per_split
    )
    test_idx = limited(np.flatnonzero(splits == "test"), args.max_records_per_split)
    concepts, concept_names, _, _ = standardized_concepts(
        [item["ecg_id"] for item in records],
        read_csv(args.concepts),
        full_train_mask,
    )
    if len(concept_names) != 49:
        raise RuntimeError("expected the canonical 49 waveform concepts")
    mean, scale = checkpoint_normalization(resolved(row["checkpoint"]))
    x_train = normalize_rows(acts, train_idx, mean, scale)
    x_validation = normalize_rows(acts, validation_idx, mean, scale)
    x_test = normalize_rows(acts, test_idx, mean, scale)
    fitted = fit_multitarget_ridge_readout(
        x_train,
        concepts[train_idx],
        x_validation,
        concepts[validation_idx],
        x_test,
        concepts[test_idx],
        alphas=args.alphas,
    )
    family_by_concept = {
        item["concept_id"]: item["family"]
        for item in read_csv(ROOT / "configs/concepts.csv")
        if item.get("main") == "yes"
    }
    atomic_npz(
        archive_path,
        concept_names=np.asarray(concept_names),
        concept_families=np.asarray([family_by_concept[name] for name in concept_names]),
        activation_mean=mean,
        activation_scale=scale,
        feature_mean=fitted.feature_mean,
        feature_scale=fitted.feature_scale,
        coefficients=fitted.coefficients,
        intercepts=fitted.intercepts,
        selected_alphas=fitted.selected_alphas,
        validation_correlations=fitted.validation_correlations,
        test_correlations=fitted.test_correlations,
    )
    summary = {
        "status": "complete",
        "protocol": PROTOCOL,
        "model_index": args.model_index,
        "model": row["model"],
        "model_safe": row["model_safe"],
        "layer": int(row["layer"]),
        "relative_depth": float(row["relative_depth"]),
        "n_train": len(train_idx),
        "n_validation": len(validation_idx),
        "n_test": len(test_idx),
        "n_concepts": len(concept_names),
        "alphas": list(args.alphas),
        "mean_validation_abs_r": float(np.mean(np.abs(fitted.validation_correlations))),
        "mean_test_abs_r": float(np.mean(np.abs(fitted.test_correlations))),
        "readout_archive": str(archive_path),
        "selection": "ridge alpha selected per concept on validation only",
        "smoke": args.max_records_per_split > 0,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
