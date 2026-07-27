#!/usr/bin/env python
"""Audit reusable 100k artifacts and freeze the live-atom matched-768 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import MODEL_SPECS, SEEDS, read_csv  # noqa: E402
from scripts.prepare_mimic_final_layer_matched_effect import atomic_csv, atomic_json  # noqa: E402
from scripts.run_mimic_final_layer_matched_effect_worker import matched_live_capacity_quality  # noqa: E402


PROTOCOL = "mimic_final_layer_live_atom_matched_effect_100k_v1"
DEFAULT_SOURCE = ROOT / "results/mimic_final_layer_matched_effect_100k_v1"
DEFAULT_OUT = ROOT / "results/mimic_final_layer_live_atom_matched_effect_100k_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-validation-r2", type=float, default=0.90)
    parser.add_argument("--min-validation-live-features", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.source_root / "training_manifest.csv"
    manifest = sorted(read_csv(manifest_path), key=lambda row: int(row["task_index"]))
    if len(manifest) != len(MODEL_SPECS) * len(SEEDS):
        raise RuntimeError(f"expected 18 reusable SAE cells, found {len(manifest)}")
    expected_cells = {(model, seed) for model, _suffix, _layer, _depth in MODEL_SPECS for seed in SEEDS}
    actual_cells = {(row["model"], int(row["seed"])) for row in manifest}
    if actual_cells != expected_cells:
        raise RuntimeError("reusable SAE model/seed cells do not match the frozen design")

    audit_rows = []
    for row in manifest:
        metrics_path = Path(row["metrics"])
        checkpoint_path = Path(row["checkpoint"])
        if not metrics_path.exists() or not checkpoint_path.exists():
            raise RuntimeError(f"missing reusable SAE artifact for task {row['task_index']}")
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("status") != "complete":
            raise RuntimeError(f"incomplete reusable SAE metrics: {metrics_path}")
        gate = matched_live_capacity_quality(
            metrics, args.min_validation_r2, args.min_validation_live_features
        )
        if not gate["pass"]:
            raise RuntimeError(f"capacity gate failed before execution: {row['model']} seed {row['seed']}")
        activation_path = Path(row["activation_path"])
        activations = np.load(activation_path, mmap_mode="r")
        if activations.shape != (100_000, 768):
            raise RuntimeError(f"unexpected reusable activation shape: {activation_path} {activations.shape}")
        audit_rows.append(
            {
                "task_index": int(row["task_index"]),
                "model": row["model"],
                "seed": int(row["seed"]),
                "checkpoint": str(checkpoint_path),
                "metrics": str(metrics_path),
                "activation_path": str(activation_path),
                "validation_reconstruction_r2": metrics["validation"]["reconstruction_r2"],
                "validation_dead_fraction": metrics["validation"]["dead_fraction"],
                "validation_live_features": gate["validation_live_features"],
                "capacity_gate_pass": gate["pass"],
            }
        )

    readout_summaries = []
    for path in sorted((args.source_root / "readouts").glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol") == "mimic_final_layer_matched_effect_100k_v1":
            readout_summaries.append(payload)
    if len(readout_summaries) != len(MODEL_SPECS):
        raise RuntimeError(f"expected six reusable readouts, found {len(readout_summaries)}")

    args.out.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.out / "input_reuse_audit.csv", audit_rows)
    protocol = {
        "status": "prepared",
        "protocol": PROTOCOL,
        "source_protocol": "mimic_final_layer_matched_effect_100k_v1",
        "records": 100_000,
        "models": [row[0] for row in MODEL_SPECS],
        "sae_cells_reused": len(manifest),
        "readouts_reused": len(readout_summaries),
        "candidate_pool": {
            "definition": "atoms with at least one positive code on the training split",
            "uses_concept_labels": False,
            "matched_candidates": 768,
            "subsets_per_seed": 20,
            "subset_seed_base": 976000,
        },
        "quality_gate": {
            "mode": "matched_live_capacity",
            "min_validation_reconstruction_r2": args.min_validation_r2,
            "min_validation_live_features": args.min_validation_live_features,
            "dead_fraction": "reported descriptively, not used as an exclusion threshold",
            "all_cells_pass_input_audit": True,
        },
        "unchanged_design": {
            "concepts": 7,
            "k": 5,
            "effect_cap": 0.25,
            "effect_floor": 0.05,
            "max_alpha": 1.0,
            "sae_seeds": list(SEEDS),
            "bootstrap_draws": 2000,
            "patient_disjoint_test": True,
        },
        "data_policy": "reuse existing derived activations, checkpoints, metrics, and readouts without modifying them",
    }
    atomic_json(args.out / "protocol.json", protocol)
    print(json.dumps(protocol, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
