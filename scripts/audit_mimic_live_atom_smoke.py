#!/usr/bin/env python
"""Audit the HuBERT live-atom matched-768 smoke worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import read_csv  # noqa: E402
from scripts.prepare_mimic_final_layer_matched_effect import atomic_json  # noqa: E402


PROTOCOL = "mimic_final_layer_live_atom_matched_effect_100k_v1"
DEFAULT_ROOT = ROOT / "results/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke/workers"
DEFAULT_OUT = ROOT / "results/mimic_final_layer_live_atom_matched_effect_100k_v1/smoke/audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = sorted(args.workers_root.glob("*/summary.json"))
    if len(summaries) != 1:
        raise RuntimeError(f"expected one smoke summary, found {len(summaries)}")
    summary = json.loads(summaries[0].read_text())
    required = {
        "status": "complete",
        "protocol": PROTOCOL,
        "model": "HuBERT-ECG",
        "candidate_pool": "train_live",
    }
    for key, expected in required.items():
        if summary.get(key) != expected:
            raise RuntimeError(f"smoke {key} mismatch: {summary.get(key)!r} != {expected!r}")
    gate = summary["sae_quality_gate"]
    if gate.get("mode") != "matched_live_capacity" or not gate.get("pass"):
        raise RuntimeError(f"smoke capacity gate failed: {gate}")
    if int(summary["candidate_pool_features"]) < 768 or int(summary["train_live_features"]) < 768:
        raise RuntimeError("smoke training-live candidate pool is smaller than 768")
    if int(gate["validation_live_features"]) < 768:
        raise RuntimeError("smoke validation live capacity is smaller than 768")

    archive_path = Path(summary["patient_metrics"])
    design_path = Path(summary["design_cells"])
    with np.load(archive_path, allow_pickle=False) as payload:
        arm_names = payload["arm_names"].astype(str).tolist()
        live = set(payload["train_live_feature_indices"].astype(int).tolist())
        if arm_names != ["live_atom_matched_768"] or len(live) < 768:
            raise RuntimeError("smoke archive live-atom axes are invalid")
    design = read_csv(design_path)
    if len(design) != 280:
        raise RuntimeError(f"expected 280 smoke design rows, found {len(design)}")
    for row in design:
        if row["candidate_arm"] != "live_atom_matched_768" or row["candidate_pool"] != "train_live":
            raise RuntimeError("smoke design row candidate metadata mismatch")
        if row["method"] == "sae":
            selected = {int(value) for value in row["selected_features"].split(";") if value}
            if not selected.issubset(live):
                raise RuntimeError("smoke selected an SAE atom outside the training-live pool")
    payload = {
        "status": "pass",
        "protocol": PROTOCOL,
        "model": summary["model"],
        "candidate_pool_features": summary["candidate_pool_features"],
        "train_live_features": summary["train_live_features"],
        "validation_live_features": gate["validation_live_features"],
        "design_rows": len(design),
        "eligible_design_rows": summary["eligible_design_rows"],
        "archive": str(archive_path),
    }
    atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
