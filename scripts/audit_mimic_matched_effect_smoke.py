#!/usr/bin/env python
"""Verify the end-to-end MIMIC matched-effect smoke chain."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import PROTOCOL  # noqa: E402


BASE = ROOT / "results/mimic_final_layer_matched_effect_v1"


def complete_summaries(root: Path) -> list[dict[str, object]]:
    values = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol") == PROTOCOL:
            values.append(payload)
    return values


def main() -> None:
    protocol = json.loads((BASE / "protocol.json").read_text())
    if protocol.get("status") != "prepared" or protocol.get("records") != 4096:
        raise RuntimeError("derived MIMIC protocol is not prepared")
    readouts = complete_summaries(BASE / "smoke/readouts")
    workers = complete_summaries(BASE / "smoke/workers")
    if len(readouts) != 1 or len(workers) != 1:
        raise RuntimeError(f"expected one smoke readout/worker, found {len(readouts)}/{len(workers)}")
    metrics = sorted((BASE / "smoke/checkpoints").glob("**/metrics.json"))
    if len(metrics) != 1:
        raise RuntimeError(f"expected one smoke training metric, found {len(metrics)}")
    training = json.loads(metrics[0].read_text())
    if training.get("status") != "complete" or not training.get("smoke"):
        raise RuntimeError("smoke SAE training did not complete")
    worker = workers[0]
    if worker.get("common_effect_methods") != ["dense", "sae"]:
        raise RuntimeError("smoke common-effect method set is not Dense/SAE only")
    with Path(worker["design_cells"]).open(newline="") as handle:
        design = list(csv.DictReader(handle))
    if len(design) != 20 * 7 * 2:
        raise RuntimeError(f"expected 280 smoke design rows, found {len(design)}")
    with np.load(Path(worker["patient_metrics"]), allow_pickle=False) as payload:
        expected = {
            "arm_names": ["matched_768"],
            "ks": [5],
            "method_names": ["dense", "sae"],
        }
        for key, values in expected.items():
            if np.asarray(payload[key]).astype(str).tolist() != [str(value) for value in values]:
                raise RuntimeError(f"smoke archive mismatch for {key}")
        if payload["concept_names"].shape != (7,):
            raise RuntimeError("smoke archive does not contain seven concepts")
    audit = {
        "status": "pass",
        "protocol": PROTOCOL,
        "records": protocol["records"],
        "readout_cells": 1,
        "training_cells": 1,
        "worker_cells": 1,
        "design_rows": len(design),
        "methods": ["dense", "sae"],
        "concepts": 7,
        "quality_gate_required_for_smoke": False,
    }
    output = BASE / "smoke/audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
