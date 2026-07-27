#!/usr/bin/env python
"""Audit the four-source smoke matrix for final-layer sparse accessibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROTOCOL = "final_layer_sparse_accessibility_e8_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=Path("results/final_layer_sparse_accessibility_e8_v2_smoke/workers"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/final_layer_sparse_accessibility_e8_v2_smoke/audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors = []
    summaries = []
    for path in sorted(args.workers_root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        summaries.append(payload)
        if payload.get("status") != "complete" or payload.get("protocol") != PROTOCOL:
            errors.append(f"incomplete or wrong protocol: {path}")
            continue
        if payload.get("source_kind") in {"sae", "random"} and payload.get(
            "semantic_encoding_stream"
        ) != "independent":
            errors.append(f"semantic subset was not independently encoded: {path}")
        metrics_path = Path(payload["metrics"])
        predictions_path = Path(payload["test_predictions"])
        if not metrics_path.exists() or not predictions_path.exists():
            errors.append(f"missing worker artifact: {path}")
            continue
        metrics = pd.read_csv(metrics_path)
        if set(metrics.k) != {1, 2, 4, 8, 16, 32} or metrics.concept.nunique() != 49:
            errors.append(f"metric support mismatch: {metrics_path}")
        if not np.isfinite(metrics[["validation_r", "test_r"]].to_numpy()).all():
            errors.append(f"nonfinite metric: {metrics_path}")
        with np.load(predictions_path, allow_pickle=False) as archive:
            predictions = np.asarray(archive["predictions"])
            if predictions.ndim != 4 or predictions.shape[1] != 6 or predictions.shape[3] != 49:
                errors.append(f"prediction shape mismatch: {predictions_path}")
            if not np.isfinite(predictions).all():
                errors.append(f"nonfinite prediction: {predictions_path}")
    kinds = sorted(str(payload.get("source_kind")) for payload in summaries)
    if kinds != ["dense", "pca", "random", "sae"]:
        errors.append(f"expected four source kinds, found {kinds}")
    audit = {
        "status": "complete" if not errors else "failed",
        "protocol": PROTOCOL,
        "workers": len(summaries),
        "source_kinds": kinds,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
