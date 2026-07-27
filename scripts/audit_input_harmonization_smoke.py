#!/usr/bin/env python3
"""Gate the 24-cell input-harmonization GPU smoke extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.input_harmonization import MODEL_INTERFACES, PROTOCOLS, final_layer_for_model  # noqa: E402


INVARIANTS = {
    "ecg_jepa": (("native", "lead"), ("native", "temporal"), ("native", "joint")),
    "csfm": (("native", "temporal"), ("lead", "joint")),
    "hubert_ecg": (("native", "temporal"), ("lead", "joint")),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "results/input_harmonization_v1/smoke",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/input_harmonization_v1/smoke/audit.json",
    )
    return parser.parse_args()


def relative_error(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), 1.0)
    return float(np.max(np.abs(left - right)) / scale)


def main() -> None:
    args = parse_args()
    cells = []
    arrays: dict[tuple[str, str], np.ndarray] = {}
    errors: list[str] = []
    for model in sorted(MODEL_INTERFACES):
        layer = final_layer_for_model(model)
        for protocol in PROTOCOLS:
            shard = args.root / "activations" / protocol / model / "test_offset000000_n0002"
            metadata_path = shard / "activation_metadata.json"
            layer_path = shard / f"layer_{layer:02d}.npy"
            if not metadata_path.exists() or not layer_path.exists():
                errors.append(f"missing outputs: {model}/{protocol}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            values = np.load(layer_path)
            finite = bool(np.isfinite(values).all())
            shape_ok = values.shape == (2, 768)
            variance = float(np.var(values))
            status_ok = metadata.get("status") == "complete"
            if not (finite and shape_ok and variance > 0 and status_ok):
                errors.append(
                    f"invalid cell {model}/{protocol}: shape={values.shape}, finite={finite}, "
                    f"variance={variance}, status={metadata.get('status')}"
                )
            arrays[(model, protocol)] = values
            cells.append(
                {
                    "model": model,
                    "protocol": protocol,
                    "shape": list(values.shape),
                    "finite": finite,
                    "variance": variance,
                    "status": metadata.get("status"),
                }
            )

    invariants = []
    for model, pairs in INVARIANTS.items():
        for left_name, right_name in pairs:
            left = arrays.get((model, left_name))
            right = arrays.get((model, right_name))
            if left is None or right is None:
                continue
            error = relative_error(left, right)
            passed = error <= 1e-6
            invariants.append(
                {
                    "model": model,
                    "left": left_name,
                    "right": right_name,
                    "max_relative_error": error,
                    "passed": passed,
                }
            )
            if not passed:
                errors.append(f"invariance failed: {model} {left_name}/{right_name} error={error}")

    payload = {
        "status": "pass" if not errors else "fail",
        "expected_cells": len(MODEL_INTERFACES) * len(PROTOCOLS),
        "observed_cells": len(cells),
        "cells": cells,
        "invariants": invariants,
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = args.output.with_suffix(".md")
    report.write_text(
        "# Input Harmonization Smoke Audit\n\n"
        f"- status: **{payload['status']}**\n"
        f"- observed cells: {len(cells)}/{payload['expected_cells']}\n"
        f"- invariant checks: {sum(item['passed'] for item in invariants)}/{len(invariants)}\n"
        f"- errors: {len(errors)}\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
