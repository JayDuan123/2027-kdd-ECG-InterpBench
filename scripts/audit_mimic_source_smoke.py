#!/usr/bin/env python
"""Require complete six-model activation and SAE smoke coverage."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "results/mimic_source_benchmark_100k_v1_smoke"


def main() -> None:
    with (SMOKE / "training_manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = []
    if len(rows) != 6 or len({row["model"] for row in rows}) != 6:
        errors.append(f"expected six model smoke cells, found {len(rows)}")
    for row in rows:
        path = Path(row["metrics"])
        if not path.exists():
            errors.append(f"missing metrics: {path}")
            continue
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or int(payload.get("split_counts", {}).get("test", 0)) == 0:
            errors.append(f"incomplete smoke cell: {path}")
    protocol = json.loads((SMOKE / "protocol.json").read_text())
    if protocol.get("model_depth_cells") != 30 or protocol.get("records") != 128:
        errors.append(f"wrong materialization coverage: {protocol}")
    report = {"status": "pass" if not errors else "fail", "audit_pass": not errors, "errors": errors}
    (SMOKE / "smoke_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
