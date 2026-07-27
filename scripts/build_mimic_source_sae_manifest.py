#!/usr/bin/env python
"""Build the matched multi-scale SAE grid from the MIMIC layer catalog."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multiscale_sae import LayerSpec, build_manifest_rows_from_specs, read_csv  # noqa: E402


def numbers(value: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/mimic_source_benchmark_100k_v1")
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--expansions", default="1,4,8,16,32")
    parser.add_argument("--seeds", default="4311,4312,4313")
    parser.add_argument("--depths", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    catalog_path = args.catalog or args.root / "derived/layer_catalog.csv"
    requested_depths = set(numbers(args.depths, float))
    specs = []
    for row in read_csv(catalog_path):
        if float(row["relative_depth"]) not in requested_depths:
            continue
        specs.append(
            LayerSpec(
                model=row["model"],
                suffix=row["feature_suffix"],
                layer=int(row["layer"]),
                target_relative_depth=float(row["relative_depth"]),
                actual_relative_depth=float(row["actual_relative_depth"]),
                n_layers=int(row["n_layers"]),
                d_hidden=int(row["d_hidden"]),
                activation_path=Path(row["activation_path"]),
                records_path=Path(row["records_path"]),
            )
        )
    rows = build_manifest_rows_from_specs(
        specs,
        args.root,
        expansions=numbers(args.expansions, int),
        seeds=numbers(args.seeds, int),
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    expected = 6 * len(requested_depths) * len(numbers(args.expansions, int)) * len(numbers(args.seeds, int))
    if len(rows) != expected:
        raise RuntimeError(f"incomplete MIMIC SAE grid: rows={len(rows)}, expected={expected}")
    atomic_csv(args.root / "training_manifest.csv", rows)
    summary = {
        "status": "planned",
        "models": 6,
        "depths": sorted(requested_depths),
        "expansions": numbers(args.expansions, int),
        "seeds": numbers(args.seeds, int),
        "training_cells": len(rows),
        "steps": args.steps,
    }
    (args.root / "sae_plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    protocol_path = args.root / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    protocol.update(
        {
            "models": sorted({row["model"] for row in rows}),
            "relative_depths": sorted(requested_depths),
            "expansion_E": list(numbers(args.expansions, int)),
            "seeds": list(numbers(args.seeds, int)),
            "training_cells": len(rows),
            "training_steps": args.steps,
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
