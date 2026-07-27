#!/usr/bin/env python
"""Plan all-depth activation extraction for the frozen 100k MIMIC benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import MODEL_SPECS  # noqa: E402
from benchmark_v1.mimic_source_benchmark import (  # noqa: E402
    ACTIVATION_ROOT,
    CANONICAL_MODEL,
    PLAN_ROOT,
    SOURCE_MANIFEST,
    expected_extraction_commands,
    selected_layers,
    source_rows,
)
from scripts.plan_external_activation_extraction import command, index_command  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / SOURCE_MANIFEST)
    parser.add_argument("--plan-root", type=Path, default=ROOT / PLAN_ROOT)
    parser.add_argument("--activation-root", type=Path, default=ROOT / ACTIVATION_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--smoke-records", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""))


def main() -> None:
    args = parse_args()
    rows = source_rows(args.manifest)
    if len(rows) != 100_000:
        raise RuntimeError(f"expected 100000 MIMIC records, found {len(rows)}")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate MIMIC record IDs")
    commands: list[str] = []
    smoke_commands: list[str] = []
    index_commands: list[str] = []
    smoke_index_commands: list[str] = []
    summary_rows = []
    smoke_root = args.activation_root.with_name(args.activation_root.name + "_smoke")
    for model, suffix, _final_layer, n_layers in MODEL_SPECS:
        canonical = CANONICAL_MODEL[model]
        layers = ",".join(map(str, selected_layers(n_layers)))
        for offset in range(0, len(rows), args.batch_size):
            limit = min(args.batch_size, len(rows) - offset)
            commands.append(
                command(
                    canonical, "mimic", offset, limit, layers,
                    args.activation_root, args.manifest, args.device,
                ) + " --pool-layer-activations"
            )
        smoke_commands.append(
            command(
                canonical, "mimic", 0, args.smoke_records, layers,
                smoke_root, args.manifest, args.device,
            ) + " --pool-layer-activations"
        )
        index_commands.append(index_command(canonical, "mimic", args.activation_root))
        smoke_index_commands.append(index_command(canonical, "mimic", smoke_root))
        summary_rows.append(
            {
                "model": model,
                "model_suffix": suffix,
                "layers": layers,
                "records": len(rows),
                "shards": math.ceil(len(rows) / args.batch_size),
            }
        )
    expected = expected_extraction_commands(len(rows), args.batch_size)
    if len(commands) != expected or len(smoke_commands) != 6:
        raise RuntimeError("activation command coverage audit failed")
    args.plan_root.mkdir(parents=True, exist_ok=True)
    write_lines(args.plan_root / "all_commands.txt", commands)
    write_lines(args.plan_root / "smoke_commands.txt", smoke_commands)
    write_lines(args.plan_root / "index_commands.txt", index_commands)
    write_lines(args.plan_root / "smoke_index_commands.txt", smoke_index_commands)
    with (args.plan_root / "plan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "status": "planned",
        "records": len(rows),
        "patients": len({row["subject_id"] for row in rows}),
        "models": 6,
        "depths_per_model": 5,
        "commands": len(commands),
        "smoke_commands": len(smoke_commands),
        "batch_size": args.batch_size,
        "activation_root": str(args.activation_root),
        "smoke_activation_root": str(smoke_root),
    }
    (args.plan_root / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
