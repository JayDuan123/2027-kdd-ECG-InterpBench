#!/usr/bin/env python
"""Build the four-model final-layer extraction plan for the 100k MIMIC protocol."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import MODEL_SPECS, normalize_record_id, read_csv  # noqa: E402
from scripts.plan_external_activation_extraction import command, index_command  # noqa: E402


EXTRACT_MODELS = {"CARDIAC-FM", "CSFM", "ECG-JEPA", "ST-MEM"}
CANONICAL = {
    "CARDIAC-FM": "cardiac_fm",
    "CSFM": "csfm",
    "ECG-JEPA": "ecg_jepa",
    "ST-MEM": "st_mem",
}
DEFAULT_MANIFEST = ROOT / "results/activations_external_full_v1/plan_mimic_100k/mimic_main_manifest.csv"
DEFAULT_PLAN = ROOT / "results/activations_external_full_v1/plan_mimic_final_layer_100k_v1"
DEFAULT_OUT = ROOT / "results/activations_external_full_v1/final_layer_100k_v1"
DEFAULT_SMOKE_OUT = ROOT / "results/activations_external_full_v1/final_layer_100k_v1_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--activation-out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke-out-dir", type=Path, default=DEFAULT_SMOKE_OUT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def write_lines(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = [row for row in read_csv(args.manifest) if row.get("status") == "ok"]
    if len(rows) != 100_000:
        raise RuntimeError(f"expected exactly 100000 usable MIMIC records, found {len(rows)}")
    ids = [normalize_record_id(row["record_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("100k source manifest contains duplicate record IDs")
    old_records = read_csv(
        ROOT / "results/activations_external_full_v1/layer_atlas/cardiac_fm_cu118_commons/mimic_f/records.csv"
    )
    old_ids = {normalize_record_id(row["ecg_id"]) for row in old_records}
    overlap_positions = [index for index, record_id in enumerate(ids) if record_id in old_ids]
    if not overlap_positions:
        raise RuntimeError("100k manifest has no overlap with the audited 4k layer atlas")
    smoke_offset = (overlap_positions[0] // args.batch_size) * args.batch_size
    smoke_limit = min(args.batch_size, len(rows) - smoke_offset)

    commands: list[str] = []
    smoke_commands: list[str] = []
    index_commands: list[str] = []
    summary_rows: list[dict[str, object]] = []
    for model, suffix, layer, _n_layers in MODEL_SPECS:
        if model not in EXTRACT_MODELS:
            continue
        canonical = CANONICAL[model]
        for offset in range(0, len(rows), args.batch_size):
            limit = min(args.batch_size, len(rows) - offset)
            commands.append(
                command(canonical, "mimic", offset, limit, str(layer), args.activation_out_dir, args.manifest, args.device)
                + " --pool-layer-activations"
            )
        smoke_commands.append(
            command(canonical, "mimic", smoke_offset, smoke_limit, str(layer), args.smoke_out_dir, args.manifest, args.device)
            + " --pool-layer-activations"
        )
        index_commands.append(index_command(canonical, "mimic", args.activation_out_dir))
        summary_rows.append(
            {
                "model": model,
                "model_suffix": suffix,
                "final_layer": layer,
                "records": len(rows),
                "batch_size": args.batch_size,
                "shards": math.ceil(len(rows) / args.batch_size),
                "commands": math.ceil(len(rows) / args.batch_size),
            }
        )

    expected_commands = len(EXTRACT_MODELS) * math.ceil(len(rows) / args.batch_size)
    if len(commands) != expected_commands or len(smoke_commands) != len(EXTRACT_MODELS):
        raise RuntimeError("extraction command coverage audit failed")
    args.plan_dir.mkdir(parents=True, exist_ok=True)
    write_lines(args.plan_dir / "all_commands.txt", commands)
    write_lines(args.plan_dir / "smoke_commands.txt", smoke_commands)
    write_lines(args.plan_dir / "index_commands.txt", index_commands)
    with (args.plan_dir / "plan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "status": "planned",
        "source_manifest": str(args.manifest),
        "records": len(rows),
        "unique_patients": len({row["subject_id"] for row in rows}),
        "models_extracted": [row["model"] for row in summary_rows],
        "models_reused_after_equivalence_audit": ["ECG-FM", "HuBERT-ECG"],
        "batch_size": args.batch_size,
        "extraction_commands": len(commands),
        "index_commands": len(index_commands),
        "smoke_offset": smoke_offset,
        "smoke_limit": smoke_limit,
        "smoke_overlap_with_4k": sum(
            record_id in old_ids for record_id in ids[smoke_offset : smoke_offset + smoke_limit]
        ),
        "activation_out_dir": str(args.activation_out_dir),
        "smoke_out_dir": str(args.smoke_out_dir),
    }
    (args.plan_dir / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
