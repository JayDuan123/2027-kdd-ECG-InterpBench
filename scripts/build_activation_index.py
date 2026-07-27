#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV
from benchmark_v1.config import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a shard and record index for extracted activations.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--activation-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split-csv", default=DEFAULT_SPLIT_CSV, type=Path)
    return parser.parse_args()


def read_split_map(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return {row["ecg_id"]: row["split"] for row in csv.DictReader(f) if row.get("ecg_id")}


def read_record_ids(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return [row["ecg_id"] for row in csv.DictReader(f) if row.get("ecg_id")]


def rel(path: Path, base: Path) -> str:
    path = path.resolve()
    base = base.resolve()
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def layer_files(shard_dir: Path) -> list[str]:
    return sorted(path.name for path in shard_dir.glob("layer_*.npy"))


def build_index(model: str, activation_dir: Path, out_dir: Path, split_csv: Path) -> dict[str, Any]:
    activation_dir = activation_dir.resolve()
    out_dir = out_dir.resolve()
    split_csv = split_csv.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    split_by_id = read_split_map(split_csv)
    shard_rows: list[dict[str, str]] = []
    record_rows: list[dict[str, str]] = []
    missing_record_ids: list[str] = []
    split_counts = {"train": 0, "val": 0, "test": 0}

    metadata_paths = sorted(activation_dir.glob("*/activation_metadata.json"))
    for metadata_path in metadata_paths:
        shard_dir = metadata_path.parent
        with metadata_path.open() as f:
            meta = json.load(f)
        record_ids_path = shard_dir / "record_ids.csv"
        ecg_ids = read_record_ids(record_ids_path) if record_ids_path.exists() else [str(x) for x in meta.get("ecg_ids", [])]
        if not ecg_ids:
            missing_record_ids.append(shard_dir.name)
            continue

        layers = [str(x) for x in meta.get("layers", [])]
        files = layer_files(shard_dir)
        shard_rows.append(
            {
                "model": model,
                "shard_name": shard_dir.name,
                "split": str(meta.get("split", "")),
                "offset": str(meta.get("offset", "")),
                "limit": str(meta.get("limit", "")),
                "n_records": str(len(ecg_ids)),
                "activation_metadata": rel(metadata_path, ROOT),
                "record_ids_file": rel(record_ids_path, ROOT) if record_ids_path.exists() else "",
                "pooled_file": rel(shard_dir / str(meta.get("pooled_file", "pooled.npy")), ROOT),
                "layers": "|".join(layers),
                "layer_files": "|".join(files),
                "input_shape": json.dumps(meta.get("input_shape", [])),
                "pooled_shape": json.dumps(meta.get("pooled_shape", [])),
                "layer_shapes": json.dumps(meta.get("layer_shapes", {}), sort_keys=True),
            }
        )

        for row_idx, ecg_id in enumerate(ecg_ids):
            split = split_by_id.get(ecg_id, str(meta.get("split", "")))
            if split in split_counts:
                split_counts[split] += 1
            record_rows.append(
                {
                    "ecg_id": ecg_id,
                    "split": split,
                    "model": model,
                    "shard_name": shard_dir.name,
                    "row_in_shard": str(row_idx),
                    "pooled_file": rel(shard_dir / str(meta.get("pooled_file", "pooled.npy")), ROOT),
                    "layer_file_template": str(meta.get("layer_file_template", "layer_{layer:02d}.npy")),
                    "shard_dir": rel(shard_dir, ROOT),
                }
            )

    shard_fields = [
        "model",
        "shard_name",
        "split",
        "offset",
        "limit",
        "n_records",
        "activation_metadata",
        "record_ids_file",
        "pooled_file",
        "layers",
        "layer_files",
        "input_shape",
        "pooled_shape",
        "layer_shapes",
    ]
    record_fields = ["ecg_id", "split", "model", "shard_name", "row_in_shard", "pooled_file", "layer_file_template", "shard_dir"]
    with (out_dir / "shards.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=shard_fields)
        writer.writeheader()
        writer.writerows(shard_rows)
    with (out_dir / "records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=record_fields)
        writer.writeheader()
        writer.writerows(record_rows)

    report = {
        "model": model,
        "activation_dir": str(activation_dir),
        "n_shards": len(shard_rows),
        "n_records": len(record_rows),
        "split_counts": split_counts,
        "missing_record_ids": missing_record_ids,
        "outputs": {
            "shards": str(out_dir / "shards.csv"),
            "records": str(out_dir / "records.csv"),
        },
    }
    (out_dir / "index_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Activation Index: {model}",
        "",
        f"- activation dir: `{activation_dir}`",
        f"- shards: {len(shard_rows)}",
        f"- records: {len(record_rows)}",
        f"- train: {split_counts['train']}",
        f"- val: {split_counts['val']}",
        f"- test: {split_counts['test']}",
        f"- missing record id files: {len(missing_record_ids)}",
        "",
    ]
    (out_dir / "index_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = build_index(args.model, args.activation_dir, args.out_dir, args.split_csv)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
