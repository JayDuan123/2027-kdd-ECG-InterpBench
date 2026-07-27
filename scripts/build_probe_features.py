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

from benchmark_v1.config import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact pooled features for probe training.")
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", default="all", help="'all', 'pooled', or comma-separated integer layers.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_layers(spec: str, shard_rows: list[dict[str, str]]) -> list[str]:
    if spec == "pooled":
        return ["pooled"]
    available: set[str] = set()
    for row in shard_rows:
        for layer in row.get("layers", "").split("|"):
            if layer:
                available.add(layer)
    if spec == "all":
        return ["pooled"] + sorted(available, key=lambda x: int(x))
    requested = []
    for part in spec.split(","):
        part = part.strip()
        if part:
            requested.append(str(int(part)))
    missing = [layer for layer in requested if layer not in available]
    if missing:
        raise ValueError(f"requested layers not present in index: {missing}")
    return requested


def load_matrix(path: Path) -> Any:
    import numpy as np

    return np.load(path, mmap_mode="r")


def layer_filename(template: str, layer: str) -> str:
    return template.format(layer=int(layer))


def write_manifest(out_dir: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with (out_dir / "records.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.index_dir = args.index_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    shard_rows = read_csv(args.index_dir / "shards.csv")
    record_rows = read_csv(args.index_dir / "records.csv")
    layers = parse_layers(args.layers, shard_rows)
    rows_by_shard: dict[str, list[dict[str, str]]] = {}
    for row in record_rows:
        rows_by_shard.setdefault(row["shard_name"], []).append(row)

    import numpy as np

    n_records = len(record_rows)
    feature_manifest: list[dict[str, str]] = []
    for layer in layers:
        first_shard = next(row for row in shard_rows if row["shard_name"] in rows_by_shard)
        if layer == "pooled":
            first_path = ROOT / first_shard["pooled_file"]
            first = load_matrix(first_path)
            feature_dim = int(first.shape[1])
        else:
            first_record = rows_by_shard[first_shard["shard_name"]][0]
            template = first_record["layer_file_template"]
            first_path = ROOT / first_record["shard_dir"] / layer_filename(template, layer)
            first = load_matrix(first_path)
            feature_dim = int(first.shape[-1])

        out_path = args.out_dir / f"{'pooled' if layer == 'pooled' else f'layer_{int(layer):02d}_mean'}.npy"
        features = np.lib.format.open_memmap(out_path, mode="w+", dtype="float32", shape=(n_records, feature_dim))
        cursor = 0
        for shard in shard_rows:
            shard_name = shard["shard_name"]
            shard_record_rows = rows_by_shard.get(shard_name, [])
            if not shard_record_rows:
                continue
            if layer == "pooled":
                values = np.asarray(load_matrix(ROOT / shard["pooled_file"]), dtype=np.float32)
            else:
                template = shard_record_rows[0]["layer_file_template"]
                values = np.asarray(load_matrix(ROOT / shard_record_rows[0]["shard_dir"] / layer_filename(template, layer)), dtype=np.float32)
                if values.ndim == 3:
                    values = values.mean(axis=1, dtype=np.float32)
                elif values.ndim != 2:
                    raise ValueError(f"expected 2D or 3D layer values, got {values.shape}")
            n = len(shard_record_rows)
            features[cursor : cursor + n] = values[:n]
            cursor += n
        features.flush()
        feature_manifest.append(
            {
                "feature": "pooled" if layer == "pooled" else f"layer_{int(layer):02d}_mean",
                "file": str(out_path.relative_to(ROOT)),
                "shape": json.dumps([n_records, feature_dim]),
                "aggregation": "model_pooled" if layer == "pooled" else "token_mean",
            }
        )

    record_fields = ["ecg_id", "split", "model", "shard_name", "row_in_shard"]
    write_manifest(args.out_dir, [{field: row.get(field, "") for field in record_fields} for row in record_rows], record_fields)
    with (args.out_dir / "features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "file", "shape", "aggregation"])
        writer.writeheader()
        writer.writerows(feature_manifest)
    report = {
        "index_dir": str(args.index_dir),
        "out_dir": str(args.out_dir),
        "n_records": n_records,
        "features": feature_manifest,
    }
    (args.out_dir / "probe_features_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
