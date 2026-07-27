#!/usr/bin/env python3
"""Merge complete harmonized activation shards into final-layer feature matrices."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV  # noqa: E402
from benchmark_v1.input_harmonization import MODEL_INTERFACES, PROTOCOLS, final_layer_for_model  # noqa: E402


SPLIT_ORDER = {"train": 0, "val": 1, "test": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activation-root",
        type=Path,
        default=ROOT / "results/input_harmonization_v1/activations",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/input_harmonization_v1/probe_features",
    )
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    return parser.parse_args()


def expected_records(path: Path) -> list[dict[str, str]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("ecg_id") and row.get("split") in SPLIT_ORDER:
                rows.append(dict(row))
    rows.sort(key=lambda row: (SPLIT_ORDER[row["split"]], int(row["ecg_id"])))
    return rows


def read_ids(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        return [row["ecg_id"] for row in csv.DictReader(handle)]


def merge_cell(
    model: str,
    protocol: str,
    activation_root: Path,
    output_root: Path,
    expected: list[dict[str, str]],
) -> dict[str, object]:
    layer = final_layer_for_model(model)
    source = activation_root / protocol / model
    metadata_paths = list(source.glob("*/activation_metadata.json"))
    shards = []
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            raise RuntimeError(f"incomplete shard: {metadata_path}")
        shards.append((SPLIT_ORDER[str(metadata["split"])], int(metadata["offset"]), metadata_path, metadata))
    shards.sort(key=lambda item: (item[0], item[1]))
    if not shards:
        raise RuntimeError(f"no complete shards found: {source}")

    values_by_id: dict[str, np.ndarray] = {}
    for _, _, metadata_path, metadata in shards:
        shard = metadata_path.parent
        ids = read_ids(shard / "record_ids.csv")
        values = np.load(shard / f"layer_{layer:02d}.npy", mmap_mode="r")
        if values.shape != (len(ids), 768):
            raise RuntimeError(f"shape/record mismatch in {shard}: {values.shape} vs {len(ids)}")
        for index, ecg_id in enumerate(ids):
            if ecg_id in values_by_id:
                raise RuntimeError(f"duplicate record {ecg_id} in {model}/{protocol}")
            values_by_id[ecg_id] = np.asarray(values[index], dtype=np.float32)

    expected_ids = [row["ecg_id"] for row in expected]
    missing = [ecg_id for ecg_id in expected_ids if ecg_id not in values_by_id]
    extra = sorted(set(values_by_id) - set(expected_ids), key=int)
    if missing or extra:
        raise RuntimeError(f"record coverage failure {model}/{protocol}: missing={len(missing)}, extra={len(extra)}")

    output = output_root / protocol / model
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / f"layer_{layer:02d}_mean.npy"
    matrix = np.lib.format.open_memmap(feature_path, mode="w+", dtype="float32", shape=(len(expected), 768))
    for index, ecg_id in enumerate(expected_ids):
        matrix[index] = values_by_id[ecg_id]
    matrix.flush()
    with (output / "records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ecg_id", "patient_id", "split", "model", "protocol"))
        writer.writeheader()
        for row in expected:
            writer.writerow(
                {
                    "ecg_id": row["ecg_id"],
                    "patient_id": row.get("patient_id", ""),
                    "split": row["split"],
                    "model": model,
                    "protocol": protocol,
                }
            )
    with (output / "features.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("feature", "file", "shape", "aggregation"))
        writer.writeheader()
        writer.writerow(
            {
                "feature": f"layer_{layer:02d}_mean",
                "file": str(feature_path.relative_to(ROOT)),
                "shape": json.dumps([len(expected), 768]),
                "aggregation": "token_mean",
            }
        )
    report = {
        "model": model,
        "protocol": protocol,
        "layer": layer,
        "records": len(expected),
        "shards": len(shards),
        "feature": str(feature_path),
        "finite": bool(np.isfinite(np.load(feature_path, mmap_mode="r")).all()),
    }
    if not report["finite"]:
        raise RuntimeError(f"non-finite merged features: {model}/{protocol}")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    expected = expected_records(args.split_csv)
    reports = []
    for protocol in PROTOCOLS[1:]:
        for model in sorted(MODEL_INTERFACES):
            reports.append(merge_cell(model, protocol, args.activation_root, args.output_root, expected))
    summary = {"status": "complete", "records": len(expected), "cells": len(reports), "reports": reports}
    destination = args.output_root.parent / "materialization_audit.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
