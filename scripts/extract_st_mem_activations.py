#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV, read_split_ids, record_name_for_ecg_id
from benchmark_v1.adapters.st_mem import (
    ST_MEM_CHECKPOINT,
    ST_MEM_DEPTH,
    ST_MEM_LEADS,
    ST_MEM_ROOT,
    ST_MEM_TARGET_SAMPLES,
    extract_activations,
    parse_layer_spec,
    prepare_inputs,
    try_load_model,
)
from benchmark_v1.config import ROOT


DEFAULT_OUT_DIR = ROOT / "results" / "activations" / "st_mem_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ST-MEM PTB-XL activation extraction.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-name", default="")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def write_record_ids(path: Path, ecg_ids: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ecg_id", "record_name"])
        writer.writeheader()
        for ecg_id in ecg_ids:
            writer.writerow({"ecg_id": ecg_id, "record_name": record_name_for_ecg_id(ecg_id)})


def main() -> None:
    args = parse_args()
    run_out_dir = args.out_dir / args.shard_name if args.shard_name else args.out_dir
    run_out_dir.mkdir(parents=True, exist_ok=True)
    ecg_ids = read_split_ids(args.split, args.limit, DEFAULT_SPLIT_CSV, offset=args.offset)
    write_record_ids(run_out_dir / "record_ids.csv", ecg_ids)
    batch, meta = prepare_inputs(ecg_ids)

    if args.skip_model_load:
        model_status = "skipped by --skip-model-load"
    else:
        _, model_status = try_load_model(device=args.device)

    payload = {
        "model": "ST-MEM-ViT-Base",
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "shard_name": args.shard_name,
        "selected_records": len(ecg_ids),
        "target_shape_per_record": [len(ST_MEM_LEADS), ST_MEM_TARGET_SAMPLES],
        "batch_shape": meta.get("batch_shape"),
        "missing_dependencies": meta.get("missing_dependencies", []),
        "model_status": model_status,
    }
    (run_out_dir / "inputs_shape.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    activation_files: list[str] = []
    if args.save_activations and batch is not None and not args.skip_model_load:
        import numpy as np

        layers = parse_layer_spec(args.layers, depth=ST_MEM_DEPTH)
        activations = extract_activations(batch, layers, device=args.device)
        np.save(run_out_dir / "pooled.npy", activations["pooled"])
        activation_files.append("pooled.npy")
        layer_shapes = {}
        for layer_idx, values in sorted(activations["layers"].items()):
            filename = f"layer_{layer_idx:02d}.npy"
            np.save(run_out_dir / filename, values)
            activation_files.append(filename)
            layer_shapes[str(layer_idx)] = list(values.shape)
        activation_meta = {
            "model": "ST-MEM-ViT-Base",
            "model_status": activations["model_status"],
            "split": args.split,
            "offset": args.offset,
            "limit": args.limit,
            "shard_name": args.shard_name,
            "device": args.device,
            "ecg_ids": ecg_ids,
            "input_shape": list(batch.shape),
            "pooled_shape": list(activations["pooled"].shape),
            "layers": layers,
            "layer_shapes": layer_shapes,
            "pooled_file": "pooled.npy",
            "layer_file_template": "layer_{layer:02d}.npy",
        }
        (run_out_dir / "activation_metadata.json").write_text(
            json.dumps(activation_meta, indent=2) + "\n",
            encoding="utf-8",
        )
        activation_files.append("activation_metadata.json")

    report = [
        "# ST-MEM Adapter Smoke Report",
        "",
        f"- split: {args.split}",
        f"- offset: {args.offset}",
        f"- limit: {args.limit}",
        f"- selected records: {len(ecg_ids)}",
        f"- input leads: {', '.join(ST_MEM_LEADS)}",
        f"- target shape per record: {len(ST_MEM_LEADS)} x {ST_MEM_TARGET_SAMPLES}",
        f"- ST-MEM root: {ST_MEM_ROOT}",
        f"- checkpoint: {ST_MEM_CHECKPOINT}",
        f"- device: {args.device}",
        f"- batch shape: {meta.get('batch_shape')}",
        f"- model status: {model_status}",
        f"- activation files: {', '.join(activation_files) if activation_files else 'none'}",
        "",
    ]
    (run_out_dir / "adapter_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {run_out_dir / 'adapter_report.md'}")


if __name__ == "__main__":
    main()
