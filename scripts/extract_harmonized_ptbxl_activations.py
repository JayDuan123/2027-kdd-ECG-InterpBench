#!/usr/bin/env python3
"""Extract one model's final-layer PTB-XL activations under an input protocol."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV, read_split_ids, record_name_for_ecg_id  # noqa: E402
from benchmark_v1.input_harmonization import (  # noqa: E402
    MODEL_INTERFACES,
    PROTOCOLS,
    canonical_model_name,
    final_layer_for_model,
    prepare_model_batch,
)


MODEL_DISPLAY = {
    "cardiac_fm": "CARDIAC-FM",
    "csfm": "CSFM",
    "ecg_fm": "ECG-FM",
    "ecg_jepa": "ECG-JEPA",
    "hubert_ecg": "HuBERT-ECG",
    "st_mem": "ST-MEM",
}


def model_runtime(model: str):
    model = canonical_model_name(model)
    if model == "cardiac_fm":
        from benchmark_v1.adapters.cardiac_fm import extract_activations

        return extract_activations
    if model == "csfm":
        from benchmark_v1.adapters.csfm import extract_activations

        return extract_activations
    if model == "ecg_fm":
        from benchmark_v1.adapters.ecg_fm import extract_activations

        return extract_activations
    if model == "ecg_jepa":
        from benchmark_v1.adapters.ecg_jepa import extract_encoder_activations

        return extract_encoder_activations
    if model == "hubert_ecg":
        from benchmark_v1.adapters.hubert_ecg import extract_activations

        return extract_activations
    if model == "st_mem":
        from benchmark_v1.adapters.st_mem import extract_activations

        return extract_activations
    raise AssertionError(model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_INTERFACES))
    parser.add_argument("--protocol", required=True, choices=PROTOCOLS)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shard-name", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_record_ids(path: Path, ids: list[str], split: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ecg_id", "record_name", "split"))
        writer.writeheader()
        for ecg_id in ids:
            writer.writerow({"ecg_id": ecg_id, "record_name": record_name_for_ecg_id(ecg_id), "split": split})
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    model = canonical_model_name(args.model)
    layer = final_layer_for_model(model)
    shard_name = args.shard_name or f"{args.split}_offset{args.offset:06d}_n{args.limit:04d}"
    output = args.out_dir / shard_name
    output.mkdir(parents=True, exist_ok=True)
    done = output / "activation_metadata.json"
    if done.exists() and not args.skip_model_load:
        payload = json.loads(done.read_text(encoding="utf-8"))
        layer_path = output / f"layer_{layer:02d}.npy"
        if payload.get("status") == "complete" and layer_path.exists():
            print(json.dumps(payload, indent=2, sort_keys=True))
            return

    ecg_ids = read_split_ids(args.split, args.limit, args.split_csv, offset=args.offset)
    if not ecg_ids:
        raise RuntimeError(f"no records selected for {args.split} offset={args.offset} limit={args.limit}")
    batch, input_metadata = prepare_model_batch(ecg_ids, model, args.protocol)
    if not np.isfinite(batch).all():
        raise RuntimeError("prepared input contains non-finite values")
    write_record_ids(output / "record_ids.csv", ecg_ids, args.split)

    base = {
        "status": "input_only" if args.skip_model_load else "running",
        "model": MODEL_DISPLAY[model],
        "model_key": model,
        "protocol": args.protocol,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "loaded_records": len(ecg_ids),
        "input_metadata": input_metadata,
        "input_shape": list(batch.shape),
        "final_layer": layer,
        "layers": [layer],
        "layer_aggregation": "token_mean",
        "pooled_file": "pooled.npy",
        "layer_file_template": "layer_{layer:02d}.npy",
    }
    atomic_json(output / "inputs_shape.json", base)
    if args.skip_model_load:
        print(json.dumps(base, indent=2, sort_keys=True))
        return

    extract = model_runtime(model)
    activations = extract(batch, [layer], device=args.device)
    pooled = np.asarray(activations["pooled"], dtype=np.float32)
    final = np.asarray(activations["layers"][layer], dtype=np.float32)
    if final.ndim == 3:
        final = final.mean(axis=1, dtype=np.float32)
    elif final.ndim != 2:
        raise RuntimeError(f"unexpected final-layer shape: {final.shape}")
    if final.shape[0] != len(ecg_ids) or final.shape[1] != 768:
        raise RuntimeError(f"unexpected pooled final-layer shape: {final.shape}")
    if not np.isfinite(final).all() or not np.isfinite(pooled).all():
        raise RuntimeError("model output contains non-finite values")
    atomic_npy(output / "pooled.npy", pooled)
    atomic_npy(output / f"layer_{layer:02d}.npy", final)
    payload = {
        **base,
        "status": "complete",
        "device": args.device,
        "model_status": activations.get("model_status", ""),
        "pooled_shape": list(pooled.shape),
        "layer_shapes": {str(layer): list(final.shape)},
        "layer_values_prepooled": True,
    }
    atomic_json(done, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
