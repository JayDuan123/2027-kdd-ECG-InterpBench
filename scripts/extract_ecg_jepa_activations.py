#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.adapters.ecg_jepa import (
    DEFAULT_SPLIT_CSV,
    ECG_JEPA_CHECKPOINT,
    ECG_JEPA_LEADS,
    ECG_JEPA_ROOT,
    ECG_JEPA_TARGET_SAMPLES,
    PTBXL_WAVEFORM_ROOT,
    dependency_status,
    extract_encoder_activations,
    parse_layer_spec,
    prepare_smoke_inputs,
    read_split_ids,
    record_name_for_ecg_id,
    try_load_encoder,
)
from benchmark_v1.config import ROOT


DEFAULT_OUT_DIR = ROOT / "results" / "activations" / "ecg_jepa_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ECG-JEPA PTB-XL input adapter smoke check.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-name", default="")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument("--layers", default="all", help="Comma-separated layer ids or 'all'.")
    parser.add_argument("--device", default="cpu", help="Torch device for model inference, e.g. cpu or cuda.")
    return parser.parse_args()


def write_record_ids(path: Path, ecg_ids: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ecg_id", "record_name"])
        writer.writeheader()
        for ecg_id in ecg_ids:
            writer.writerow({"ecg_id": ecg_id, "record_name": record_name_for_ecg_id(ecg_id)})


def render_report(args: argparse.Namespace, ecg_ids: list[str], meta: dict, model_status: str) -> str:
    deps = dependency_status()
    lines = [
        "# ECG-JEPA Adapter Smoke Report",
        "",
        "## Inputs",
        "",
        f"- split: {args.split}",
        f"- limit: {args.limit}",
        f"- offset: {args.offset}",
        f"- shard name: {args.shard_name or 'none'}",
        f"- selected records: {len(ecg_ids)}",
        f"- waveform root: {PTBXL_WAVEFORM_ROOT}",
        f"- ECG-JEPA root: {ECG_JEPA_ROOT}",
        f"- ECG-JEPA checkpoint: {ECG_JEPA_CHECKPOINT}",
        f"- device: {args.device}",
        "",
        "## Protocol",
        "",
        "- source waveform: PTB-XL Challenge-format 12-lead ECG",
        f"- ECG-JEPA input leads: {', '.join(ECG_JEPA_LEADS)}",
        f"- target shape per record: {len(ECG_JEPA_LEADS)} x {ECG_JEPA_TARGET_SAMPLES}",
        "- transformation: select reduced leads, resample to target length, per-lead z-score",
        "",
        "## Dependencies",
        "",
    ]
    for dep in deps:
        status = "available" if dep.available else "missing"
        lines.append(f"- {dep.name}: {status} ({dep.detail})")
    lines.extend(
        [
            "",
            "## Waveform Index",
            "",
            f"- requested records: {meta.get('requested_records')}",
            f"- found records: {meta.get('found_records')}",
        ]
    )
    if meta.get("missing_dependencies"):
        lines.append(f"- waveform tensor build: skipped, missing {', '.join(meta['missing_dependencies'])}")
    else:
        lines.append(f"- batch shape: {meta.get('batch_shape')}")
    lines.extend(["", "## Model Load", "", f"- status: {model_status}", ""])
    if args.save_activations:
        lines.extend(
            [
                "## Activation Save",
                "",
                f"- requested layers: {args.layers}",
                f"- status: {meta.get('activation_status', 'not recorded')}",
            ]
        )
        for path in meta.get("activation_files", []):
            lines.append(f"- file: {path}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_out_dir = args.out_dir / args.shard_name if args.shard_name else args.out_dir
    run_out_dir.mkdir(parents=True, exist_ok=True)
    ecg_ids = read_split_ids(args.split, args.limit, DEFAULT_SPLIT_CSV, offset=args.offset)
    write_record_ids(run_out_dir / "record_ids.csv", ecg_ids)
    batch, meta = prepare_smoke_inputs(ecg_ids)

    shape_payload = {
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "shard_name": args.shard_name,
        "selected_records": len(ecg_ids),
        "target_shape_per_record": [len(ECG_JEPA_LEADS), ECG_JEPA_TARGET_SAMPLES],
        "batch_shape": meta.get("batch_shape"),
        "missing_dependencies": meta.get("missing_dependencies", []),
    }
    if batch is not None:
        shape_payload["batch_shape"] = list(batch.shape)
    (run_out_dir / "inputs_shape.json").write_text(json.dumps(shape_payload, indent=2) + "\n", encoding="utf-8")

    if args.skip_model_load:
        model_status = "skipped by --skip-model-load"
    else:
        _, model_status = try_load_encoder(device=args.device)

    if args.save_activations:
        if args.skip_model_load:
            meta["activation_status"] = "skipped because --skip-model-load was set"
            meta["activation_files"] = []
        elif batch is None:
            meta["activation_status"] = "skipped because input batch was not built"
            meta["activation_files"] = []
        else:
            import numpy as np

            layers = parse_layer_spec(args.layers)
            activations = extract_encoder_activations(batch, layers, device=args.device)
            np.save(run_out_dir / "pooled.npy", activations["pooled"])
            activation_files = ["pooled.npy"]
            layer_shapes = {}
            for layer_idx, values in sorted(activations["layers"].items()):
                filename = f"layer_{layer_idx:02d}.npy"
                np.save(run_out_dir / filename, values)
                activation_files.append(filename)
                layer_shapes[str(layer_idx)] = list(values.shape)
            activation_meta = {
                "model": "ECG-JEPA",
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
            meta["activation_status"] = f"saved {len(layers)} layers"
            meta["activation_files"] = activation_files

    report = render_report(args, ecg_ids, meta, model_status)
    (run_out_dir / "adapter_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {run_out_dir / 'adapter_report.md'}")


if __name__ == "__main__":
    main()
