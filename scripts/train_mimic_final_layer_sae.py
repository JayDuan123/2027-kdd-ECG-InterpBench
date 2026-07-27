#!/usr/bin/env python
"""Train and quality-audit one MIMIC final-layer BatchTopK SAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import PROTOCOL, read_csv  # noqa: E402
from benchmark_v1.multiscale_sae import canonical_config_hash  # noqa: E402
from scripts.run_multiscale_sae_task import (  # noqa: E402
    atomic_json,
    manifest_row,
    train,
    typed_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/training_manifest.csv",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--min-validation-r2", type=float, default=0.90)
    parser.add_argument("--max-validation-dead-fraction", type=float, default=0.20)
    return parser.parse_args()


def evaluate_quality(model, activations: np.ndarray, indices: np.ndarray, batch_size: int, device: str):
    import torch

    if not len(indices):
        raise ValueError("quality split is empty")
    split_mean = np.mean(np.asarray(activations[indices], dtype=np.float32), axis=0)
    mean_normalized = model.normalise(torch.from_numpy(split_mean).to(device)).detach()
    sse = 0.0
    sst = 0.0
    active = 0.0
    fired = np.zeros(model.N, dtype=bool)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(indices), batch_size):
            selected = indices[lo : lo + batch_size]
            raw = torch.as_tensor(
                np.asarray(activations[selected]), dtype=torch.float32, device=device
            )
            reconstruction, codes, normalized = model(raw)
            sse += float((normalized - reconstruction).square().sum().item())
            sst += float((normalized - mean_normalized).square().sum().item())
            positive = codes > 0
            active += float(positive.sum().item())
            fired |= positive.any(dim=0).cpu().numpy()
    return {
        "records": int(len(indices)),
        "reconstruction_r2": float(1.0 - sse / max(sst, 1e-12)),
        "dead_fraction": float(1.0 - fired.mean()),
        "mean_l0": float(active / len(indices)),
    }


def main() -> None:
    args = parse_args()
    config = typed_config(manifest_row(args.manifest, args.task_index))
    if canonical_config_hash(config) != config["config_hash"]:
        raise ValueError("manifest config hash mismatch")
    metrics_path = Path(config["metrics"])
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if existing.get("status") == "complete" and existing.get("config_hash") == config["config_hash"]:
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    activations = np.load(Path(config["activation_path"]), mmap_mode="r")
    records = read_csv(Path(config["records_path"]))
    if activations.shape != (len(records), int(config["d_hidden"])):
        raise RuntimeError("activation and record dimensions do not align")
    splits = np.asarray([row["split"] for row in records])
    indices = {name: np.flatnonzero(splits == name) for name in ("train", "val", "test")}
    if any(len(value) == 0 for value in indices.values()):
        raise RuntimeError("train/validation/test splits must all be nonempty")

    model = train(
        config,
        activations,
        indices["train"],
        args.device,
        args.checkpoint_every,
    )
    validation = evaluate_quality(
        model, activations, indices["val"], int(config["batch_size"]), args.device
    )
    test = evaluate_quality(
        model, activations, indices["test"], int(config["batch_size"]), args.device
    )
    quality_pass = bool(
        validation["reconstruction_r2"] >= args.min_validation_r2
        and validation["dead_fraction"] < args.max_validation_dead_fraction
    )
    payload = {
        "status": "complete",
        "protocol": args.protocol,
        "config_hash": config["config_hash"],
        "task_index": config["task_index"],
        "model": config["model"],
        "model_safe": config["model_safe"],
        "layer": config["layer"],
        "seed": config["seed"],
        "d_hidden": config["d_hidden"],
        "N": config["N"],
        "k": config["k"],
        "steps": config["steps"],
        "checkpoint": config["checkpoint"],
        "split_counts": {name: int(len(value)) for name, value in indices.items()},
        "validation": validation,
        "test": test,
        "quality_gate": {
            "min_validation_reconstruction_r2": args.min_validation_r2,
            "max_validation_dead_fraction_exclusive": args.max_validation_dead_fraction,
            "pass": quality_pass,
        },
        "smoke": int(config["steps"]) < 8000,
    }
    atomic_json(metrics_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
