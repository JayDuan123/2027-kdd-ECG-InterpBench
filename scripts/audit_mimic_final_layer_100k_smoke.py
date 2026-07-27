#!/usr/bin/env python
"""Audit four final-layer extraction smoke shards against the existing 4k atlas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import MODEL_SPECS, load_final_layer, normalize_record_id, read_csv  # noqa: E402


EXTRACT_MODELS = {"CARDIAC-FM", "CSFM", "ECG-JEPA", "ST-MEM"}
DEFAULT_ROOT = ROOT / "results/activations_external_full_v1/final_layer_100k_v1_smoke"
DEFAULT_OUT = ROOT / "results/activations_external_full_v1/plan_mimic_final_layer_100k_v1/smoke_audit.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-r2", type=float, default=0.999)
    parser.add_argument("--max-normalized-rmse", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_rows = []
    for model, suffix, layer, _n_layers in MODEL_SPECS:
        if model not in EXTRACT_MODELS:
            continue
        model_root = args.activation_root / suffix / "mimic_f"
        metadata_paths = sorted(model_root.glob("*/activation_metadata.json"))
        if len(metadata_paths) != 1:
            raise RuntimeError(f"expected one smoke shard for {model}, found {len(metadata_paths)}")
        shard = metadata_paths[0].parent
        metadata = json.loads(metadata_paths[0].read_text())
        records = read_csv(shard / "record_ids.csv")
        values = np.asarray(np.load(shard / f"layer_{layer:02d}.npy"), dtype=np.float32)
        if values.shape != (len(records), 768) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid smoke activation array for {model}: {values.shape}")
        if [int(value) for value in metadata.get("layers", [])] != [layer]:
            raise RuntimeError(f"wrong smoke layer metadata for {model}: {metadata.get('layers')}")

        old_records, old_values = load_final_layer(ROOT, suffix, layer)
        old_by_id = {normalize_record_id(row["ecg_id"]): index for index, row in enumerate(old_records)}
        pairs = [
            (index, old_by_id[normalize_record_id(row["ecg_id"])])
            for index, row in enumerate(records)
            if normalize_record_id(row["ecg_id"]) in old_by_id
        ]
        if not pairs:
            raise RuntimeError(f"smoke shard for {model} has no overlap with 4k audit set")
        current = values[[left for left, _right in pairs]].astype(np.float64)
        reference = old_values[[right for _left, right in pairs]].astype(np.float64)
        difference = current - reference
        rmse = float(np.sqrt(np.mean(difference**2)))
        reference_sd = float(np.std(reference))
        normalized_rmse = rmse / max(reference_sd, 1e-12)
        r2 = float(1.0 - np.sum(difference**2) / max(np.sum((reference - reference.mean()) ** 2), 1e-12))
        passed = bool(r2 >= args.min_r2 and normalized_rmse <= args.max_normalized_rmse)
        model_rows.append(
            {
                "model": model,
                "model_suffix": suffix,
                "layer": layer,
                "records": len(records),
                "overlap_records": len(pairs),
                "r2_vs_4k": r2,
                "rmse": rmse,
                "normalized_rmse": normalized_rmse,
                "max_abs": float(np.max(np.abs(difference))),
                "pass": passed,
            }
        )
        if not passed:
            raise RuntimeError(f"smoke equivalence gate failed for {model}: {model_rows[-1]}")
    payload = {
        "status": "pass",
        "models_expected": len(EXTRACT_MODELS),
        "models_passed": sum(row["pass"] for row in model_rows),
        "thresholds": {"min_r2": args.min_r2, "max_normalized_rmse": args.max_normalized_rmse},
        "models": model_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
