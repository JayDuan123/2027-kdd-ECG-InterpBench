#!/usr/bin/env python
"""Build and audit the 6-model x 4-cohort x 3-seed comparison manifest."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.method_comparison_common import (  # noqa: E402
    ACTIVATIONS,
    BASE,
    COHORTS,
    EXTERNAL,
    MODEL_NAMES,
    SEEDS,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=BASE / "manifest.csv")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def only_complete_cache(pair_root: Path, seed: int) -> Path:
    candidates = sorted(
        (pair_root / "steering_cache" / "cohort_adapted").glob(f"seed{seed}_N*_k*")
    )
    valid = [path for path in candidates if (path / "complete.json").exists()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one cohort-adapted cache for {pair_root}/seed{seed}: {valid}")
    return valid[0]


def main() -> None:
    args = parse_args()
    rows = []
    for model_suffix, model in MODEL_NAMES.items():
        for cohort in COHORTS:
            pair_root = EXTERNAL / model_suffix / cohort
            activation_root = ACTIVATIONS / model_suffix / cohort
            head_path = pair_root / "frozen_heads.joblib"
            shards_path = activation_root / "shards.csv"
            if not head_path.exists() or not shards_path.exists():
                raise FileNotFoundError(f"Missing pair inputs: {model_suffix}/{cohort}")
            bundle = joblib.load(head_path, mmap_mode="r")
            split = np.asarray(bundle["split"]).astype(str)
            targets = list(bundle["targets"])
            dimension = int(len(np.asarray(bundle["scaler"].scale_)))
            if dimension < 64:
                raise RuntimeError(f"{model_suffix}/{cohort}: d={dimension} < common rank 64")
            for seed in SEEDS:
                cache = only_complete_cache(pair_root, seed)
                checkpoint_candidates = sorted(
                    (pair_root / "cohort_adapted_sae" / f"seed{seed}").glob("batchtopk_N*_k*.pt")
                )
                checkpoints = [
                    path
                    for path in checkpoint_candidates
                    if path.with_suffix(".metrics.json").exists()
                ]
                if len(checkpoints) != 1:
                    raise RuntimeError(
                        f"Expected one complete SAE checkpoint for {model_suffix}/{cohort}/seed{seed}: {checkpoints}"
                    )
                missing_results = [
                    target
                    for target in targets
                    if not (
                        pair_root
                        / "steering"
                        / "cohort_adapted_atom"
                        / f"seed{seed}"
                        / target
                        / "result.json"
                    ).exists()
                ]
                if missing_results:
                    raise RuntimeError(
                        f"Missing steering results for {model_suffix}/{cohort}/seed{seed}: {missing_results}"
                    )
                rows.append(
                    {
                        "task_index": len(rows),
                        "model": model,
                        "model_suffix": model_suffix,
                        "cohort": cohort,
                        "seed": seed,
                        "dimension": dimension,
                        "records": len(split),
                        "train_records": int((split == "train").sum()),
                        "validation_records": int((split == "val").sum()),
                        "test_records": int((split == "test").sum()),
                        "targets": len(targets),
                        "target_names": "|".join(targets),
                        "head_path": str(head_path.resolve()),
                        "activation_root": str(activation_root.resolve()),
                        "existing_sae_checkpoint": str(checkpoints[0].resolve()),
                        "existing_sae_cache": str(cache.resolve()),
                    }
                )
    manifest = pd.DataFrame(rows)
    expected = len(MODEL_NAMES) * len(COHORTS) * len(SEEDS)
    if len(manifest) != expected or manifest.task_index.tolist() != list(range(expected)):
        raise RuntimeError(f"Manifest audit failed: {len(manifest)} rows, expected {expected}")
    summary = {
        "schema_version": 1,
        "tasks": len(manifest),
        "model_cohort_pairs": int(manifest[["model_suffix", "cohort"]].drop_duplicates().shape[0]),
        "models": int(manifest.model_suffix.nunique()),
        "cohorts": int(manifest.cohort.nunique()),
        "seeds": int(manifest.seed.nunique()),
        "target_seed_cells": int(manifest.targets.sum()),
        "all_inputs_present": True,
        "data_files_modified": False,
    }
    print(manifest.groupby(["model", "cohort"], as_index=False).first()[
        ["model", "cohort", "dimension", "records", "targets"]
    ].to_string(index=False))
    print(summary)
    if args.preflight_only:
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    write_json(args.out.with_suffix(".metadata.json"), summary)


if __name__ == "__main__":
    main()
