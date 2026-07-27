#!/usr/bin/env python
"""Build the row-aligned manifest and model gate for multimodel steering."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/sae_reconciliation/steering_benchmark_v1"
OUT = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"
MODELS = {
    "CSFM": "csfm_cu118_commons",
    "CARDIAC-FM": "cardiac_fm_cu118_commons",
    "ECG-FM": "ecg_fm_cu118_commons",
    "ECG-JEPA": "ecg_jepa_cu118_commons",
    "HuBERT-ECG": "hubert_ecg_cu118_commons",
    "ST-MEM": "st_mem_cu118_commons",
}


def main() -> None:
    source = pd.read_csv(SOURCE / "manifest.csv")
    registry = pd.read_csv(SOURCE / "target_registry.csv")
    reference = None; gate = []
    for model, suffix in MODELS.items():
        feature_dir = ROOT / "results/probe_features" / suffix
        records = pd.read_csv(feature_dir / "records.csv")
        ids = records.ecg_id.astype(int).tolist()
        if reference is None:
            reference = ids
        if ids != reference:
            raise RuntimeError(f"Record order differs for {model}")
        import numpy as np
        pooled = np.load(feature_dir / "pooled.npy", mmap_mode="r")
        gate.append({"model": model, "feature_suffix": suffix, "n_records": len(records),
                     "representation": "model_pooled", "d_hidden": int(pooled.shape[1]),
                     "activation_available": bool(len(pooled) == len(records)),
                     "head_protocol": "shared L2-logistic/Ridge", "sae_seed4311": "pending_phase0",
                     "main_status": "phase0_pending"})
    aligned = pd.DataFrame({"ecg_id": reference}).merge(source.drop(columns=["row_index"]), on="ecg_id", how="left", validate="one_to_one")
    if aligned.split.isna().any():
        raise RuntimeError("Missing split after row alignment")
    aligned.insert(0, "row_index", range(len(aligned)))
    OUT.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(OUT / "manifest.csv", index=False)
    registry.to_csv(OUT / "target_registry.csv", index=False)
    pd.DataFrame(gate).to_csv(OUT / "model_gate.csv", index=False)
    print(pd.DataFrame(gate).to_string(index=False))


if __name__ == "__main__":
    main()
