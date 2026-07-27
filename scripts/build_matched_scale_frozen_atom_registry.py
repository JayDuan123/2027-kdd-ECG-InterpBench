#!/usr/bin/env python
"""Build PTB-trained frozen atom sets for external native tasks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCALE = ROOT / "results/sae_reconciliation/matched_scale_v1"
SOURCE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
OUT = ROOT / "results/external_benchmark_v1/frozen_atom_registry"
TASK_MAP = {
    "af_rhythm_native": ["afib"],
    "bbb_conduction_native": ["lbbb", "rbbb"],
    "qt_interval_native": ["qt_interval"],
    "st_t_abnormal_native": ["invt", "std_generic", "nst_generic"],
    "af_rhythm_icd": ["afib"],
    "bbb_conduction_icd": ["lbbb", "rbbb"],
    "qt_interval_icd": ["qt_interval"],
    "mi_ischemia_icd": ["imi", "asmi", "ilmi", "ami", "isc_generic"],
    "hypertrophy_icd": ["lvh", "lao_lae"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(); p.add_argument("--task-index", type=int, required=True)
    return p.parse_args()


def encode(sae, acts: np.ndarray, device: str = "cuda", batch: int = 256) -> np.ndarray:
    import torch
    chunks = []
    sae.eval()
    with torch.no_grad():
        for lo in range(0, len(acts), batch):
            raw = torch.as_tensor(np.asarray(acts[lo:lo+batch]), dtype=torch.float32, device=device)
            chunks.append(sae.encode(raw).cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    a = parse_args()
    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE
    import torch

    manifest = pd.read_csv(SCALE / "training_manifest.csv")
    row = manifest.iloc[a.task_index]
    model = row.model; seed = int(row.seed); safe = model.lower().replace("-", "_")
    split = pd.read_csv(SOURCE / "manifest.csv").split.to_numpy()
    tr = np.where(split == "train")[0]
    acts = np.load(ROOT / f"results/probe_features/{row.feature_suffix}/pooled.npy", mmap_mode="r")
    train = np.asarray(acts[tr], dtype=np.float32)
    bundle = joblib.load(SOURCE / "models" / safe / "frozen_heads.joblib")
    scaler = bundle["scaler"]; heads = bundle["heads"]
    checkpoint = Path(row.checkpoint)
    saved = torch.load(checkpoint, map_location="cuda"); cfg = saved["config"]
    sae = BatchTopKSAE(train.shape[1], int(cfg["n_features"]), int(cfg["k"])).cuda()
    sae.load_state_dict(saved["model"]); sae.eval()
    ztr = encode(sae, train)
    decoder = sae.W_dec.detach().cpu().numpy(); sigma = sae.sigma.detach().cpu().numpy()
    source_scores = {}
    for target in sorted({t for targets in TASK_MAP.values() for t in targets}):
        head = heads[target]; labels = np.asarray(head["labels"], dtype=float)[tr]
        coef = np.asarray(head["clf"].coef_).reshape(-1)
        gradient = ((coef / scaler.scale_) * sigma) @ decoder
        valid = np.isfinite(labels)
        if head.get("type", "binary") == "binary":
            focus = valid & (labels == 1)
        else:
            focus = valid & (labels >= np.nanquantile(labels[valid], 0.75))
        score = (ztr[focus] * gradient).mean(axis=0)
        source_scores[target] = score / max(float(np.max(np.abs(score))), 1e-12)
    tasks = {}
    for external_task, source_targets in TASK_MAP.items():
        combined = np.mean([source_scores[target] for target in source_targets], axis=0)
        ranking = np.argsort(combined)[::-1]
        tasks[external_task] = {
            "source_targets": source_targets,
            "top1": ranking[:1].astype(int).tolist(),
            "top5": ranking[:5].astype(int).tolist(),
            "top10": ranking[:10].astype(int).tolist(),
        }
    payload = {
        "schema_version": 1, "model": model, "model_suffix": row.feature_suffix,
        "seed": seed, "checkpoint": str(checkpoint), "selection_cohort": "PTB-XL train",
        "selection_method": "normalized positive integrated-gradient aggregation", "tasks": tasks,
    }
    out = OUT / safe / f"seed{seed}.json"; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".json.tmp.{os.getpid()}"); tmp.write_text(json.dumps(payload, indent=2)+"\n"); tmp.replace(out)
    print(json.dumps({"model": model, "seed": seed, "tasks": {k:v["top5"] for k,v in tasks.items()}}))


if __name__ == "__main__":
    main()
