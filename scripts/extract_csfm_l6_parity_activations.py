#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CSFM_ROOT = Path("/rhf/allocations/wq8/yd68/Cardiac-Sensing-FM")
sys.path.insert(0, str(CSFM_ROOT))
sys.path.insert(0, str(CSFM_ROOT / "utils"))

from benchmark_v1.adapters.ecg_jepa import build_waveform_index, parse_header, record_name_for_ecg_id
from benchmark_v1.adapters.csfm import CSFM_LEADS, try_load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/manifest.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/activation_shards")
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--shard-size", type=int, default=64)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_and_preprocess(ecg_id: str, index):
    from scipy.io import loadmat
    # The upstream utils/__init__.py uses a non-relative import and cannot be
    # imported as a package. Importing its module directory executes the exact
    # official preprocessing implementation without modifying the upstream repo.
    from preprocess import preprocess_ecg

    entry = index[record_name_for_ecg_id(ecg_id)]
    header = parse_header(entry["hea"])
    lead_to_idx = {lead: i for i, lead in enumerate(header.leads)}
    raw = np.asarray(loadmat(entry["mat"])["val"], dtype=np.float32)
    if raw.shape[0] != header.n_signals and raw.shape[-1] == header.n_signals:
        raw = raw.T
    raw = raw[[lead_to_idx[x] for x in CSFM_LEADS]]
    return preprocess_ecg(raw, fs=header.sample_rate).astype(np.float32)


def main() -> None:
    a = parse_args()
    rows = list(csv.DictReader(a.manifest.open()))
    lo = a.shard_id * a.shard_size
    selected = rows[lo:lo + a.shard_size]
    if not selected:
        raise SystemExit(f"empty shard {a.shard_id}")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard_{a.shard_id:04d}"
    final_npy = a.out_dir / f"{stem}.npy"
    final_csv = a.out_dir / f"{stem}.csv"
    final_json = a.out_dir / f"{stem}.json"
    if final_npy.exists() and final_csv.exists() and final_json.exists():
        print(f"already complete: {stem}")
        return

    import torch
    model, status = try_load_model(a.device)
    if model is None:
        raise RuntimeError(status)
    model.pool = "mean"
    index = build_waveform_index()
    output = []
    with torch.no_grad():
        for start in range(0, len(selected), a.micro_batch):
            batch_rows = selected[start:start + a.micro_batch]
            waves = np.stack([load_and_preprocess(r["ecg_id"], index) for r in batch_rows])
            source = torch.as_tensor(waves, dtype=torch.float32, device=a.device)
            channel = torch.arange(12, dtype=torch.long, device=a.device)
            # mlp_head is Identity and pool='mean': final LayerNorm output after
            # transformer block 6, mean-pooled over CLS and ECG patch tokens.
            emb = model(source, channel, task="cls")
            output.append(emb.detach().cpu().numpy().astype(np.float32))
    acts = np.concatenate(output, axis=0)
    tmp_npy = final_npy.with_suffix(f".npy.tmp.{os.getpid()}")
    with tmp_npy.open("wb") as f:
        np.save(f, acts)
    tmp_npy.replace(final_npy)
    tmp_csv = final_csv.with_suffix(f".csv.tmp.{os.getpid()}")
    with tmp_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0]))
        w.writeheader(); w.writerows(selected)
    tmp_csv.replace(final_csv)
    meta = {
        "shard_id": a.shard_id, "rows": len(selected), "shape": list(acts.shape),
        "model_status": status, "layer": "CSFM Tiny transformer block 6 final LayerNorm",
        "pooling": "mean over CLS plus all ECG patch tokens",
        "preprocessing": "official utils.preprocess.preprocess_ecg: 250 Hz, NeuroKit2 clean, 2500 samples, per-lead z-normalization",
    }
    tmp_json = final_json.with_suffix(f".json.tmp.{os.getpid()}")
    tmp_json.write_text(json.dumps(meta, indent=2) + "\n")
    tmp_json.replace(final_json)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
