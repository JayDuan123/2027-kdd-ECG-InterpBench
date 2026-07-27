#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/manifest.csv")
    p.add_argument("--shards", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/activation_shards")
    p.add_argument("--out", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    a = p.parse_args()
    expected = list(csv.DictReader(a.manifest.open()))
    csvs = sorted(a.shards.glob("shard_*.csv"))
    arrays, rows = [], []
    for cp in csvs:
        npy = cp.with_suffix(".npy")
        meta = cp.with_suffix(".json")
        if not npy.exists() or not meta.exists():
            continue
        part_rows = list(csv.DictReader(cp.open()))
        part = np.load(npy)
        if len(part_rows) != len(part):
            raise RuntimeError(f"row mismatch: {cp}")
        rows.extend(part_rows); arrays.append(part)
    if len(rows) != len(expected):
        raise RuntimeError(f"incomplete shards: found {len(rows)}, expected {len(expected)}")
    if [r["ecg_id"] for r in rows] != [r["ecg_id"] for r in expected]:
        raise RuntimeError("shard order does not match manifest")
    acts = np.concatenate(arrays).astype(np.float32)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".npy.tmp")
    with tmp.open("wb") as f: np.save(f, acts)
    tmp.replace(a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"shape": list(acts.shape), "records": len(rows)}, indent=2)+"\n")
    print(f"merged {acts.shape} -> {a.out}")


if __name__ == "__main__": main()
