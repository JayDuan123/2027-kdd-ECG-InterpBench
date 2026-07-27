#!/usr/bin/env python
"""Build the frozen 24-pair x 3-seed cohort-adapted SAE manifest."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/external_benchmark_v1"
SOURCE = ROOT / "results/sae_reconciliation/matched_scale_v1/training_manifest.csv"


def main() -> None:
    pairs = pd.read_csv(BASE / "head_pair_manifest.csv").sort_values("task_index")
    source = pd.read_csv(SOURCE)
    dims = source.groupby("feature_suffix").d_hidden.first().astype(int).to_dict()
    rows = []
    for pair in pairs.itertuples(index=False):
        if pair.model_suffix not in dims:
            raise RuntimeError(f"No validated source hidden dimension for {pair.model_suffix}")
        root = BASE / pair.model_suffix / pair.cohort / "cohort_adapted_sae"
        acts = root / "pooled_all.npy"
        manifest = root / "sae_manifest.csv"
        d = int(dims[pair.model_suffix]); n_features = d * 8; k = d // 8
        for seed in (4311, 4312, 4313):
            rows.append({
                "task_index": len(rows), "pair_index": int(pair.task_index),
                "model_suffix": pair.model_suffix, "cohort": pair.cohort, "seed": seed,
                "acts": str(acts), "manifest": str(manifest), "d_hidden": d,
                "expansion_E": 8, "k_over_d": 0.125, "k_over_N": 0.015625,
                "N": n_features, "k": k,
                "checkpoint": str(root / f"seed{seed}" / f"batchtopk_N{n_features}_k{k}.pt"),
                "materialized": acts.exists() and manifest.exists(),
            })
    frame = pd.DataFrame(rows)
    if len(frame) != 72 or not frame.task_index.eq(range(72)).all():
        raise RuntimeError(f"Expected frozen 72-row manifest, got {len(frame)}")
    out = BASE / "cohort_adapted_sae_manifest.csv"
    frame.to_csv(out, index=False)
    print(out, len(frame), "materialized", int(frame.materialized.sum()))


if __name__ == "__main__":
    main()
