#!/usr/bin/env python
"""Audit paired-protocol shards and apply global bootstrap FDR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_external_protocol_pairs import CONTRASTS, METRICS, OUT, bh, discover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=6)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = []
    for shard in range(args.num_shards):
        worker = args.out / "workers" / f"shard_{shard:02d}_of_{args.num_shards:02d}"
        metadata = json.loads((worker / "metadata.json").read_text())
        if metadata["shard_index"] != shard or metadata["num_shards"] != args.num_shards:
            raise RuntimeError(f"Invalid shard metadata: {worker}")
        frame = pd.read_csv(worker / "paired_protocol_seed_cells.csv")
        if len(frame) != metadata["contrast_rows"]:
            raise RuntimeError(f"Shard row-count mismatch: {worker}")
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)
    keys = ["model_suffix", "cohort", "target", "seed", "contrast"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate paired-protocol seed cells across shards")
    entries = discover()
    base_keys = {key[:4] for key in entries}
    complete = [
        key for key in base_keys
        if all(key + (protocol,) in entries for _, a, b in CONTRASTS for protocol in (a, b))
    ]
    expected_rows = len(complete) * len(CONTRASTS)
    if len(frame) != expected_rows:
        raise RuntimeError(f"Paired protocol matrix incomplete: {len(frame)} != {expected_rows}")

    for metric in METRICS:
        p_col = f"delta_{metric}_p_one_sided"
        q_col = f"delta_{metric}_q"
        frame[q_col] = np.nan
        for _, indices in frame.groupby("contrast").groups.items():
            frame.loc[indices, q_col] = bh(frame.loc[indices, p_col].to_numpy())

    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "paired_protocol_seed_cells.csv", index=False)
    target_profile = frame.groupby(
        ["model", "cohort", "target", "family", "contrast"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        frozen_tier0_eligible=("frozen_tier0_eligible", "all"),
        delta_ste_mean=("delta_ste", "mean"),
        delta_wbi_mean=("delta_wbi", "mean"),
        delta_excess_selectivity_mean=("delta_excess_selectivity", "mean"),
        delta_behavior_excess_mean=("delta_behavior_excess", "mean"),
        ste_improved_seeds=("delta_ste", lambda x: int((x > 0).sum())),
        wbi_improved_seeds=("delta_wbi", lambda x: int((x < 0).sum())),
        selective_q05_seeds=("delta_excess_selectivity_q", lambda x: int((x < 0.05).sum())),
        behavior_q05_seeds=("delta_behavior_excess_q", lambda x: int((x < 0.05).sum())),
    )
    target_profile.to_csv(args.out / "paired_protocol_target_profile.csv", index=False)
    summary = frame.groupby("contrast", as_index=False).agg(
        seed_cells=("target", "size"),
        frozen_tier0_cells=("frozen_tier0_eligible", "sum"),
        delta_ste_mean=("delta_ste", "mean"),
        delta_wbi_mean=("delta_wbi", "mean"),
        delta_excess_selectivity_mean=("delta_excess_selectivity", "mean"),
        delta_behavior_excess_mean=("delta_behavior_excess", "mean"),
        ste_q05_cells=("delta_ste_q", lambda x: int((x < 0.05).sum())),
        wbi_q05_cells=("delta_wbi_q", lambda x: int((x < 0.05).sum())),
        selectivity_q05_cells=("delta_excess_selectivity_q", lambda x: int((x < 0.05).sum())),
        behavior_q05_cells=("delta_behavior_excess_q", lambda x: int((x < 0.05).sum())),
    )
    combinations = (
        frame[["contrast", "model", "cohort", "target"]]
        .drop_duplicates().groupby("contrast").size()
    )
    summary.insert(2, "target_combinations", summary.contrast.map(combinations).astype(int))
    summary.to_csv(args.out / "paired_protocol_summary.csv", index=False)
    stratified_input = pd.concat(
        [
            frame.assign(analysis_stratum="all_cells"),
            frame[frame.frozen_tier0_eligible].assign(analysis_stratum="frozen_tier0_only"),
        ],
        ignore_index=True,
    )
    stratified = stratified_input.groupby(["analysis_stratum", "contrast"], as_index=False).agg(
        seed_cells=("target", "size"),
        target_combinations=("target", lambda x: len(
            stratified_input.loc[x.index, ["model", "cohort", "target"]].drop_duplicates()
        )),
        delta_ste_mean=("delta_ste", "mean"),
        delta_wbi_mean=("delta_wbi", "mean"),
        delta_excess_selectivity_mean=("delta_excess_selectivity", "mean"),
        delta_behavior_excess_mean=("delta_behavior_excess", "mean"),
        ste_q05_cells=("delta_ste_q", lambda x: int((x < 0.05).sum())),
        selectivity_q05_cells=("delta_excess_selectivity_q", lambda x: int((x < 0.05).sum())),
        behavior_q05_cells=("delta_behavior_excess_q", lambda x: int((x < 0.05).sum())),
    )
    stratified.to_csv(args.out / "paired_protocol_stratified_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "complete_seed_cells": len(complete),
        "contrast_rows": len(frame),
        "bootstrap_samples": int(frame.bootstrap_samples.iloc[0]),
        "shards": args.num_shards,
        "fdr_family": "all seed cells within each contrast, separately by metric",
        "paired_unit": "identical patient/record bootstrap weights across protocols",
        "all_complete": True,
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
