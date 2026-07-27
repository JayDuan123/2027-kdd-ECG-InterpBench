#!/usr/bin/env python
"""Paired grouped-bootstrap contrasts among external atom protocols."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "external_benchmark_v1"
OUT = ROOT / "results" / "benchmark_extension_v1" / "paired_protocols"

from scripts.benchmark_extension_common import (  # noqa: E402
    bh,
    bootstrap_steering_metrics,
    group_bootstrap_weights,
    interval_and_p,
    load_json,
    load_npz,
)

SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "steering_summary", ROOT / "scripts" / "summarize_steering_benchmark.py"
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY)

CONTRASTS = (
    ("local_minus_frozen", "frozen_atom", "local_atom"),
    ("adapted_minus_frozen", "frozen_atom", "cohort_adapted_atom"),
    ("adapted_minus_local", "local_atom", "cohort_adapted_atom"),
)
METRICS = (
    "ste",
    "otd_mean",
    "selectivity_margin",
    "wbi",
    "tier1_excess_attribution",
    "excess_selectivity",
    "wbi_improvement",
    "behavior_effect",
    "behavior_excess",
)
IMPROVEMENT_SIGN = {"otd_mean": -1, "wbi": -1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-cells", type=int, default=0, help="Smoke-test cap before contrasts.")
    parser.add_argument("--shard-index", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def discover() -> dict[tuple[str, str, str, int, str], tuple[Path, dict]]:
    entries = {}
    for result_path in sorted(BASE.glob("*/*/steering/*/seed*/*/result.json")):
        result = load_json(result_path)
        key = (
            result["model_suffix"],
            result["cohort"],
            result["target"],
            int(result["seed"]),
            result["protocol"],
        )
        entries[key] = (result_path, result)
    return entries


def point_metrics(data: dict[str, np.ndarray], result: dict) -> dict[str, float]:
    point = SUMMARY.one_stats(data, result, [])
    return {metric: float(point[metric]) for metric in METRICS}


def main() -> None:
    args = parse_args()
    entries = discover()
    base_keys = sorted({key[:4] for key in entries})
    complete = [key for key in base_keys if all(key + (protocol,) in entries for _, a, b in CONTRASTS for protocol in (a, b))]
    total_complete = len(complete)
    if args.shard_index >= 0:
        if args.num_shards <= 1 or args.shard_index >= args.num_shards:
            raise ValueError("shard-index requires num-shards > 1 and shard-index < num-shards")
        complete = complete[args.shard_index :: args.num_shards]
        args.out = args.out / "workers" / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    if args.max_cells:
        complete = complete[: args.max_cells]
    if not complete:
        raise RuntimeError("No complete three-protocol cells found")

    rows = []
    for cell_index, base_key in enumerate(complete):
        model_suffix, cohort, target, seed = base_key
        loaded = {}
        for protocol in ("frozen_atom", "local_atom", "cohort_adapted_atom"):
            path, result = entries[base_key + (protocol,)]
            loaded[protocol] = (result, load_npz(path.with_name("records.npz")))
        reference = loaded["frozen_atom"][1]
        for protocol, (_, data) in loaded.items():
            if not np.array_equal(reference["patient_ids"], data["patient_ids"]):
                raise RuntimeError(f"{base_key}: patient/record order differs for {protocol}")
            if not np.array_equal(reference["target_names"], data["target_names"]):
                raise RuntimeError(f"{base_key}: head order differs for {protocol}")

        rng = np.random.default_rng(args.seed + seed + sum(map(ord, "|".join(base_key[:3]))))
        weights, inverse = group_bootstrap_weights(reference["patient_ids"], args.bootstrap, rng)
        protocol_samples = {}
        protocol_points = {}
        for protocol, (result, data) in loaded.items():
            protocol_points[protocol] = point_metrics(data, result)
            protocol_samples[protocol] = bootstrap_steering_metrics(data, result, weights, inverse)

        model = loaded["frozen_atom"][0]["model"]
        family = loaded["frozen_atom"][0]["family"]
        split_unit = loaded["frozen_atom"][0]["split_unit"]
        tier0 = bool(
            loaded["frozen_atom"][0]["source_fidelity_eligible"]
            and loaded["frozen_atom"][0]["transport_eligible"]
            and loaded["frozen_atom"][0]["raw_head_test_auroc"] >= 0.70
            and loaded["frozen_atom"][0]["sae_readout_retention"] >= 0.95
        )
        for contrast, protocol_a, protocol_b in CONTRASTS:
            row = {
                "model": model,
                "model_suffix": model_suffix,
                "cohort": cohort,
                "target": target,
                "family": family,
                "seed": seed,
                "split_unit": split_unit,
                "contrast": contrast,
                "protocol_a": protocol_a,
                "protocol_b": protocol_b,
                "frozen_tier0_eligible": tier0,
                "bootstrap_samples": args.bootstrap,
            }
            for metric in METRICS:
                delta = protocol_points[protocol_b][metric] - protocol_points[protocol_a][metric]
                samples = protocol_samples[protocol_b][metric] - protocol_samples[protocol_a][metric]
                stats = interval_and_p(samples, IMPROVEMENT_SIGN.get(metric, 1))
                row[f"delta_{metric}"] = delta
                row[f"delta_{metric}_ci_low"] = stats["ci_low"]
                row[f"delta_{metric}_ci_high"] = stats["ci_high"]
                row[f"delta_{metric}_p_one_sided"] = stats["p_one_sided"]
                row[f"delta_{metric}_p_two_sided"] = stats["p_two_sided"]
            rows.append(row)
        print(f"paired cell {cell_index + 1}/{len(complete)}: {model}/{cohort}/{target}/seed{seed}", flush=True)

    frame = pd.DataFrame(rows)
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
        .drop_duplicates()
        .groupby("contrast")
        .size()
    )
    summary.insert(2, "target_combinations", summary.contrast.map(combinations).astype(int))
    summary.to_csv(args.out / "paired_protocol_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "complete_seed_cells": len(complete),
        "total_complete_seed_cells": total_complete,
        "contrast_rows": len(frame),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "bootstrap_samples": args.bootstrap,
        "fdr_family": "all seed cells within each contrast, separately by metric",
        "paired_unit": "identical patient/record bootstrap weights across protocols",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
