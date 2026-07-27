#!/usr/bin/env python
"""Audit and summarize all 72 transport-ladder workers."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "benchmark_extension_v1" / "transport_ladder"
WORKERS = BASE / "workers"
from scripts.benchmark_extension_common import bh  # noqa: E402
METHODS = (
    "frozen", "diagonal_full_train", "coral_full_train", "fewshot_n128",
    "fewshot_n512", "fewshot_n2048", "cohort_adapted_full",
)
QUALITY_METRICS = {
    "recon_r2": 1,
    "dead_fraction": -1,
    "readout_retention_median": 1,
}
STEERING_METRICS = {
    "ste": 1,
    "otd_mean": -1,
    "selectivity_margin": 1,
    "wbi": -1,
    "excess_selectivity": 1,
    "behavior_excess": 1,
}


def paired_inference(
    frame: pd.DataFrame, domain: str, metrics: dict[str, int], seed: int = 20260713
) -> list[dict]:
    """Bootstrap method-minus-frozen means over 24 model/cohort units."""
    unit = frame.groupby(["method", "model", "model_suffix", "cohort"], as_index=False)[list(metrics)].mean()
    frozen = unit[unit.method.eq("frozen")].drop(columns="method")
    rows = []
    for method in METHODS:
        if method == "frozen":
            continue
        paired = unit[unit.method.eq(method)].merge(
            frozen, on=["model", "model_suffix", "cohort"], suffixes=("", "_frozen"), validate="one_to_one"
        )
        if len(paired) != 24:
            raise RuntimeError(f"{domain}/{method}: expected 24 paired model/cohort units, found {len(paired)}")
        rng = np.random.default_rng(seed + sum(map(ord, domain + method)))
        indices = rng.integers(0, len(paired), size=(10000, len(paired)))
        for metric, sign in metrics.items():
            delta = paired[metric].to_numpy() - paired[f"{metric}_frozen"].to_numpy()
            samples = delta[indices].mean(axis=1)
            low, high = np.quantile(samples, [.025, .975])
            p_left = (1.0 + float((samples <= 0).sum())) / (len(samples) + 1.0)
            p_right = (1.0 + float((samples >= 0).sum())) / (len(samples) + 1.0)
            oriented = samples * sign
            rows.append(
                {
                    "domain": domain, "method": method, "reference": "frozen", "metric": metric,
                    "model_cohort_pairs": len(paired), "improvement_sign": sign,
                    "mean_delta": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
                    "p_one_sided_improvement": (1.0 + float((oriented <= 0).sum())) / (len(oriented) + 1.0),
                    "p_two_sided": min(1.0, 2.0 * min(p_left, p_right)),
                    "bootstrap_samples": len(samples),
                }
            )
    return rows


def main() -> None:
    complete = sorted(WORKERS.glob("*/*/seed*/complete.json"))
    quality_paths = sorted(WORKERS.glob("*/*/seed*/quality_metrics.csv"))
    steering_paths = sorted(WORKERS.glob("*/*/seed*/steering_metrics.csv"))
    if not (len(complete) == len(quality_paths) == len(steering_paths) == 72):
        raise RuntimeError(
            f"Transport ladder incomplete: complete={len(complete)} quality={len(quality_paths)} steering={len(steering_paths)}"
        )
    quality = pd.concat([pd.read_csv(path) for path in quality_paths], ignore_index=True)
    steering = pd.concat([pd.read_csv(path) for path in steering_paths], ignore_index=True)
    expected_quality = 72 * len(METHODS)
    if len(quality) != expected_quality or set(quality.method) != set(METHODS):
        raise RuntimeError(f"Unexpected quality matrix: rows={len(quality)}, methods={sorted(quality.method.unique())}")
    expected_steering = int(quality.merge(
        pd.DataFrame({"cohort": ["chapman_f", "cpsc_f", "ningbo_f", "mimic_f"], "tasks": [4, 2, 4, 5]}),
        on="cohort"
    ).tasks.sum())
    if len(steering) != expected_steering:
        raise RuntimeError(f"Unexpected steering rows: {len(steering)} != {expected_steering}")
    BASE.mkdir(parents=True, exist_ok=True)
    quality.to_csv(BASE / "transport_quality_seed_cells.csv", index=False)
    steering.to_csv(BASE / "transport_steering_seed_cells.csv", index=False)
    quality_profile = quality.groupby("method", as_index=False).agg(
        seed_pairs=("cohort", "size"), model_cohort_pairs=("cohort", lambda x: len(x) // 3),
        recon_r2_mean=("recon_r2", "mean"), recon_r2_min=("recon_r2", "min"),
        recon_pass_085=("recon_r2", lambda x: int((x >= .85).sum())),
        dead_fraction_mean=("dead_fraction", "mean"), dead_pass_020=("dead_fraction", lambda x: int((x < .20).sum())),
        readout_retention_median=("readout_retention_median", "median"),
        retention_pass_095=("readout_retention_median", lambda x: int((x >= .95).sum())),
        source_alignment_mean=("decoder_source_max_cosine_sample_mean", "mean"),
    )
    quality_profile["model_cohort_pairs"] = quality_profile.seed_pairs // 3
    quality_profile.to_csv(BASE / "transport_quality_profile.csv", index=False)
    steering_profile = steering.groupby("method", as_index=False).agg(
        seed_task_cells=("target", "size"), ste_mean=("ste", "mean"),
        otd_mean=("otd_mean", "mean"), wbi_median=("wbi", "median"),
        excess_selectivity_mean=("excess_selectivity", "mean"),
        behavior_excess_mean=("behavior_excess", "mean"),
        positive_excess_selectivity=("excess_selectivity", lambda x: int((x > 0).sum())),
        positive_behavior_excess=("behavior_excess", lambda x: int((x > 0).sum())),
    )
    steering_profile.to_csv(BASE / "transport_steering_profile.csv", index=False)

    frozen_quality = quality[quality.method.eq("frozen")].drop(columns="method")
    quality_deltas = quality[~quality.method.eq("frozen")].merge(
        frozen_quality, on=["model", "model_suffix", "cohort", "seed"], suffixes=("", "_frozen")
    )
    for metric in ("recon_r2", "dead_fraction", "readout_retention_median"):
        quality_deltas[f"delta_{metric}"] = quality_deltas[metric] - quality_deltas[f"{metric}_frozen"]
    quality_deltas.to_csv(BASE / "transport_quality_paired_deltas.csv", index=False)
    frozen_steering = steering[steering.method.eq("frozen")].drop(columns="method")
    steering_deltas = steering[~steering.method.eq("frozen")].merge(
        frozen_steering, on=["model", "model_suffix", "cohort", "seed", "target"], suffixes=("", "_frozen")
    )
    for metric in ("ste", "otd_mean", "selectivity_margin", "wbi", "excess_selectivity", "behavior_excess"):
        steering_deltas[f"delta_{metric}"] = steering_deltas[metric] - steering_deltas[f"{metric}_frozen"]
    steering_deltas.to_csv(BASE / "transport_steering_paired_deltas.csv", index=False)
    inference = pd.DataFrame(
        paired_inference(quality, "quality", QUALITY_METRICS)
        + paired_inference(steering, "steering", STEERING_METRICS)
    )
    inference["q_two_sided"] = np.nan
    inference["q_one_sided_improvement"] = np.nan
    for _, indices in inference.groupby("domain").groups.items():
        inference.loc[indices, "q_two_sided"] = bh(inference.loc[indices, "p_two_sided"].to_numpy())
        inference.loc[indices, "q_one_sided_improvement"] = bh(
            inference.loc[indices, "p_one_sided_improvement"].to_numpy()
        )
    inference.to_csv(BASE / "transport_paired_inference.csv", index=False)
    metadata = {
        "schema_version": 1, "workers": len(complete), "quality_rows": len(quality),
        "steering_rows": len(steering), "methods": list(METHODS), "all_complete": True,
        "evaluation": "deterministic held-out subsets; train-only alignment/adaptation",
        "paired_inference": "seed/target aggregated within each of 24 model-cohort units; 10000 paired bootstrap samples; BH within quality and steering domains",
    }
    (BASE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(quality_profile.to_string(index=False)); print(steering_profile.to_string(index=False))


if __name__ == "__main__":
    main()
