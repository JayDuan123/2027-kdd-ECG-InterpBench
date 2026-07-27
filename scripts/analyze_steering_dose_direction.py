#!/usr/bin/env python
"""Patient-level dose and direction audit for robust Top-5 steering cells."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
SPEC = importlib.util.spec_from_file_location(
    "steering_summary", ROOT / "scripts/summarize_steering_benchmark.py"
)
STEERING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STEERING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--robustness",
        choices=("robust_3_of_3", "at_least_2_of_3"),
        default="robust_3_of_3",
    )
    return parser.parse_args()


def robust_pairs(base: Path, mode: str) -> set[tuple[str, str]]:
    profile = pd.read_csv(base / "summary/multimodel_target_profile.csv")
    keep = profile.analysis_role.eq("main") & profile.headline_eligible.astype(bool)
    if mode == "robust_3_of_3":
        keep &= profile.tier2_pass.eq(3)
    else:
        keep &= profile.tier2_pass.ge(2)
    return set(zip(profile.loc[keep, "model"], profile.loc[keep, "target"]))


def scaled_data(data: dict[str, np.ndarray], scale: float) -> dict[str, np.ndarray]:
    out = dict(data)
    out["top5_delta"] = np.asarray(data["top5_delta"], dtype=np.float32) * scale
    out["random_top5_delta"] = np.asarray(data["random_top5_delta"], dtype=np.float32) * scale
    return out


def signed_target_change(data: dict[str, np.ndarray], result: dict) -> tuple[float, float]:
    names = data["target_names"].astype(str).tolist()
    kinds = data["target_types"].astype(str).tolist()
    j = names.index(result["target"])
    labels = data["labels"][:, j]
    focus = STEERING.focus_mask(
        labels, kinds[j], float(result["focus_thresholds_train"][result["target"]])
    )
    valid = np.isfinite(labels)
    denominator = max(float(np.nanstd(data["baseline_logits"][valid, j])), 1e-8)
    target = float(np.nanmean(data["top5_delta"][focus, j])) / denominator
    random = float(np.nanmean(data["random_top5_delta"][focus, :, j], axis=0).mean()) / denominator
    return target, random


def main() -> None:
    args = parse_args()
    pairs = robust_pairs(args.base, args.robustness)
    if not pairs:
        raise RuntimeError("No robust steering pairs found; run the v2 summary first")
    scales = (-1.0, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
    rows = []
    for result_path in sorted((args.base / "models").glob("*/tasks/seed*/*/result.json")):
        result = json.loads(result_path.read_text())
        pair = (result["model"], result["target"])
        if pair not in pairs:
            continue
        with np.load(result_path.with_name("records.npz"), allow_pickle=False) as loaded:
            original = {key: loaded[key] for key in loaded.files}
        for scale in scales:
            data = scaled_data(original, scale)
            point = STEERING.one_stats(data, result, [])
            signed, random_signed = signed_target_change(data, result)
            rng = np.random.default_rng(
                args.seed
                + int(result["seed"])
                + sum(map(ord, result["model"] + result["target"]))
                + int(round((scale + 2.0) * 1000))
            )
            samples = STEERING.vectorized_bootstrap(data, result, [], args.bootstrap, rng)
            row = {
                "model": result["model"],
                "target": result["target"],
                "family": result["family"],
                "seed": int(result["seed"]),
                "direction": "neutralize" if scale > 0 else "enhance_away_from_centroid",
                "dose": abs(scale),
                "signed_target_change": signed,
                "random_signed_target_change": random_signed,
                "bootstrap_samples": args.bootstrap,
                **point,
            }
            for metric in ("excess_selectivity", "wbi_improvement", "behavior_excess"):
                values = np.asarray(samples[metric])
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(
                    values, [0.025, 0.975]
                )
                row[f"{metric}_p_one_sided"] = (1.0 + float((values <= 0).sum())) / (
                    len(values) + 1.0
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No robust task records were available")
    for metric in ("excess_selectivity", "wbi_improvement", "behavior_excess"):
        frame[f"{metric}_q"] = np.nan
        for _, indices in frame.groupby(["model", "family", "direction", "dose"]).groups.items():
            frame.loc[indices, f"{metric}_q"] = STEERING.bh(
                frame.loc[indices, f"{metric}_p_one_sided"].to_numpy()
            )
    frame["selective_vs_random"] = (
        (frame.excess_selectivity_ci_low > 0)
        & (frame.wbi_improvement_ci_low > 0)
        & (frame.excess_selectivity_q < 0.05)
        & (frame.wbi_improvement_q < 0.05)
    )
    out = args.base / "summary/dose_direction"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "dose_direction_seed_cells.csv", index=False)
    profile = frame.groupby(
        ["model", "target", "family", "direction", "dose"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        signed_target_change_mean=("signed_target_change", "mean"),
        ste_mean=("ste", "mean"),
        excess_selectivity_mean=("excess_selectivity", "mean"),
        behavior_excess_mean=("behavior_excess", "mean"),
        selective_seed_pass=("selective_vs_random", "sum"),
    )
    profile["robust_selective"] = profile.selective_seed_pass.eq(3)
    profile.to_csv(out / "dose_direction_profile.csv", index=False)
    print(
        profile.groupby(["direction", "dose"], as_index=False).agg(
            cells=("target", "size"), robust=("robust_selective", "sum")
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()
