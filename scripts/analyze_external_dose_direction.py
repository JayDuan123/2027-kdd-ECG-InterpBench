#!/usr/bin/env python
"""Dose, direction, and zero-ablation audit for a frozen external target panel."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "external_benchmark_v1"
OUT = ROOT / "results" / "benchmark_extension_v1" / "dose_direction"
PROFILE = BASE / "summary" / "external_steering_target_profile.csv"

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

TOP_K = (1, 3, 5, 10)
ALPHAS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--max-panel", type=int, default=0, help="Smoke-test cap.")
    parser.add_argument("--panel-index", type=int, default=-1)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def freeze_panel() -> pd.DataFrame:
    profile = pd.read_csv(PROFILE)
    frozen = profile[profile.protocol.eq("frozen_atom")].copy()
    robust = frozen[frozen.robustness.eq("robust_3_of_3")].copy()
    if len(robust) != 5:
        raise RuntimeError(f"Expected exactly five frozen robust targets, found {len(robust)}")
    robust["panel_role"] = "robust"

    same_strata = frozen.merge(
        robust[["model", "cohort"]].drop_duplicates(), on=["model", "cohort"], how="inner"
    )
    strict = same_strata[
        same_strata.tier0_pass.eq(3)
        & same_strata.tier2_pass.eq(0)
        & ~same_strata.robustness.eq("robust_3_of_3")
    ].sort_values(["cohort", "target"])
    if len(strict) < 3:
        raise RuntimeError(f"Expected at least three strict null controls, found {len(strict)}")
    strict = strict.iloc[:3].copy()
    strict["panel_role"] = "strict_null"

    selected_keys = set(zip(robust.cohort, robust.target)) | set(zip(strict.cohort, strict.target))
    near = same_strata[
        same_strata.tier0_pass.eq(3)
        & same_strata.tier2_pass.le(1)
        & ~same_strata.robustness.eq("robust_3_of_3")
        & ~pd.Series(list(zip(same_strata.cohort, same_strata.target)), index=same_strata.index).isin(selected_keys)
    ].copy()
    near["preferred"] = near.target.isin(["qt_interval_native", "st_t_abnormal_native"])
    near = near.sort_values(["preferred", "tier2_pass", "ste_mean", "cohort", "target"], ascending=[False, True, True, True, True])
    if len(near) < 2:
        raise RuntimeError(f"Expected at least two near-null controls, found {len(near)}")
    near = near.iloc[:2].copy()
    near["panel_role"] = "near_null_tier2_le1"

    panel = pd.concat([robust, strict, near], ignore_index=True)
    columns = [
        "model", "cohort", "target", "panel_role", "seeds", "tier0_pass", "tier1_pass",
        "tier2_pass", "tier3_pass", "ste_mean", "wbi_median", "robustness",
    ]
    panel = panel[columns].sort_values(["panel_role", "cohort", "target"]).reset_index(drop=True)
    if len(panel) != 10:
        raise RuntimeError(f"Frozen panel must contain 10 targets, found {len(panel)}")
    return panel


def cache_for(result: dict) -> Path:
    candidates = sorted(
        (BASE / result["model_suffix"] / result["cohort"] / "steering_cache" / "source").glob(
            f"seed{int(result['seed'])}_N*_k*"
        )
    )
    valid = [path for path in candidates if (path / "complete.json").exists()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one source steering cache for {result}, found {valid}")
    return valid[0]


def matched_random_groups(
    selected: np.ndarray,
    freq: np.ndarray,
    mag: np.ndarray,
    rankings: np.ndarray,
    n_random: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    excluded = np.unique(rankings[:, :10])
    groups = []
    for _ in range(n_random):
        group = []
        for atom in selected:
            distance = np.abs(np.log((freq + 1e-6) / (freq[atom] + 1e-6)))
            distance += np.abs(np.log((mag + 1e-6) / (mag[atom] + 1e-6)))
            blocked = np.unique(np.concatenate([excluded, np.asarray(group, dtype=int)]))
            distance[blocked] = np.inf
            candidates = np.argsort(distance)[:200]
            candidates = candidates[np.isfinite(distance[candidates])]
            if not len(candidates):
                raise RuntimeError("No matched random atom candidate")
            group.append(int(rng.choice(candidates)))
        groups.append(group)
    return np.asarray(groups, dtype=int)


def atom_delta(
    zte: np.ndarray,
    gradients: np.ndarray,
    centroid: np.ndarray,
    indices: np.ndarray,
    mode: str,
    alpha: float,
) -> np.ndarray:
    idx = np.asarray(indices, dtype=int)
    if mode == "centroid_scale":
        dz = alpha * (centroid[idx][None, :] - zte[:, idx])
    elif mode == "zero_ablation":
        dz = -zte[:, idx]
    else:
        raise ValueError(mode)
    return np.column_stack([dz @ gradients[head, idx] for head in range(len(gradients))])


def signed_target_change(data: dict[str, np.ndarray], result: dict) -> float:
    names = data["target_names"].astype(str).tolist()
    kinds = data["target_types"].astype(str).tolist()
    target_j = names.index(result["target"])
    labels = data["labels"][:, target_j]
    focus = SUMMARY.focus_mask(
        labels, kinds[target_j], float(result["focus_thresholds_train"][result["target"]])
    )
    valid = np.isfinite(labels)
    denominator = max(float(np.nanstd(data["baseline_logits"][valid, target_j])), 1e-8)
    return float(np.nanmean(data["top5_delta"][focus, target_j])) / denominator


def main() -> None:
    args = parse_args()
    panel = freeze_panel()
    total_panel_targets = len(panel)
    if args.panel_index >= 0:
        if args.panel_index >= total_panel_targets:
            raise ValueError(f"panel-index must be in 0..{total_panel_targets - 1}")
        panel = panel.iloc[[args.panel_index]].copy()
        args.out = args.out / "workers" / f"panel_{args.panel_index:02d}"
    elif args.max_panel:
        panel = panel.iloc[: args.max_panel].copy()
    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out / "frozen_target_panel.csv", index=False)

    rows = []
    for panel_index, panel_row in enumerate(panel.itertuples(index=False)):
        result_paths = sorted(
            (BASE / "ecg_jepa_cu118_commons" / panel_row.cohort / "steering" / "frozen_atom").glob(
                f"seed*/{panel_row.target}/result.json"
            )
        )
        if len(result_paths) != 3:
            raise RuntimeError(f"{panel_row.cohort}/{panel_row.target}: expected three seeds")
        for result_path in result_paths:
            result = load_json(result_path)
            original = load_npz(result_path.with_name("records.npz"))
            cache = cache_for(result)
            zte = np.load(cache / "zte.npy", mmap_mode="r")
            gradients = np.load(cache / "gradients.npy", mmap_mode="r")
            centroid = np.load(cache / "centroid.npy", mmap_mode="r")
            freq = np.load(cache / "freq.npy", mmap_mode="r")
            mag = np.load(cache / "mag.npy", mmap_mode="r")
            rankings = np.load(cache / "rankings.npy", mmap_mode="r")
            selected_top10 = np.asarray(result["selected_atoms"]["top10"], dtype=int)
            weights, inverse = group_bootstrap_weights(
                original["patient_ids"],
                args.bootstrap,
                np.random.default_rng(
                    args.seed + int(result["seed"]) + sum(map(ord, result["cohort"] + result["target"]))
                ),
            )
            for top_k in TOP_K:
                selected = selected_top10[:top_k]
                random_groups = matched_random_groups(
                    selected,
                    freq,
                    mag,
                    rankings,
                    args.n_random,
                    args.seed + int(result["seed"]) + top_k + sum(map(ord, result["target"])),
                )
                settings = [("centroid_scale", alpha) for alpha in ALPHAS] + [("zero_ablation", np.nan)]
                for mode, alpha in settings:
                    effective_alpha = 0.0 if np.isnan(alpha) else float(alpha)
                    target_delta = atom_delta(zte, gradients, centroid, selected, mode, effective_alpha)
                    random_delta = np.stack(
                        [atom_delta(zte, gradients, centroid, group, mode, effective_alpha) for group in random_groups],
                        axis=1,
                    )
                    data = dict(original)
                    data["top5_delta"] = target_delta.astype(np.float32)
                    data["random_top5_delta"] = random_delta.astype(np.float32)
                    point = SUMMARY.one_stats(data, result, [])
                    samples = bootstrap_steering_metrics(data, result, weights, inverse)
                    row = {
                        "model": result["model"],
                        "model_suffix": result["model_suffix"],
                        "cohort": result["cohort"],
                        "target": result["target"],
                        "family": result["family"],
                        "panel_role": panel_row.panel_role,
                        "seed": int(result["seed"]),
                        "top_k": top_k,
                        "mode": mode,
                        "alpha": alpha,
                        "direction": (
                            "enhance_away_from_centroid"
                            if mode == "centroid_scale" and effective_alpha < 0
                            else "neutralize_toward_centroid"
                            if mode == "centroid_scale" and effective_alpha > 0
                            else "identity"
                            if mode == "centroid_scale"
                            else "zero_ablation"
                        ),
                        "signed_target_change": signed_target_change(data, result),
                        "bootstrap_samples": args.bootstrap,
                        "n_random": args.n_random,
                    }
                    for metric in METRICS:
                        row[metric] = float(point[metric])
                        stats = interval_and_p(samples[metric], -1 if metric in {"otd_mean", "wbi"} else 1)
                        row[f"{metric}_ci_low"] = stats["ci_low"]
                        row[f"{metric}_ci_high"] = stats["ci_high"]
                        row[f"{metric}_p_one_sided"] = stats["p_one_sided"]
                    rows.append(row)
            print(
                f"dose panel {panel_index + 1}/{len(panel)}: {result['cohort']}/{result['target']}/seed{result['seed']}",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    for metric in METRICS:
        frame[f"{metric}_q"] = bh(frame[f"{metric}_p_one_sided"].to_numpy())
    frame.to_csv(args.out / "dose_direction_seed_cells.csv", index=False)
    profile = frame.groupby(
        ["model", "cohort", "target", "family", "panel_role", "top_k", "mode", "alpha", "direction"],
        dropna=False,
        as_index=False,
    ).agg(
        seeds=("seed", "nunique"),
        signed_target_change_mean=("signed_target_change", "mean"),
        ste_mean=("ste", "mean"),
        otd_mean=("otd_mean", "mean"),
        wbi_mean=("wbi", "mean"),
        excess_selectivity_mean=("excess_selectivity", "mean"),
        behavior_excess_mean=("behavior_excess", "mean"),
        selectivity_q05_seeds=("excess_selectivity_q", lambda x: int((x < 0.05).sum())),
        behavior_q05_seeds=("behavior_excess_q", lambda x: int((x < 0.05).sum())),
    )
    profile.to_csv(args.out / "dose_direction_profile.csv", index=False)

    centroid = profile[profile["mode"].eq("centroid_scale")].copy()
    monotonic_rows = []
    for keys, group in centroid.groupby(["model", "cohort", "target", "panel_role", "top_k"]):
        positive = group[group.alpha.ge(0)].sort_values("alpha")
        negative = group[group.alpha.le(0)].sort_values("alpha")
        rho_behavior = spearmanr(positive.alpha, positive.behavior_excess_mean).statistic
        rho_signed = spearmanr(positive.alpha, positive.signed_target_change_mean).statistic
        at_pos = group[np.isclose(group.alpha, 1.0)].signed_target_change_mean
        at_neg = group[np.isclose(group.alpha, -1.0)].signed_target_change_mean
        reversal = bool(len(at_pos) == 1 and len(at_neg) == 1 and float(at_pos.iloc[0]) * float(at_neg.iloc[0]) < 0)
        monotonic_rows.append(
            dict(
                zip(("model", "cohort", "target", "panel_role", "top_k"), keys),
                positive_dose_signed_rho=float(rho_signed),
                positive_dose_behavior_rho=float(rho_behavior),
                alpha_plus_minus_one_sign_reversal=reversal,
                positive_doses=len(positive),
                negative_doses=len(negative),
            )
        )
    monotonic = pd.DataFrame(monotonic_rows)
    monotonic.to_csv(args.out / "dose_monotonicity_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "panel_targets": len(panel),
        "total_panel_targets": total_panel_targets,
        "panel_index": args.panel_index,
        "top_k": list(TOP_K),
        "centroid_alphas": list(ALPHAS),
        "zero_ablation": True,
        "bootstrap_samples": args.bootstrap,
        "fdr_family": "all panel x seed x top-k x dose/mode cells, separately by metric",
        "interpretation_note": "Frozen linear readouts make logit deltas algebraically linear in alpha; behavior/selectivity tradeoffs are primary.",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(profile.groupby(["panel_role", "mode"]).size().to_string())


if __name__ == "__main__":
    main()
