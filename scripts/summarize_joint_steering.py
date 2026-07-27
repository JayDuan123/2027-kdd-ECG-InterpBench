#!/usr/bin/env python
"""Patient-bootstrap summary for family and cross-family joint steering."""
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
    "single_summary", ROOT / "scripts/summarize_steering_benchmark.py"
)
SINGLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SINGLE)
WBI_EPS = SINGLE.WBI_EPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def point_stats(data: dict[str, np.ndarray], result: dict, scheme: str) -> dict[str, float]:
    names = data["target_names"].astype(str).tolist()
    kinds = data["target_types"].astype(str).tolist()
    members = np.asarray(result["member_indices"], dtype=int)
    off = np.asarray(result["offtarget_indices"], dtype=int)
    labels = data["labels"]
    base = data["baseline_logits"]
    delta = data[f"{scheme}_delta"]
    random_delta = data[f"random_{scheme}_delta"]
    focus_thresholds = result["focus_thresholds_train"]

    def effect(values: np.ndarray, head: int) -> float:
        focus = SINGLE.focus_mask(
            labels[:, head], kinds[head], float(focus_thresholds[names[head]])
        )
        valid = np.isfinite(labels[:, head])
        denominator = max(float(np.nanstd(base[valid, head])), 1e-8)
        return abs(float(np.nanmean(values[focus, head]))) / denominator

    effects = np.asarray([effect(delta, head) for head in range(len(names))])
    random_effects = np.asarray(
        [
            [effect(random_delta[:, random_index], head) for head in range(len(names))]
            for random_index in range(random_delta.shape[1])
        ]
    )
    target_effect = float(effects[members].mean())
    off_damage = float(effects[off].mean())
    random_target = random_effects[:, members].mean(axis=1)
    random_off = random_effects[:, off].mean(axis=1)
    selectivity = target_effect - off_damage
    random_selectivity = random_target - random_off
    wbi = off_damage / (target_effect + WBI_EPS)
    random_wbi = random_off / (random_target + WBI_EPS)
    return {
        "target_effect": target_effect,
        "offtarget_damage": off_damage,
        "selectivity": selectivity,
        "random_target_effect_mean": float(random_target.mean()),
        "random_offtarget_damage_mean": float(random_off.mean()),
        "random_selectivity_mean": float(random_selectivity.mean()),
        "excess_selectivity": selectivity - float(random_selectivity.mean()),
        "wbi": wbi,
        "random_wbi_mean": float(random_wbi.mean()),
        "wbi_improvement": float(random_wbi.mean()) - wbi,
    }


def bootstrap(
    data: dict[str, np.ndarray], result: dict, scheme: str, n_boot: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    names = data["target_names"].astype(str).tolist()
    kinds = data["target_types"].astype(str).tolist()
    members = np.asarray(result["member_indices"], dtype=int)
    off = np.asarray(result["offtarget_indices"], dtype=int)
    labels = data["labels"]
    base = data["baseline_logits"]
    delta = data[f"{scheme}_delta"]
    random_delta = data[f"random_{scheme}_delta"]
    _, inverse = np.unique(data["patient_ids"].astype(str), return_inverse=True)
    n_patients = int(inverse.max()) + 1
    weights = rng.multinomial(
        n_patients, np.full(n_patients, 1.0 / n_patients), size=n_boot
    ).astype(np.float64)
    n_records = len(inverse)
    focus_thresholds = result["focus_thresholds_train"]

    def patient_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        out = np.zeros((n_patients,) + values.shape[1:], dtype=np.float64)
        np.add.at(out, inverse[mask], values[mask])
        return out

    effects = []
    random_effects = []
    for head, name in enumerate(names):
        valid = np.isfinite(labels[:, head])
        focus = SINGLE.focus_mask(
            labels[:, head], kinds[head], float(focus_thresholds[name])
        )
        valid_count = weights @ patient_sum(np.ones((n_records, 1)), valid)
        base_sum = weights @ patient_sum(base[:, head : head + 1], valid)
        base_sq = weights @ patient_sum(base[:, head : head + 1] ** 2, valid)
        variance = np.maximum(
            base_sq / np.maximum(valid_count, 1.0)
            - (base_sum / np.maximum(valid_count, 1.0)) ** 2,
            1e-12,
        )
        denominator = np.sqrt(variance)
        focus_count = weights @ patient_sum(np.ones((n_records, 1)), focus)
        value_sum = weights @ patient_sum(delta[:, head : head + 1], focus)
        random_sum = weights @ patient_sum(random_delta[:, :, head], focus)
        effects.append(np.abs(value_sum[:, 0] / np.maximum(focus_count[:, 0], 1.0)) / denominator[:, 0])
        random_effects.append(
            np.abs(random_sum / np.maximum(focus_count, 1.0)) / denominator
        )
    effects = np.column_stack(effects)
    random_effects = np.stack(random_effects, axis=2)
    target = effects[:, members].mean(axis=1)
    off_damage = effects[:, off].mean(axis=1)
    random_target = random_effects[:, :, members].mean(axis=2)
    random_off = random_effects[:, :, off].mean(axis=2)
    selectivity = target - off_damage
    random_selectivity = random_target - random_off
    wbi = off_damage / (target + WBI_EPS)
    random_wbi = random_off / (random_target + WBI_EPS)
    return {
        "excess_selectivity": selectivity - random_selectivity.mean(axis=1),
        "wbi_improvement": random_wbi.mean(axis=1) - wbi,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for result_path in sorted(
        (args.base / "joint_steering/tasks").glob("*/seed*/*/result.json")
    ):
        result = json.loads(result_path.read_text())
        with np.load(result_path.with_name("records.npz"), allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        for scheme in ("top5_union", "top10_union"):
            point = point_stats(data, result, scheme)
            rng = np.random.default_rng(
                args.seed
                + int(result["seed"])
                + sum(map(ord, result["model"] + result["group_id"] + scheme))
            )
            samples = bootstrap(data, result, scheme, args.bootstrap, rng)
            row = {
                "model": result["model"],
                "group_id": result["group_id"],
                "group_type": result["group_type"],
                "family_scope": result["family_scope"],
                "members": ";".join(result["members"]),
                "member_count": len(result["members"]),
                "seed": int(result["seed"]),
                "scheme": scheme,
                "selected_atom_count": len(result["selected_atoms"][scheme]),
                "bootstrap_samples": args.bootstrap,
                **point,
            }
            for metric in ("excess_selectivity", "wbi_improvement"):
                values = samples[metric]
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(
                    values, [0.025, 0.975]
                )
                row[f"{metric}_p_one_sided"] = (1.0 + float((values <= 0).sum())) / (
                    len(values) + 1.0
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No completed joint steering records")
    for metric in ("excess_selectivity", "wbi_improvement"):
        frame[f"{metric}_q"] = np.nan
        for _, indices in frame.groupby(["model", "group_type", "scheme"]).groups.items():
            frame.loc[indices, f"{metric}_q"] = SINGLE.bh(
                frame.loc[indices, f"{metric}_p_one_sided"].to_numpy()
            )
    frame["selective_vs_random"] = (
        (frame.excess_selectivity_ci_low > 0)
        & (frame.wbi_improvement_ci_low > 0)
        & (frame.excess_selectivity_q < 0.05)
        & (frame.wbi_improvement_q < 0.05)
    )
    output = args.base / "joint_steering/summary"
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "joint_steering_seed_cells.csv", index=False)
    profile = frame.groupby(
        ["model", "group_id", "group_type", "family_scope", "members", "scheme"],
        as_index=False,
    ).agg(
        seeds=("seed", "nunique"),
        selected_atoms_mean=("selected_atom_count", "mean"),
        target_effect_mean=("target_effect", "mean"),
        offtarget_damage_mean=("offtarget_damage", "mean"),
        excess_selectivity_mean=("excess_selectivity", "mean"),
        wbi_median=("wbi", "median"),
        selective_seed_pass=("selective_vs_random", "sum"),
    )
    profile["robust_selective"] = profile.selective_seed_pass.eq(3)
    profile.to_csv(output / "joint_steering_profile.csv", index=False)
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
