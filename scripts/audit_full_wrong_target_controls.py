#!/usr/bin/env python
"""Strengthen v2.1 Tier-2 using every non-nuisance target as a wrong-atom control."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_1_finegrained"
SPEC = importlib.util.spec_from_file_location(
    "single_summary", ROOT / "scripts/summarize_steering_benchmark.py"
)
SINGLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SINGLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = pd.read_csv(args.base / "target_registry.csv").set_index("target")
    existing = pd.read_csv(args.base / "summary/multimodel_steering_cells.csv")
    rows = []
    for result_path in sorted((args.base / "models").glob("*/tasks/seed*/*/result.json")):
        result = json.loads(result_path.read_text())
        with np.load(result_path.with_name("records.npz"), allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        names = data["target_names"].astype(str).tolist()
        target = result["target"]
        target_j = names.index(target)
        safe = result["model"].lower().replace("-", "_")
        candidates = sorted(
            (args.base / "models" / safe / "shared_cache").glob(
                f"seed{int(result['seed'])}_N*_k*"
            )
        )
        candidates = [path for path in candidates if (path / "complete.json").exists()]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one cache for {result['model']} seed {result['seed']}")
        cache = candidates[0]
        rankings = np.load(cache / "rankings.npy", mmap_mode="r")
        gradients = np.load(cache / "gradients.npy", mmap_mode="r")
        zte = np.load(cache / "zte.npy", mmap_mode="r")
        centroid = np.load(cache / "centroid.npy", mmap_mode="r")
        wrong_names = [
            name
            for name in names
            if name != target and registry.loc[name, "analysis_role"] != "nuisance_control"
        ]
        wrong_columns = []
        for wrong_name in wrong_names:
            wrong_j = names.index(wrong_name)
            atoms = np.asarray(rankings[wrong_j, :5], dtype=int)
            wrong_columns.append(
                np.asarray(
                    (centroid[atoms][None, :] - zte[:, atoms])
                    @ gradients[target_j, atoms],
                    dtype=np.float64,
                )
            )
        wrong_delta = np.column_stack(wrong_columns)
        labels = data["labels"][:, target_j]
        kind = data["target_types"].astype(str).tolist()[target_j]
        focus = SINGLE.focus_mask(
            labels, kind, float(result["focus_thresholds_train"][target])
        )
        valid = np.isfinite(labels)
        target_delta = np.asarray(data["top5_delta"][:, target_j], dtype=np.float64)
        base = np.asarray(data["baseline_logits"][:, target_j], dtype=np.float64)
        denominator = max(float(np.nanstd(base[valid])), 1e-8)
        target_effect = abs(float(target_delta[focus].mean())) / denominator
        wrong_effects = np.abs(wrong_delta[focus].mean(axis=0)) / denominator
        max_index = int(np.argmax(wrong_effects))
        max_wrong = float(wrong_effects[max_index])
        margin = target_effect - max_wrong

        patient_ids = data["patient_ids"].astype(str)
        _, inverse = np.unique(patient_ids, return_inverse=True)
        n_patients = int(inverse.max()) + 1
        rng = np.random.default_rng(
            args.seed
            + int(result["seed"])
            + sum(map(ord, result["model"] + target))
        )
        weights = rng.multinomial(
            n_patients,
            np.full(n_patients, 1.0 / n_patients),
            size=args.bootstrap,
        ).astype(np.float64)

        def patient_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            out = np.zeros((n_patients,) + values.shape[1:], dtype=np.float64)
            np.add.at(out, inverse[mask], values[mask])
            return out

        focus_count = weights @ patient_sum(np.ones((len(labels), 1)), focus)
        valid_count = weights @ patient_sum(np.ones((len(labels), 1)), valid)
        base_sum = weights @ patient_sum(base[:, None], valid)
        base_sq = weights @ patient_sum((base**2)[:, None], valid)
        sd = np.sqrt(
            np.maximum(
                base_sq / np.maximum(valid_count, 1.0)
                - (base_sum / np.maximum(valid_count, 1.0)) ** 2,
                1e-12,
            )
        )[:, 0]
        target_mean = (
            weights @ patient_sum(target_delta[:, None], focus)
        )[:, 0] / np.maximum(focus_count[:, 0], 1.0)
        wrong_mean = (
            weights @ patient_sum(wrong_delta, focus)
        ) / np.maximum(focus_count, 1.0)
        bootstrap_margin = np.abs(target_mean) / sd - (np.abs(wrong_mean) / sd[:, None]).max(axis=1)
        low, high = np.quantile(bootstrap_margin, [0.025, 0.975])
        p_value = (1.0 + float((bootstrap_margin <= 0).sum())) / (args.bootstrap + 1.0)
        rows.append(
            {
                "model": result["model"],
                "target": target,
                "family": result["family"],
                "seed": int(result["seed"]),
                "wrong_target_controls": len(wrong_names),
                "target_ste": target_effect,
                "max_full_registry_wrong_ste": max_wrong,
                "max_full_registry_wrong_target": wrong_names[max_index],
                "full_registry_wrong_margin": margin,
                "full_registry_wrong_margin_ci_low": low,
                "full_registry_wrong_margin_ci_high": high,
                "full_registry_wrong_margin_p_one_sided": p_value,
                "bootstrap_samples": args.bootstrap,
            }
        )
    audit = pd.DataFrame(rows)
    audit["full_registry_wrong_margin_q"] = np.nan
    for _, indices in audit.groupby(["model", "family"]).groups.items():
        audit.loc[indices, "full_registry_wrong_margin_q"] = SINGLE.bh(
            audit.loc[indices, "full_registry_wrong_margin_p_one_sided"].to_numpy()
        )
    audit["full_registry_wrong_pass"] = (
        (audit.full_registry_wrong_margin_ci_low > 0)
        & (audit.full_registry_wrong_margin_q < 0.05)
    )
    merged = existing.merge(
        audit,
        on=["model", "target", "family", "seed"],
        how="left",
        validate="one_to_one",
    )
    if merged.full_registry_wrong_margin_ci_low.isna().any():
        raise RuntimeError("Missing full-registry wrong-target audit rows")
    merged["tier2_selective_steering_full_registry"] = (
        merged.tier0_fidelity.astype(bool)
        & (merged.excess_selectivity_ci_low > 0)
        & (merged.wbi_improvement_ci_low > 0)
        & merged.full_registry_wrong_pass.astype(bool)
        & (merged.excess_selectivity_q < 0.05)
        & (merged.wbi_improvement_q < 0.05)
    )
    out = args.base / "summary/full_wrong_target_audit"
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "full_registry_wrong_target_cells.csv", index=False)
    merged.to_csv(out / "strict_multimodel_steering_cells.csv", index=False)
    profile = merged.groupby(["model", "target", "family"], as_index=False).agg(
        seeds=("seed", "nunique"),
        original_tier2_pass=("tier2_selective_steering", "sum"),
        strict_tier2_pass=("tier2_selective_steering_full_registry", "sum"),
        strict_margin_mean=("full_registry_wrong_margin", "mean"),
    )
    profile["strict_robust_3_of_3"] = profile.strict_tier2_pass.eq(3)
    profile.to_csv(out / "strict_multimodel_target_profile.csv", index=False)
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
