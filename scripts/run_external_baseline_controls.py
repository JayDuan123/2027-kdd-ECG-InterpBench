#!/usr/bin/env python
"""Norm-matched SAE, supervised-CAV, PCA, and random-direction controls."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "external_benchmark_v1"
ACT_ROOT = ROOT / "results" / "activations_external_full_v1" / "pooled"
OUT = ROOT / "results" / "benchmark_extension_v1" / "baseline_controls" / "workers"

from scripts.analyze_external_dose_direction import freeze_panel  # noqa: E402
from scripts.benchmark_extension_common import (  # noqa: E402
    bootstrap_steering_metrics,
    group_bootstrap_weights,
    interval_and_p,
    load_json,
    load_npz,
)
from scripts.run_external_sae_steering_task import (  # noqa: E402
    load_activations,
    threshold_at_specificity,
)

SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "steering_summary", ROOT / "scripts" / "summarize_steering_benchmark.py"
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY)

METHOD_METRICS = ("ste", "otd_mean", "selectivity_margin", "wbi", "behavior_effect")
METHODS = ("sae_top5", "supervised_cav", "pca_top5", "random_orthogonal_5d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, choices=("chapman_f", "ningbo_f", "mimic_f"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def source_cache(result: dict) -> Path:
    candidates = sorted(
        (BASE / result["model_suffix"] / result["cohort"] / "steering_cache" / "source").glob(
            f"seed{int(result['seed'])}_N*_k*"
        )
    )
    valid = [path for path in candidates if (path / "complete.json").exists()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one source cache, found {valid}")
    return valid[0]


def norm_match(delta: np.ndarray, reference_norm: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(delta, axis=1)
    out = np.asarray(delta, dtype=np.float64).copy()
    good = norm > 1e-10
    out[good] *= (reference_norm[good] / norm[good])[:, None]
    if (~good).any():
        unit = fallback / max(float(np.linalg.norm(fallback)), 1e-12)
        out[~good] = reference_norm[~good, None] * unit[None, :]
    return out


def logit_delta(delta_standardized: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(delta_standardized @ coefficients.T, dtype=np.float32)


def method_point(data: dict[str, np.ndarray], result: dict) -> dict[str, float]:
    point = SUMMARY.one_stats(data, result, [])
    return {metric: float(point[metric]) for metric in METHOD_METRICS}


def random_point(data: dict[str, np.ndarray], result: dict) -> dict[str, float]:
    point = SUMMARY.one_stats(data, result, [])
    return {
        "ste": float(point["random_ste_mean"]),
        "otd_mean": float(point["random_ste_mean"] - point["random_margin_mean"]),
        "selectivity_margin": float(point["random_margin_mean"]),
        "wbi": float(point["random_wbi_mean"]),
        "behavior_effect": float(point["random_behavior_mean"]),
    }


def main() -> None:
    args = parse_args()
    panel = freeze_panel()
    panel = panel[panel.cohort.eq(args.cohort)].copy()
    if args.max_targets:
        panel = panel.iloc[: args.max_targets].copy()
    if panel.empty:
        raise RuntimeError(f"No frozen panel targets for {args.cohort}")

    model_suffix = "ecg_jepa_cu118_commons"
    pair_root = BASE / model_suffix / args.cohort
    bundle = joblib.load(pair_root / "frozen_heads.joblib")
    names = list(bundle["targets"])
    heads = bundle["heads"]
    scaler = bundle["scaler"]
    record_ids = np.asarray(bundle["record_ids"])
    group_ids = np.asarray(bundle.get("group_ids", record_ids))
    split = np.asarray(bundle["split"])
    labels = {name: np.asarray(heads[name]["labels"], dtype=float) for name in names}
    x, loaded_ids = load_activations(ACT_ROOT / model_suffix / args.cohort)
    if not np.array_equal(record_ids.astype(str), loaded_ids.astype(str)):
        raise RuntimeError("Activation/head record order mismatch")
    train_idx = np.where(split == "train")[0]
    val_idx = np.where(split == "val")[0]
    test_idx = np.where(split == "test")[0]
    x_train = scaler.transform(np.asarray(x[train_idx], dtype=np.float32)).astype(np.float32)
    x_val = scaler.transform(np.asarray(x[val_idx], dtype=np.float32)).astype(np.float32)
    x_test = scaler.transform(np.asarray(x[test_idx], dtype=np.float32)).astype(np.float32)
    train_mean = x_train.mean(axis=0)
    coefficients = np.vstack([np.asarray(heads[name]["clf"].coef_).reshape(-1) for name in names])
    base_val = np.column_stack([heads[name]["clf"].decision_function(x_val) for name in names]).astype(np.float32)
    base_test = np.column_stack([heads[name]["clf"].decision_function(x_test) for name in names]).astype(np.float32)
    thresholds = np.asarray(
        [threshold_at_specificity(labels[name][val_idx].astype(int), base_val[:, j]) for j, name in enumerate(names)],
        dtype=np.float32,
    )

    n_components = min(args.pca_components, x_train.shape[1], len(x_train) - 1)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=args.seed, iterated_power=3)
    train_scores = pca.fit_transform(x_train)
    test_centered = x_test - pca.mean_
    test_scores = pca.transform(x_test)
    print(f"fit PCA cohort={args.cohort} train={len(train_idx)} components={n_components}", flush=True)

    method_rows = []
    contrast_rows = []
    for panel_row in panel.itertuples(index=False):
        target = panel_row.target
        target_j = names.index(target)
        y_train = labels[target][train_idx].astype(int)
        positive = y_train == 1
        negative = y_train == 0
        cav = x_train[positive].mean(axis=0) - x_train[negative].mean(axis=0)
        cav /= max(float(np.linalg.norm(cav)), 1e-12)
        cav_score = (x_test - train_mean) @ cav
        cav_raw = -cav_score[:, None] * cav[None, :]
        pca_association = train_scores[positive].mean(axis=0) - train_scores[negative].mean(axis=0)
        pca_selected = np.argsort(np.abs(pca_association))[::-1][:5]
        pca_raw = -(test_scores[:, pca_selected] @ pca.components_[pca_selected])

        result_paths = sorted(
            (pair_root / "steering" / "frozen_atom").glob(f"seed*/{target}/result.json")
        )
        if len(result_paths) != 3:
            raise RuntimeError(f"{args.cohort}/{target}: expected three frozen seeds")
        for result_path in result_paths:
            result = load_json(result_path)
            existing = load_npz(result_path.with_name("records.npz"))
            if not np.array_equal(existing["patient_ids"].astype(str), group_ids[test_idx].astype(str)):
                raise RuntimeError(f"{result_path}: test group order mismatch")
            cache = source_cache(result)
            zte = np.load(cache / "zte.npy", mmap_mode="r")
            centroid = np.load(cache / "centroid.npy", mmap_mode="r")
            selected = np.asarray(result["selected_atoms"]["top5"], dtype=int)
            import torch
            saved = torch.load(Path(result["checkpoint"]), map_location="cpu")
            state = saved["model"]
            decoder = state["W_dec"].numpy()
            sigma = state["sigma"].numpy()
            dz = centroid[selected][None, :] - zte[:, selected]
            sae_raw = (dz @ decoder[:, selected].T) * sigma[None, :]
            sae_standardized = sae_raw / scaler.scale_[None, :]
            reference_norm = np.linalg.norm(sae_standardized, axis=1)
            sae_logit_delta = logit_delta(sae_standardized, coefficients)
            parity = float(np.max(np.abs(sae_logit_delta - existing["top5_delta"])))
            if parity > 2e-4:
                raise RuntimeError(f"SAE logit-delta parity failed: max_abs={parity}")

            cav_delta = norm_match(cav_raw, reference_norm, -cav)
            pca_fallback = -pca.components_[pca_selected[0]]
            pca_delta = norm_match(pca_raw, reference_norm, pca_fallback)
            rng = np.random.default_rng(
                args.seed + int(result["seed"]) + sum(map(ord, args.cohort + target))
            )
            random_standardized = []
            for _ in range(args.n_random):
                q, _ = np.linalg.qr(rng.normal(size=(x_test.shape[1], 5)))
                raw = -((test_centered @ q) @ q.T)
                random_standardized.append(norm_match(raw, reference_norm, -q[:, 0]))
            random_logit = np.stack(
                [logit_delta(delta, coefficients) for delta in random_standardized], axis=1
            )

            common = {
                "patient_ids": group_ids[test_idx].astype("U64"),
                "target_names": np.asarray(names, dtype="U64"),
                "target_types": np.asarray(["binary"] * len(names), dtype="U16"),
                "labels": np.column_stack([labels[name][test_idx] for name in names]).astype(np.float32),
                "baseline_logits": base_test,
                "thresholds_95spec": thresholds,
                "continuous_target_means": np.full(len(names), np.nan),
                "continuous_target_stds": np.full(len(names), np.nan),
            }
            method_data = {}
            for method, target_delta, random_delta in (
                ("sae_top5", sae_logit_delta, existing["random_top5_delta"]),
                ("supervised_cav", logit_delta(cav_delta, coefficients), random_logit),
                ("pca_top5", logit_delta(pca_delta, coefficients), random_logit),
            ):
                data = dict(common)
                data["top5_delta"] = np.asarray(target_delta, dtype=np.float32)
                data["random_top5_delta"] = np.asarray(random_delta, dtype=np.float32)
                method_data[method] = data

            weights, inverse = group_bootstrap_weights(
                common["patient_ids"],
                args.bootstrap,
                np.random.default_rng(
                    args.seed + 100000 + int(result["seed"]) + sum(map(ord, args.cohort + target))
                ),
            )
            point = {}
            samples = {}
            for method, data in method_data.items():
                point[method] = method_point(data, result)
                samples[method] = bootstrap_steering_metrics(data, result, weights, inverse)
            point["random_orthogonal_5d"] = random_point(method_data["pca_top5"], result)
            random_samples = samples["pca_top5"]
            samples["random_orthogonal_5d"] = {
                "ste": random_samples["random_ste_mean"],
                "otd_mean": random_samples["random_otd_mean"],
                "selectivity_margin": random_samples["random_selectivity_margin_mean"],
                "wbi": random_samples["random_wbi_mean"],
                "behavior_effect": random_samples["random_behavior_mean"],
            }

            for method in METHODS:
                row = {
                    "model": result["model"], "cohort": args.cohort, "target": target,
                    "family": result["family"], "panel_role": panel_row.panel_role,
                    "seed": int(result["seed"]), "method": method,
                    "bootstrap_samples": args.bootstrap, "n_random": args.n_random,
                    "sae_delta_parity_max_abs": parity,
                    "activation_l2_rms": float(np.sqrt(np.mean(reference_norm**2))),
                }
                for metric in METHOD_METRICS:
                    row[metric] = point[method][metric]
                    stats = interval_and_p(samples[method][metric], -1 if metric in {"otd_mean", "wbi"} else 1)
                    row[f"{metric}_ci_low"] = stats["ci_low"]
                    row[f"{metric}_ci_high"] = stats["ci_high"]
                method_rows.append(row)

            for method in ("supervised_cav", "pca_top5", "random_orthogonal_5d"):
                row = {
                    "model": result["model"], "cohort": args.cohort, "target": target,
                    "family": result["family"], "panel_role": panel_row.panel_role,
                    "seed": int(result["seed"]), "contrast": f"{method}_minus_sae",
                    "method": method, "reference": "sae_top5", "bootstrap_samples": args.bootstrap,
                }
                for metric in METHOD_METRICS:
                    delta = point[method][metric] - point["sae_top5"][metric]
                    difference = samples[method][metric] - samples["sae_top5"][metric]
                    stats = interval_and_p(difference, -1 if metric in {"otd_mean", "wbi"} else 1)
                    row[f"delta_{metric}"] = delta
                    row[f"delta_{metric}_ci_low"] = stats["ci_low"]
                    row[f"delta_{metric}_ci_high"] = stats["ci_high"]
                    row[f"delta_{metric}_p_one_sided"] = stats["p_one_sided"]
                    row[f"delta_{metric}_p_two_sided"] = stats["p_two_sided"]
                contrast_rows.append(row)
        print(f"baseline cohort={args.cohort} target={target}", flush=True)

    worker_out = args.out / args.cohort
    worker_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(method_rows).to_csv(worker_out / "method_seed_cells.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(worker_out / "paired_method_contrasts.csv", index=False)
    metadata = {
        "schema_version": 1, "cohort": args.cohort, "targets": len(panel),
        "seeds": 3, "methods": list(METHODS), "pca_components": n_components,
        "n_random": args.n_random, "bootstrap_samples": args.bootstrap,
        "norm_matching": "exact per-test-record L2 in frozen-head standardized activation space",
        "baseline_logits": "raw frozen-head logits shared by all methods",
    }
    (worker_out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(method_rows).groupby("method").ste.mean().to_string())


if __name__ == "__main__":
    main()
