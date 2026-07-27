#!/usr/bin/env python
"""Aggregate fair-comparison workers with crossed-factor inference and stability."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_common import bh  # noqa: E402
from scripts.benchmark_extension_v2_common import interval_and_p  # noqa: E402
from scripts.method_comparison_common import (  # noqa: E402
    BASE,
    LABEL_BUDGETS,
    METHOD_METRICS,
    METHODS,
    RECONSTRUCTIVE_METHODS,
    RATE_DISTORTION_K,
    SEEDS,
    stable_seed,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def worker_path(base: Path, row: pd.Series) -> Path:
    return (
        base
        / "workers"
        / f"task_{int(row.task_index):03d}_{row.model_suffix}_{row.cohort}_seed{int(row.seed)}"
    )


def crossed_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    factors: list[str],
    samples: int,
    seed: int,
) -> np.ndarray:
    values = frame[value_column].to_numpy(dtype=float)
    valid = np.isfinite(values)
    frame = frame.loc[valid].reset_index(drop=True)
    values = values[valid]
    if not len(values):
        return np.full(samples, np.nan)
    rng = np.random.default_rng(seed)
    weights = np.ones((samples, len(frame)), dtype=np.float64)
    for factor in factors:
        levels, inverse = np.unique(frame[factor].astype(str), return_inverse=True)
        factor_weights = rng.multinomial(
            len(levels), np.full(len(levels), 1.0 / len(levels)), size=samples
        ).astype(np.float64)
        weights *= factor_weights[:, inverse]
    denominator = weights.sum(axis=1)
    numerator = weights @ values
    return np.divide(
        numerator,
        denominator,
        out=np.full(samples, np.nan),
        where=denominator > 0,
    )


def hierarchical_inference(
    contrasts: pd.DataFrame, bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_averaged = (
        contrasts.groupby(
            ["model", "model_suffix", "cohort", "target", "family", "regime", "method", "reference"],
            as_index=False,
        )[[f"delta_{metric}" for metric in METHOD_METRICS]]
        .mean()
    )
    rows = []
    leave_rows = []
    for (regime, method, reference), group in seed_averaged.groupby(
        ["regime", "method", "reference"], sort=True
    ):
        for metric in METHOD_METRICS:
            column = f"delta_{metric}"
            samples = crossed_bootstrap(
                group,
                column,
                ["model", "cohort", "family"],
                bootstrap,
                stable_seed("method-hierarchical", regime, method, reference, metric),
            )
            stats = interval_and_p(samples)
            mean_delta = float(group[column].mean())
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "reference": reference,
                    "metric": metric,
                    "target_units": len(group),
                    "models": int(group.model.nunique()),
                    "cohorts": int(group.cohort.nunique()),
                    "families": int(group.family.nunique()),
                    "mean_delta": mean_delta,
                    **stats,
                    "bootstrap_samples": bootstrap,
                }
            )
            for factor in ("model", "cohort", "family"):
                for level in sorted(group[factor].astype(str).unique()):
                    retained = group[group[factor].astype(str) != level]
                    leave_rows.append(
                        {
                            "regime": regime,
                            "method": method,
                            "reference": reference,
                            "metric": metric,
                            "left_out_factor": factor,
                            "left_out_level": level,
                            "retained_units": len(retained),
                            "leave_one_out_mean_delta": float(retained[column].mean()),
                            "full_mean_delta": mean_delta,
                        }
                    )
    inference = pd.DataFrame(rows)
    for (regime, metric), indices in inference.groupby(["regime", "metric"]).groups.items():
        inference.loc[indices, "q_two_sided"] = bh(
            inference.loc[indices, "p_two_sided"].to_numpy()
        )
    leave = pd.DataFrame(leave_rows)
    envelopes = (
        leave.groupby(["regime", "method", "reference", "metric"], as_index=False)
        .agg(
            full_mean_delta=("full_mean_delta", "first"),
            leave_one_out_min=("leave_one_out_mean_delta", "min"),
            leave_one_out_max=("leave_one_out_mean_delta", "max"),
            leave_one_out_checks=("leave_one_out_mean_delta", "size"),
        )
    )
    envelopes["leave_one_out_all_same_sign"] = (
        (envelopes.full_mean_delta > 0)
        & (envelopes.leave_one_out_min > 0)
    ) | (
        (envelopes.full_mean_delta < 0)
        & (envelopes.leave_one_out_max < 0)
    )
    return inference, leave, envelopes


def reconstruction_inference(
    reconstruction: pd.DataFrame, bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = (
        reconstruction.groupby("method", as_index=False)
        .agg(
            seed_pairs=("model", "size"),
            model_cohort_pairs=("model_suffix", lambda x: len(x) // len(SEEDS)),
            dense_recon_r2_mean=("dense_recon_r2", "mean"),
            dense_recon_r2_min=("dense_recon_r2", "min"),
            topk_recon_r2_mean=("topk_recon_r2", "mean"),
            topk_recon_r2_min=("topk_recon_r2", "min"),
            dense_normalized_mse_mean=("dense_normalized_mse", "mean"),
            topk_normalized_mse_mean=("topk_normalized_mse", "mean"),
            mean_active_coefficients=("mean_active_coefficients", "mean"),
        )
    )
    averaged = (
        reconstruction.groupby(["model", "model_suffix", "cohort", "method"], as_index=False)
        .mean(numeric_only=True)
    )
    specs = {
        "common64_capacity": ("sae_common64", [m for m in RECONSTRUCTIVE_METHODS if m != "sae_common64"]),
        "existing_sae_practical": ("sae_existing_8d", list(RECONSTRUCTIVE_METHODS)),
    }
    rows = []
    metrics = (
        "dense_recon_r2",
        "topk_recon_r2",
        "dense_normalized_mse",
        "topk_normalized_mse",
    )
    for regime, (reference, methods) in specs.items():
        reference_frame = averaged[averaged.method.eq(reference)][
            ["model", "model_suffix", "cohort", *metrics]
        ].copy()
        reference_frame = reference_frame.rename(
            columns={metric: f"{metric}_reference" for metric in metrics}
        )
        for method in methods:
            current = averaged[averaged.method.eq(method)].merge(
                reference_frame,
                on=["model", "model_suffix", "cohort"],
                validate="one_to_one",
            )
            for metric in metrics:
                current["delta"] = current[metric] - current[f"{metric}_reference"]
                samples = crossed_bootstrap(
                    current,
                    "delta",
                    ["model", "cohort"],
                    bootstrap,
                    stable_seed("reconstruction-hierarchical", regime, method, metric),
                )
                stats = interval_and_p(samples)
                rows.append(
                    {
                        "regime": regime,
                        "method": method,
                        "reference": reference,
                        "metric": metric,
                        "model_cohort_pairs": len(current),
                        "mean_delta": float(current.delta.mean()),
                        **stats,
                        "bootstrap_samples": bootstrap,
                    }
                )
    inference = pd.DataFrame(rows)
    for (regime, metric), indices in inference.groupby(["regime", "metric"]).groups.items():
        inference.loc[indices, "q_two_sided"] = bh(
            inference.loc[indices, "p_two_sided"].to_numpy()
        )
    return profile, inference


def reconstruction_matched_points(
    rate_distortion: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keys = ["task_index", "model", "model_suffix", "cohort", "seed"]
    for group_keys, group in rate_distortion.groupby(keys, sort=True):
        reference_candidates = group[
            group.method.eq("sae_common64") & group.code_budget_k.eq(5)
        ]
        if len(reference_candidates) != 1:
            raise RuntimeError(f"Missing SAE k=5 rate-distortion reference for {group_keys}")
        reference = reference_candidates.iloc[0]
        for method in RECONSTRUCTIVE_METHODS:
            if method == "sae_common64":
                continue
            candidates = group[group.method.eq(method)].copy()
            candidates["absolute_r2_gap"] = (
                candidates.recon_r2 - float(reference.recon_r2)
            ).abs()
            selected = candidates.sort_values(
                ["absolute_r2_gap", "code_budget_k"]
            ).iloc[0]
            rows.append(
                {
                    **dict(zip(keys, group_keys)),
                    "method": method,
                    "reference": "sae_common64",
                    "selected_code_budget_k": int(selected.code_budget_k),
                    "reference_code_budget_k": 5,
                    "recon_r2": float(selected.recon_r2),
                    "reference_recon_r2": float(reference.recon_r2),
                    "recon_r2_gap": float(selected.recon_r2 - reference.recon_r2),
                    "absolute_r2_gap": float(selected.absolute_r2_gap),
                    "normalized_mse": float(selected.normalized_mse),
                    "reference_normalized_mse": float(reference.normalized_mse),
                    "cosine_mean": float(selected.cosine_mean),
                    "reference_cosine_mean": float(reference.cosine_mean),
                }
            )
    points = pd.DataFrame(rows)
    profile = (
        points.groupby("method", as_index=False)
        .agg(
            model_cohort_seed_cells=("task_index", "size"),
            selected_code_budget_mean=("selected_code_budget_k", "mean"),
            selected_code_budget_median=("selected_code_budget_k", "median"),
            recon_r2_mean=("recon_r2", "mean"),
            reference_recon_r2_mean=("reference_recon_r2", "mean"),
            absolute_r2_gap_mean=("absolute_r2_gap", "mean"),
            absolute_r2_gap_max=("absolute_r2_gap", "max"),
            cosine_mean=("cosine_mean", "mean"),
            reference_cosine_mean=("reference_cosine_mean", "mean"),
        )
    )
    return points, profile


def label_budget_inference(
    frame: pd.DataFrame, bootstrap: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profile = (
        frame.groupby(["label_budget_requested", "method"], as_index=False)
        .agg(
            seed_cells=("target", "size"),
            actual_labels_mean=("label_budget_actual", "mean"),
            ste_mean=("ste", "mean"),
            otd_mean=("otd_mean", "mean"),
            selectivity_margin_mean=("selectivity_margin", "mean"),
            wbi_mean=("wbi", "mean"),
            behavior_effect_mean=("behavior_effect", "mean"),
        )
    )
    averaged = (
        frame.groupby(
            [
                "model",
                "model_suffix",
                "cohort",
                "target",
                "family",
                "label_budget_requested",
                "method",
            ],
            as_index=False,
        )[list(METHOD_METRICS)]
        .mean()
    )
    rows = []
    for budget in LABEL_BUDGETS:
        budget_frame = averaged[averaged.label_budget_requested.eq(budget)]
        reference = budget_frame[budget_frame.method.eq("sae_common64")][
            ["model", "model_suffix", "cohort", "target", "family", *METHOD_METRICS]
        ].copy()
        reference = reference.rename(
            columns={metric: f"{metric}_reference" for metric in METHOD_METRICS}
        )
        for method in METHODS:
            if method == "sae_common64":
                continue
            current = budget_frame[budget_frame.method.eq(method)].merge(
                reference,
                on=["model", "model_suffix", "cohort", "target", "family"],
                validate="one_to_one",
            )
            for metric in METHOD_METRICS:
                current["delta"] = current[metric] - current[f"{metric}_reference"]
                samples = crossed_bootstrap(
                    current,
                    "delta",
                    ["model", "cohort", "family"],
                    bootstrap,
                    stable_seed("label-budget-hierarchical", budget, method, metric),
                )
                stats = interval_and_p(samples)
                rows.append(
                    {
                        "label_budget": budget,
                        "method": method,
                        "reference": "sae_common64",
                        "metric": metric,
                        "target_units": len(current),
                        "mean_delta": float(current.delta.mean()),
                        **stats,
                        "bootstrap_samples": bootstrap,
                    }
                )
    inference = pd.DataFrame(rows)
    for (budget, metric), indices in inference.groupby(["label_budget", "metric"]).groups.items():
        inference.loc[indices, "q_two_sided"] = bh(
            inference.loc[indices, "p_two_sided"].to_numpy()
        )
    return profile, inference


def normalized_rows(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def matched_abs_cosine(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    similarity = np.abs(normalized_rows(a) @ normalized_rows(b).T)
    row, column = linear_sum_assignment(-similarity)
    matched = similarity[row, column]
    return float(matched.mean()), float(np.median(matched)), float(matched.min())


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(np.asarray(a, dtype=np.float64).T)
    qb, _ = np.linalg.qr(np.asarray(b, dtype=np.float64).T)
    rank = max(1, min(qa.shape[1], qb.shape[1]))
    return float(np.square(qa.T @ qb).sum() / rank)


class ArtifactLoader:
    def __init__(self, base: Path, manifest: pd.DataFrame):
        self.base = base
        self.rows = {
            (row.model_suffix, row.cohort, int(row.seed)): row
            for row in manifest.itertuples(index=False)
        }
        self.decoders: dict[tuple[str, str, int, str], np.ndarray] = {}
        self.scalers: dict[tuple[str, str], object] = {}

    def worker(self, model_suffix: str, cohort: str, seed: int) -> Path:
        row = self.rows[(model_suffix, cohort, seed)]
        return worker_path(self.base, pd.Series(row._asdict()))

    def decoder(self, model_suffix: str, cohort: str, seed: int, method: str) -> np.ndarray:
        key = (model_suffix, cohort, seed, method)
        if key in self.decoders:
            return self.decoders[key]
        worker = self.worker(model_suffix, cohort, seed)
        if method == "sae_common64":
            import torch

            saved = torch.load(worker / "fits" / "sae_common64.pt", map_location="cpu", weights_only=False)
            state = saved["model"]
            decoder = state["W_dec"].numpy().T * state["sigma"].numpy()[None, :]
        elif method == "pca64":
            decoder = joblib.load(worker / "fits" / "pca64.joblib").components_
        elif method == "ica64":
            decoder = joblib.load(worker / "fits" / "ica64.joblib").mixing_.T
        elif method == "semi_nmf64":
            with np.load(worker / "fits" / "semi_nmf64.npz", allow_pickle=False) as saved:
                decoder = saved["decoder"]
        elif method == "random_basis64":
            with np.load(worker / "fits" / "random_basis64.npz", allow_pickle=False) as saved:
                decoder = saved["basis"].T
        elif method == "sae_existing_8d":
            import torch

            row = self.rows[(model_suffix, cohort, seed)]
            saved = torch.load(row.existing_sae_checkpoint, map_location="cpu", weights_only=False)
            state = saved["model"]
            scale_key = (model_suffix, cohort)
            if scale_key not in self.scalers:
                self.scalers[scale_key] = joblib.load(row.head_path, mmap_mode="r")["scaler"]
            scaler = self.scalers[scale_key]
            decoder = state["W_dec"].numpy().T * (
                state["sigma"].numpy() / scaler.scale_
            )[None, :]
        else:
            raise KeyError(method)
        self.decoders[key] = normalized_rows(decoder)
        return self.decoders[key]

    def supervised_direction(
        self, model_suffix: str, cohort: str, seed: int, target: str, method: str
    ) -> np.ndarray:
        path = self.worker(model_suffix, cohort, seed) / "fits" / f"supervised_directions_{target}.npz"
        with np.load(path, allow_pickle=False) as saved:
            return normalized_rows(np.asarray(saved[method], dtype=float)[None, :])[0]


def stability_analyses(
    base: Path,
    manifest: pd.DataFrame,
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    loader = ArtifactLoader(base, manifest)
    dictionary_rows = []
    representation_methods = [*RECONSTRUCTIVE_METHODS, "sae_existing_8d"]
    for (model, model_suffix, cohort), _ in manifest.groupby(
        ["model", "model_suffix", "cohort"], sort=True
    ):
        for method in representation_methods:
            for seed_i, seed_j in itertools.combinations(SEEDS, 2):
                a = loader.decoder(model_suffix, cohort, seed_i, method)
                b = loader.decoder(model_suffix, cohort, seed_j, method)
                if method == "sae_existing_8d":
                    row_i = loader.rows[(model_suffix, cohort, seed_i)]
                    row_j = loader.rows[(model_suffix, cohort, seed_j)]
                    frequency_i = np.load(Path(row_i.existing_sae_cache) / "freq.npy", mmap_mode="r")
                    frequency_j = np.load(Path(row_j.existing_sae_cache) / "freq.npy", mmap_mode="r")
                    keep = min(512, len(frequency_i), len(frequency_j))
                    a = a[np.argsort(-np.asarray(frequency_i))[:keep]]
                    b = b[np.argsort(-np.asarray(frequency_j))[:keep]]
                mean, median, minimum = matched_abs_cosine(a, b)
                dictionary_rows.append(
                    {
                        "model": model,
                        "model_suffix": model_suffix,
                        "cohort": cohort,
                        "method": method,
                        "seed_i": seed_i,
                        "seed_j": seed_j,
                        "components_i": len(a),
                        "components_j": len(b),
                        "matched_abs_cosine_mean": mean,
                        "matched_abs_cosine_median": median,
                        "matched_abs_cosine_min": minimum,
                        "subspace_overlap": subspace_overlap(a, b),
                    }
                )

    functional_rows = []
    common_selected = selected[selected.regime.eq("common64_energy")].copy()
    practical_existing = selected[
        selected.regime.eq("existing_sae_energy") & selected.method.eq("sae_existing_8d")
    ].copy()
    functional_source = pd.concat([common_selected, practical_existing], ignore_index=True)
    key_columns = ["model", "model_suffix", "cohort", "target", "method"]
    for keys, group in functional_source.groupby(key_columns, sort=True):
        model, model_suffix, cohort, target, method = keys
        by_seed = {int(item.seed): item for item in group.itertuples(index=False)}
        if set(by_seed) != set(SEEDS):
            raise RuntimeError(f"Incomplete functional seed group {keys}: {sorted(by_seed)}")
        for seed_i, seed_j in itertools.combinations(SEEDS, 2):
            if method in {"sparse_probe", "supervised_cav"}:
                a = loader.supervised_direction(model_suffix, cohort, seed_i, target, method)
                b = loader.supervised_direction(model_suffix, cohort, seed_j, target, method)
                cosine = float(abs(a @ b))
                overlap = cosine**2
                components = 1
                minimum = cosine
            else:
                indices_i = [int(value) for value in str(by_seed[seed_i].selected_components).split("|")]
                indices_j = [int(value) for value in str(by_seed[seed_j].selected_components).split("|")]
                a = loader.decoder(model_suffix, cohort, seed_i, method)[indices_i]
                b = loader.decoder(model_suffix, cohort, seed_j, method)[indices_j]
                cosine, _, minimum = matched_abs_cosine(a, b)
                overlap = subspace_overlap(a, b)
                components = len(indices_i)
            functional_rows.append(
                {
                    "model": model,
                    "model_suffix": model_suffix,
                    "cohort": cohort,
                    "target": target,
                    "method": method,
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "selected_components": components,
                    "functional_abs_cosine_mean": cosine,
                    "functional_abs_cosine_min": minimum,
                    "functional_subspace_overlap": overlap,
                }
            )
    return pd.DataFrame(dictionary_rows), pd.DataFrame(functional_rows)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.base / "manifest.csv")
    complete_paths = []
    files: dict[str, list[Path]] = {
        "methods": [],
        "contrasts": [],
        "reconstruction": [],
        "selected": [],
        "label_budget": [],
        "rate_distortion": [],
    }
    invalid = []
    for _, row in manifest.iterrows():
        worker = worker_path(args.base, row)
        complete = worker / "complete.json"
        if not complete.exists():
            invalid.append(str(worker))
            continue
        metadata = json.loads(complete.read_text())
        if metadata.get("status") != "complete" or metadata.get("data_files_modified") is not False:
            invalid.append(str(worker))
            continue
        complete_paths.append(complete)
        files["methods"].append(worker / "method_seed_cells.csv")
        files["contrasts"].append(worker / "paired_method_contrasts.csv")
        files["reconstruction"].append(worker / "reconstruction_metrics.csv")
        files["selected"].append(worker / "selected_directions.csv")
        files["label_budget"].append(worker / "label_budget_seed_cells.csv")
        files["rate_distortion"].append(worker / "rate_distortion.csv")
    preflight = {
        "manifest_tasks": len(manifest),
        "complete_workers": len(complete_paths),
        "invalid_or_missing_workers": invalid,
    }
    if invalid or len(complete_paths) != len(manifest):
        raise RuntimeError(f"Worker preflight failed: {preflight}")
    if args.preflight_only:
        print(preflight)
        return

    methods = pd.concat([pd.read_csv(path) for path in files["methods"]], ignore_index=True)
    contrasts = pd.concat([pd.read_csv(path) for path in files["contrasts"]], ignore_index=True)
    reconstruction = pd.concat(
        [pd.read_csv(path) for path in files["reconstruction"]], ignore_index=True
    )
    selected = pd.concat([pd.read_csv(path) for path in files["selected"]], ignore_index=True)
    label_budget = pd.concat(
        [pd.read_csv(path) for path in files["label_budget"]], ignore_index=True
    )
    rate_distortion = pd.concat(
        [pd.read_csv(path) for path in files["rate_distortion"]], ignore_index=True
    )
    expected_target_seed_cells = int(manifest.targets.sum())
    expected = {
        "method_rows": expected_target_seed_cells * (len(METHODS) * 2 + 1),
        "contrast_rows": expected_target_seed_cells * ((len(METHODS) - 1) + len(METHODS)),
        "reconstruction_rows": len(manifest) * (len(RECONSTRUCTIVE_METHODS) + 1),
        "selected_rows": expected_target_seed_cells * (len(METHODS) * 2 + 1),
        "label_budget_rows": expected_target_seed_cells * len(LABEL_BUDGETS) * len(METHODS),
        "rate_distortion_rows": len(manifest)
        * (3 + (len(RECONSTRUCTIVE_METHODS) - 1) * len(RATE_DISTORTION_K)),
    }
    observed = {
        "method_rows": len(methods),
        "contrast_rows": len(contrasts),
        "reconstruction_rows": len(reconstruction),
        "selected_rows": len(selected),
        "label_budget_rows": len(label_budget),
        "rate_distortion_rows": len(rate_distortion),
    }
    if observed != expected:
        raise RuntimeError(f"Combined row audit failed: observed={observed}, expected={expected}")

    for regime, indices in contrasts.groupby("regime").groups.items():
        for metric in METHOD_METRICS:
            contrasts.loc[indices, f"delta_{metric}_q_two_sided"] = bh(
                contrasts.loc[indices, f"delta_{metric}_p_two_sided"].to_numpy()
            )

    method_profile = (
        methods.groupby(["regime", "method", "reference"], as_index=False)
        .agg(
            seed_cells=("target", "size"),
            target_units=("target", lambda x: len(x) // 3),
            ste_mean=("ste", "mean"),
            otd_mean=("otd_mean", "mean"),
            selectivity_margin_mean=("selectivity_margin", "mean"),
            wbi_mean=("wbi", "mean"),
            behavior_effect_mean=("behavior_effect", "mean"),
            l2_match_max_error=("matched_l2_max_abs_error", "max"),
            existing_sae_parity_max_abs=("existing_sae_logit_parity_max_abs", "max"),
        )
    )
    hierarchical, leave, envelopes = hierarchical_inference(contrasts, args.bootstrap)
    reconstruction_profile, reconstruction_hierarchical = reconstruction_inference(
        reconstruction, args.bootstrap
    )
    dictionary_stability, functional_stability = stability_analyses(
        args.base, manifest, selected
    )
    label_budget_profile, label_budget_hierarchical = label_budget_inference(
        label_budget, args.bootstrap
    )
    matched_points, matched_profile = reconstruction_matched_points(rate_distortion)

    output_frames = {
        "method_seed_cells.csv": methods,
        "paired_method_contrasts.csv": contrasts,
        "method_profile.csv": method_profile,
        "hierarchical_method_inference.csv": hierarchical,
        "hierarchical_leave_one_out.csv": leave,
        "hierarchical_leave_one_out_envelopes.csv": envelopes,
        "reconstruction_seed_cells.csv": reconstruction,
        "reconstruction_profile.csv": reconstruction_profile,
        "reconstruction_hierarchical_inference.csv": reconstruction_hierarchical,
        "selected_directions.csv": selected,
        "dictionary_seed_pair_stability.csv": dictionary_stability,
        "functional_seed_pair_stability.csv": functional_stability,
        "label_budget_seed_cells.csv": label_budget,
        "label_budget_profile.csv": label_budget_profile,
        "label_budget_hierarchical_inference.csv": label_budget_hierarchical,
        "rate_distortion_seed_cells.csv": rate_distortion,
        "reconstruction_matched_operating_points.csv": matched_points,
        "reconstruction_matched_profile.csv": matched_profile,
    }
    summary_root = args.base / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)
    for filename, frame in output_frames.items():
        frame.to_csv(summary_root / filename, index=False)
    metadata = {
        "schema_version": 1,
        **preflight,
        **observed,
        "expected_rows": expected,
        "hierarchical_inference_rows": len(hierarchical),
        "hierarchical_leave_one_out_rows": len(leave),
        "reconstruction_inference_rows": len(reconstruction_hierarchical),
        "dictionary_stability_rows": len(dictionary_stability),
        "functional_stability_rows": len(functional_stability),
        "label_budget_inference_rows": len(label_budget_hierarchical),
        "reconstruction_matched_rows": len(matched_points),
        "crossed_bootstrap_samples": args.bootstrap,
        "crossed_factors": ["model", "cohort", "family"],
        "all_complete": True,
        "data_files_modified": False,
    }
    write_json(summary_root / "metadata.json", metadata)
    print(method_profile.to_string(index=False))
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
