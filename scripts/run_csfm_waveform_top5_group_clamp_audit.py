#!/usr/bin/env python
"""CSFM waveform top-5 SAE feature group clamp audit.

This tests the user's proposed intervention: treat the top-5 SAE atoms for a
waveform measurement concept as one steering group. Continuous concepts are
binarized at the train median, as in the high-vs-low AUROC audit.

The intervention is code-space and observational-readout based:
  z_test[:, top5] <- train centroid of those features
Then full-SAE readouts for all waveform concepts are re-evaluated. This is a
cheap sanity check before any expensive frozen-model forward steering.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_csfm_top_atom_probe_audit import directionless_auc, encode_sae, load_sae, markdown_table, single_feature_train_scores  # noqa: E402
from scripts.run_csfm_waveform_concept_auroc_audit import WAVEFORM_RE, median_binarize  # noqa: E402
from scripts.run_sae_monosemanticity_taxonomy import collect_runs, load_concept_sets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="CSFM")
    parser.add_argument("--feature-dir", type=Path, default=REPO_ROOT / "results/probe_features/csfm_cu118_commons")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/top_atom_probe",
    )
    parser.add_argument("--concept-regex", default=WAVEFORM_RE)
    parser.add_argument("--n-selected", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=4311)
    parser.add_argument("--max-runs-per-model", type=int, default=1)
    return parser.parse_args()


def fit_binary_readout(z_train: np.ndarray, y_train: np.ndarray, alpha: float) -> Ridge:
    return Ridge(alpha=alpha, solver="lsqr").fit(z_train, y_train.astype(float))


def auroc_or_nan(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y.astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), score))


def wbi(target_effect: float, off_damage: float, eps: float = 1e-6) -> float:
    return float(off_damage / (target_effect + eps))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    tax_args = argparse.Namespace(
        combined_results=REPO_ROOT / "results/sae_extension/six_model_sae_audit/l0clamp_reclassified/sae_l0clamp_reclassified_cells.csv",
        selected_transformer=REPO_ROOT
        / "results/sae_extension/six_model_sae_audit/phase0_selected_transformer_operating_points_l0clamp.csv",
        seed_fill_root=REPO_ROOT / "results/sae_extension/six_model_sae_audit/monosem_seed_fill",
        max_runs_per_model=args.max_runs_per_model,
    )
    runs = [run for run in collect_runs(tax_args) if run.model == args.model]
    if not runs:
        raise RuntimeError(f"No SAE runs found for model={args.model}")

    records = pd.read_csv(args.feature_dir / "records.csv")
    train_idx = np.where(records["split"].to_numpy() == "train")[0]
    test_idx = np.where(records["split"].to_numpy() == "test")[0]
    concept_sets = load_concept_sets(
        argparse.Namespace(
            concepts_matrix=REPO_ROOT / "results/manifest/concepts_matrix.csv",
            concept_config=REPO_ROOT / "configs/concepts.csv",
            orthogonal_set=REPO_ROOT / "results/analysis/model_comparison/orthogonal_concepts/locked_orthogonal_set.csv",
            include_metadata_ortho=False,
            metadata=Path("/rhf/allocations/wq8/yd68/data/ptb-xl/1.0.3/ptbxl_database.csv"),
        ),
        records,
    )
    full = concept_sets["FULL"]
    concepts = [c for c in full.columns if pd.Series([c]).str.contains(args.concept_regex, regex=True).iloc[0]]

    rows: list[dict[str, object]] = []
    for run in runs:
        acts = np.load(args.feature_dir / f"layer_{run.layer:02d}_mean.npy", mmap_mode="r")
        x_train_raw = np.asarray(acts[train_idx])
        x_test_raw = np.asarray(acts[test_idx])
        sae = load_sae(run.checkpoint)
        z_train = encode_sae(sae, x_train_raw)
        z_test = encode_sae(sae, x_test_raw)
        centroid = z_train.mean(axis=0)
        n_features = z_train.shape[1]
        print(f"Top5 group clamp {run.model} seed={run.seed} layer={run.layer} N={n_features}", flush=True)

        # Build all waveform high-vs-low labels and full-code readouts once per seed.
        label_train: dict[str, np.ndarray] = {}
        label_test: dict[str, np.ndarray] = {}
        valid_train: dict[str, np.ndarray] = {}
        valid_test: dict[str, np.ndarray] = {}
        readouts: dict[str, Ridge] = {}
        clean_auroc: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        rankings: dict[str, np.ndarray] = {}
        for concept in concepts:
            y_all = full[concept].to_numpy(dtype=float)
            tr_valid = np.isfinite(y_all[train_idx])
            te_valid = np.isfinite(y_all[test_idx])
            if tr_valid.sum() < 20 or te_valid.sum() < 20:
                continue
            y_train_raw = y_all[train_idx][tr_valid]
            y_test_raw = y_all[test_idx][te_valid]
            y_train, y_test, threshold = median_binarize(y_train_raw, y_test_raw)
            if len(np.unique(y_train.astype(int))) < 2 or len(np.unique(y_test.astype(int))) < 2:
                continue
            ztr = z_train[tr_valid]
            zte = z_test[te_valid]
            model = fit_binary_readout(ztr, y_train, args.ridge_alpha)
            clean = auroc_or_nan(y_test, model.predict(zte))
            rank_scores = single_feature_train_scores(ztr, y_train, "binary")
            label_train[concept] = y_train
            label_test[concept] = y_test
            valid_train[concept] = tr_valid
            valid_test[concept] = te_valid
            readouts[concept] = model
            clean_auroc[concept] = clean
            thresholds[concept] = threshold
            rankings[concept] = np.argsort(-rank_scores)

        metric_cache: dict[tuple[int, ...], dict[str, float]] = {}

        def patched_metrics_all(clamp_idx: np.ndarray) -> dict[str, float]:
            key = tuple(sorted(int(i) for i in clamp_idx))
            if key in metric_cache:
                return metric_cache[key]
            zp_all = np.array(z_test, copy=True)
            if key:
                idx = np.asarray(key, dtype=int)
                zp_all[:, idx] = centroid[idx][None, :]
            metrics = {}
            for name in readouts:
                te_valid = valid_test[name]
                metrics[name] = auroc_or_nan(label_test[name], readouts[name].predict(zp_all[te_valid]))
            metric_cache[key] = metrics
            return metrics

        clean_metrics = patched_metrics_all(np.asarray([], dtype=int))

        population = np.arange(n_features)
        for concept in readouts:
            selected = rankings[concept][: min(args.n_selected, n_features)].astype(int)
            target_clean = clean_metrics[concept]
            concept_metrics = patched_metrics_all(selected)
            target_patched = concept_metrics[concept]
            target_effect = target_clean - target_patched
            off_damages = []
            for other in readouts:
                if other == concept:
                    continue
                damage = clean_metrics[other] - concept_metrics[other]
                if np.isfinite(damage):
                    off_damages.append(max(0.0, float(damage)))
            off_damage = float(np.mean(off_damages)) if off_damages else float("nan")
            selectivity = float(target_effect - off_damage)

            random_effects = []
            random_off = []
            random_selectivities = []
            random_wbis = []
            random_sets = []
            for _ in range(args.n_random):
                choice = rng.choice(population, size=len(selected), replace=False)
                random_sets.append(";".join(map(str, choice.astype(int).tolist())))
                random_metrics = patched_metrics_all(choice)
                r_target = target_clean - random_metrics[concept]
                r_off_vals = []
                for other in readouts:
                    if other == concept:
                        continue
                    damage = clean_metrics[other] - random_metrics[other]
                    if np.isfinite(damage):
                        r_off_vals.append(max(0.0, float(damage)))
                r_off = float(np.mean(r_off_vals)) if r_off_vals else float("nan")
                random_effects.append(float(r_target))
                random_off.append(r_off)
                random_selectivities.append(float(r_target - r_off))
                random_wbis.append(wbi(float(r_target), r_off))

            concept_wbi = wbi(float(target_effect), float(off_damage))
            rows.append(
                {
                    "model": run.model,
                    "seed": run.seed,
                    "layer": run.layer,
                    "cell_id": run.cell_id,
                    "concept": concept,
                    "threshold_train_median": thresholds[concept],
                    "n_selected": int(len(selected)),
                    "selected_features": ";".join(map(str, selected.tolist())),
                    "clean_auroc": target_clean,
                    "patched_auroc": target_patched,
                    "target_effect": float(target_effect),
                    "offtarget_damage": off_damage,
                    "selectivity": selectivity,
                    "wbi": concept_wbi,
                    "random_target_effect_mean": float(np.nanmean(random_effects)),
                    "random_offtarget_damage_mean": float(np.nanmean(random_off)),
                    "random_selectivity_mean": float(np.nanmean(random_selectivities)),
                    "random_wbi_mean": float(np.nanmean(random_wbis)),
                    "random_wbi_median": float(np.nanmedian(random_wbis)),
                    "excess_selectivity": float(selectivity - np.nanmean(random_selectivities)),
                    "wbi_improvement_vs_random_mean": float(np.nanmean(random_wbis) - concept_wbi),
                    "n_random": int(args.n_random),
                }
            )

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "csfm_waveform_top5_group_clamp.csv"
    df.to_csv(out_csv, index=False)
    summary = (
        df.groupby("concept", as_index=False)
        .agg(
            clean_auroc_mean=("clean_auroc", "mean"),
            patched_auroc_mean=("patched_auroc", "mean"),
            target_effect_mean=("target_effect", "mean"),
            offtarget_damage_mean=("offtarget_damage", "mean"),
            selectivity_mean=("selectivity", "mean"),
            random_selectivity_mean=("random_selectivity_mean", "mean"),
            excess_selectivity_mean=("excess_selectivity", "mean"),
            wbi_median=("wbi", "median"),
            random_wbi_mean=("random_wbi_mean", "mean"),
            wbi_improvement_mean=("wbi_improvement_vs_random_mean", "mean"),
        )
        .sort_values("target_effect_mean", ascending=False)
    )
    out_summary = args.out_dir / "csfm_waveform_top5_group_clamp_summary.csv"
    summary.to_csv(out_summary, index=False)

    md = [
        "# CSFM Waveform Top5 Group Clamp Audit",
        "",
        "Code-space steering sanity check: each concept's train-selected top5 SAE atoms are clamped to the train centroid together, then full-code waveform readout AUROCs are re-evaluated on test. Same-size random feature groups provide the baseline.",
        "",
        f"- CSV: `{out_csv}`",
        f"- summary: `{out_summary}`",
        "",
        "## Summary",
        "",
        markdown_table(
            summary,
            [
                "concept",
                "clean_auroc_mean",
                "patched_auroc_mean",
                "target_effect_mean",
                "offtarget_damage_mean",
                "excess_selectivity_mean",
                "wbi_median",
                "random_wbi_mean",
                "wbi_improvement_mean",
            ],
            max_rows=40,
        ),
    ]
    out_md = args.out_dir / "csfm_waveform_top5_group_clamp_report.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
