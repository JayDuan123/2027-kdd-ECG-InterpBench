#!/usr/bin/env python3
"""Summarize five common SAE scales against a fixed train-fitted PCA-768."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.five_scale_dense_comparison import (  # noqa: E402
    clustered_mean_bootstrap,
    validate_complete_factorial,
)

MODELS = (
    "CARDIAC-FM", "CSFM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"
)
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPANSIONS = (1, 4, 8, 16, 32)
SEEDS = (4311, 4312, 4313)
N_CONCEPTS = 49
BOOTSTRAP_SEED = 20_260_714


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument(
        "--cell-metrics",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/cell_metrics.csv",
    )
    parser.add_argument(
        "--pca-workers",
        type=Path,
        default=ROOT / "results/five_scale_pca_comparison_v1/workers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/five_scale_pca_comparison_v1/summary",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_sae(manifest: pd.DataFrame) -> pd.DataFrame:
    frames = []
    panel: set[str] | None = None
    for row in manifest.itertuples(index=False):
        path = Path(row.concept_metrics)
        if not path.is_absolute():
            path = ROOT / path
        test = pd.read_csv(path).query("split == 'test'").copy()
        if len(test) != N_CONCEPTS:
            raise ValueError(f"expected 49 SAE test rows in {path}")
        current = set(test["concept"].astype(str))
        if panel is None:
            panel = current
        elif current != panel:
            raise ValueError(f"SAE concept panel mismatch in {path}")
        frames.append(test)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != 450 * N_CONCEPTS:
        raise ValueError("incomplete SAE concept grid")
    return result


def load_pca(worker_root: Path) -> tuple[pd.DataFrame, list[dict[str, object]], list[Path]]:
    summaries = sorted(worker_root.glob("group_*/summary.json"))
    if len(summaries) != 30:
        raise ValueError(f"expected 30 PCA summaries, found {len(summaries)}")
    payloads = [json.loads(path.read_text()) for path in summaries]
    if any(item.get("status") != "complete" for item in payloads):
        raise ValueError("at least one PCA worker is incomplete")
    if {int(item["group_index"]) for item in payloads} != set(range(30)):
        raise ValueError("PCA worker indices are not exactly 0..29")
    metric_paths = [Path(str(item["concept_metrics"])) for item in payloads]
    frames = [pd.read_csv(path).query("split == 'test'").copy() for path in metric_paths]
    if any(len(frame) != N_CONCEPTS for frame in frames):
        raise ValueError("each PCA worker must contain 49 test concepts")
    pca = pd.concat(frames, ignore_index=True)
    if len(pca) != 30 * N_CONCEPTS:
        raise ValueError("incomplete PCA concept grid")
    return pca, payloads, metric_paths


def derive_tables(
    manifest: pd.DataFrame, sae: pd.DataFrame, pca: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sae_keys = [
        "model", "layer", "relative_depth", "expansion_E", "concept", "family"
    ]
    paired = (
        sae.groupby(sae_keys, as_index=False)
        .agg(
            sae_mean_abs_r=("abs_eval_correlation", "mean"),
            sae_sd_abs_r=("abs_eval_correlation", "std"),
            sae_coverage_probability=(
                "abs_eval_correlation", lambda values: float(np.mean(values >= 0.20))
            ),
            sae_seed_count=("seed", "nunique"),
        )
    )
    pca_keep = pca[
        [
            "model", "layer", "relative_depth", "concept", "family",
            "selected_feature", "abs_eval_correlation", "covered_020",
        ]
    ].rename(
        columns={
            "selected_feature": "pca_selected_feature",
            "abs_eval_correlation": "pca_abs_r",
            "covered_020": "pca_covered_020",
        }
    )
    paired = paired.merge(
        pca_keep,
        on=["model", "layer", "relative_depth", "concept", "family"],
        validate="many_to_one",
    )
    if len(paired) != len(MODELS) * len(DEPTHS) * len(EXPANSIONS) * N_CONCEPTS:
        raise ValueError("unexpected paired concept row count")
    if not (paired["sae_seed_count"] == len(SEEDS)).all():
        raise ValueError("each SAE point must average exactly three seeds")
    paired["delta_abs_r"] = paired["sae_mean_abs_r"] - paired["pca_abs_r"]
    paired["delta_coverage"] = (
        paired["sae_coverage_probability"] - paired["pca_covered_020"]
    )
    paired["is_final_layer"] = np.isclose(paired["relative_depth"], 1.0)

    cells = (
        paired.groupby(
            ["model", "layer", "relative_depth", "expansion_E"], as_index=False
        )
        .agg(
            sae_mean_abs_r=("sae_mean_abs_r", "mean"),
            sae_coverage=("sae_coverage_probability", "mean"),
            pca_mean_abs_r=("pca_abs_r", "mean"),
            pca_coverage=("pca_covered_020", "mean"),
            concept_count=("concept", "nunique"),
        )
    )
    cells["delta_abs_r"] = cells["sae_mean_abs_r"] - cells["pca_mean_abs_r"]
    cells["delta_coverage"] = cells["sae_coverage"] - cells["pca_coverage"]
    cells["is_final_layer"] = np.isclose(cells["relative_depth"], 1.0)
    if not (cells["concept_count"] == N_CONCEPTS).all():
        raise ValueError("each cell must contain all concepts")
    validate_complete_factorial(
        set(zip(cells["model"], cells["relative_depth"], cells["expansion_E"])),
        (MODELS, DEPTHS, EXPANSIONS),
    )
    validate_complete_factorial(
        set(zip(manifest["model"], manifest["relative_depth"], manifest["expansion_E"], manifest["seed"])),
        (MODELS, DEPTHS, EXPANSIONS, SEEDS),
    )
    return paired, cells


def estimate(values, models, replicates: int, seed: int) -> tuple[float, float, float]:
    result = clustered_mean_bootstrap(
        np.asarray(values), np.asarray(models), replicates=replicates, seed=seed
    )
    return result.mean, result.lower, result.upper


def summarize_scales(cells: pd.DataFrame, replicates: int) -> pd.DataFrame:
    metrics = (
        "sae_mean_abs_r", "pca_mean_abs_r", "delta_abs_r",
        "sae_coverage", "pca_coverage", "delta_coverage",
    )
    output = []
    for expansion in EXPANSIONS:
        subset = cells.loc[cells["expansion_E"] == expansion]
        row: dict[str, object] = {
            "expansion_E": expansion,
            "sae_dictionary_width": expansion * 768,
            "pca_components": 768,
            "model_depth_cells": len(subset),
            "sae_wins_cells": int((subset["delta_abs_r"] > 0).sum()),
        }
        for metric_index, metric in enumerate(metrics):
            seed = (
                BOOTSTRAP_SEED + 10 + metric_index
                if metric in {"pca_mean_abs_r", "pca_coverage"}
                else BOOTSTRAP_SEED + expansion * 100 + metric_index
            )
            mean, lower, upper = estimate(
                subset[metric], subset["model"], replicates, seed
            )
            row[metric] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        output.append(row)
    return pd.DataFrame(output)


def e1_e8_contrast(cells: pd.DataFrame, replicates: int) -> dict[str, object]:
    wide = cells.pivot(
        index=["model", "layer", "relative_depth"],
        columns="expansion_E",
        values=["sae_mean_abs_r", "sae_coverage"],
    )
    values = (wide[("sae_mean_abs_r", 1)] - wide[("sae_mean_abs_r", 8)]).to_numpy()
    coverage = (wide[("sae_coverage", 1)] - wide[("sae_coverage", 8)]).to_numpy()
    models = wide.index.get_level_values("model").to_numpy()
    r = clustered_mean_bootstrap(values, models, replicates=replicates, seed=BOOTSTRAP_SEED + 1001)
    c = clustered_mean_bootstrap(coverage, models, replicates=replicates, seed=BOOTSTRAP_SEED + 1002)
    return {
        "contrast": "E1_minus_E8",
        "mean_abs_r_difference": r.mean,
        "mean_abs_r_ci_lower": r.lower,
        "mean_abs_r_ci_upper": r.upper,
        "coverage_difference": c.mean,
        "coverage_ci_lower": c.lower,
        "coverage_ci_upper": c.upper,
        "cells_E1_above_E8": int(np.sum(values > 0)),
        "cells_total": len(values),
    }


def plot_curves(summary: pd.DataFrame, output: Path) -> None:
    x = summary["expansion_E"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    panels = (
        ("sae_mean_abs_r", "pca_mean_abs_r", "Mean held-out |r|"),
        ("sae_coverage", "pca_coverage", "Concept coverage (|r| >= 0.20)"),
    )
    for axis, (sae_metric, pca_metric, ylabel) in zip(axes, panels):
        sae = summary[sae_metric].to_numpy()
        pca = summary[pca_metric].to_numpy()
        axis.fill_between(
            x, summary[f"{sae_metric}_ci_lower"], summary[f"{sae_metric}_ci_upper"],
            color="#D55E00", alpha=0.18,
        )
        axis.plot(x, sae, color="#D55E00", marker="o", linewidth=2, label="SAE")
        axis.fill_between(
            x, summary[f"{pca_metric}_ci_lower"], summary[f"{pca_metric}_ci_upper"],
            color="#0072B2", alpha=0.12,
        )
        axis.plot(x, pca, color="#0072B2", linestyle="--", linewidth=2, label="PCA-768 (fixed)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [f"E{int(value)}" for value in x])
        axis.set_xlabel("SAE expansion scale")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Five-scale SAE accessibility versus train-fitted PCA-768")
    fig.tight_layout()
    fig.savefig(output / "five_scale_pca_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "five_scale_pca_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_final_heatmap(final: pd.DataFrame, output: Path) -> None:
    matrix = final.pivot(index="model", columns="expansion_E", values="delta_abs_r").reindex(
        index=MODELS, columns=EXPANSIONS
    )
    values = matrix.to_numpy()
    limit = max(0.01, float(np.max(np.abs(values))))
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(EXPANSIONS)), [f"E{value}" for value in EXPANSIONS])
    axis.set_yticks(np.arange(len(MODELS)), MODELS)
    axis.set_xlabel("SAE expansion scale")
    axis.set_title("Final-layer SAE minus PCA-768 mean held-out |r|")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if abs(values[row, column]) > 0.55 * limit else "black"
            axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("SAE - PCA-768")
    fig.tight_layout()
    fig.savefig(output / "final_layer_sae_pca_delta_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "final_layer_sae_pca_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != 450:
        raise ValueError("expected complete 450-cell SAE manifest")
    sae = load_sae(manifest)
    pca, worker_summaries, pca_metric_paths = load_pca(args.pca_workers)
    paired, cells = derive_tables(manifest, sae, pca)

    released = pd.read_csv(args.cell_metrics)
    reconstructed = (
        sae.assign(covered=(sae["abs_eval_correlation"] >= 0.20).astype(int))
        .groupby(["model", "layer", "relative_depth", "expansion_E", "seed"], as_index=False)
        .agg(alignment=("abs_eval_correlation", "mean"), coverage=("covered", "mean"))
    )
    check = reconstructed.merge(
        released,
        on=["model", "layer", "relative_depth", "expansion_E", "seed"],
        validate="one_to_one",
    )
    alignment_error = float(np.max(np.abs(check["alignment"] - check["test_semantic_alignment"])))
    coverage_error = float(np.max(np.abs(check["coverage"] - check["test_concept_coverage_020"])))
    if alignment_error > 1e-7 or coverage_error > 1e-12:
        raise ValueError("SAE concept reconstruction does not match released metrics")

    summary = summarize_scales(cells, args.bootstrap_replicates)
    contrast = e1_e8_contrast(cells, args.bootstrap_replicates)
    final = cells.loc[cells["is_final_layer"]].copy()
    if len(cells) != 150 or len(final) != 30:
        raise ValueError("expected 150 scale cells and 30 final-layer rows")

    pca.to_csv(args.output_dir / "all_pca_concept_scores.csv", index=False)
    paired.to_csv(args.output_dir / "paired_concept_scores.csv", index=False)
    cells.to_csv(args.output_dir / "model_depth_scale_summary.csv", index=False)
    summary.to_csv(args.output_dir / "scale_summary.csv", index=False)
    final.to_csv(args.output_dir / "final_layer_summary.csv", index=False)
    (args.output_dir / "e1_vs_e8_contrast.json").write_text(json.dumps(contrast, indent=2) + "\n")
    plot_curves(summary, args.output_dir)
    plot_final_heatmap(final, args.output_dir)

    pca_numerical = {
        "max_normalization_mean_error": max(
            item["normalization_audit"]["max_abs_mean_error"] for item in worker_summaries
        ),
        "max_normalization_scale_error": max(
            item["normalization_audit"]["max_abs_scale_error"] for item in worker_summaries
        ),
        "max_orthonormal_error": max(item["orthonormal_max_abs_error"] for item in worker_summaries),
        "min_full_rank_reconstruction_r2": min(
            item["test_full_rank_reconstruction_r2"] for item in worker_summaries
        ),
    }
    worker_hash_lines = []
    for item, metric_path in zip(worker_summaries, pca_metric_paths):
        model_path = Path(str(item["pca_model"]))
        worker_hash_lines.append(f"{metric_path}:{sha256(metric_path)}:{model_path}:{sha256(model_path)}")
    audit = {
        "status": "complete",
        "protocol": "five_scale_pca_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "relative_depths": list(DEPTHS),
        "expansions": list(EXPANSIONS),
        "sae_seeds": list(SEEDS),
        "concepts": N_CONCEPTS,
        "pca": {
            "components": 768,
            "fit": "unsupervised on complete train split after SAE-matched normalization",
            "selection": "best PC per concept on fixed 4096-record semantic train subset",
            "evaluation": "frozen on patient-disjoint test",
        },
        "bootstrap": {
            "unit": "model cluster retaining all five depths",
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
        },
        "counts": {
            "pca_workers": len(worker_summaries),
            "pca_test_concept_rows": len(pca),
            "sae_seed_concept_rows": len(sae),
            "paired_concept_rows": len(paired),
            "model_depth_scale_rows": len(cells),
            "final_layer_rows": len(final),
        },
        "pca_numerical_audit": pca_numerical,
        "sae_reproduction": {
            "max_abs_alignment_error": alignment_error,
            "max_abs_coverage_error": coverage_error,
        },
        "e1_vs_e8": contrast,
        "sources": {
            "manifest": {"path": str(args.manifest), "sha256": sha256(args.manifest)},
            "cell_metrics": {"path": str(args.cell_metrics), "sha256": sha256(args.cell_metrics)},
            "pca_worker_artifact_manifest_sha256": hashlib.sha256(
                "\n".join(sorted(worker_hash_lines)).encode()
            ).hexdigest(),
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    lines = [
        "| E | SAE mean |r| | PCA-768 mean |r| | SAE-PCA (95% CI) | SAE wins | SAE coverage | PCA coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.expansion_E} | {row.sae_mean_abs_r:.4f} | {row.pca_mean_abs_r:.4f} | "
            f"{row.delta_abs_r:+.4f} [{row.delta_abs_r_ci_lower:+.4f}, {row.delta_abs_r_ci_upper:+.4f}] | "
            f"{row.sae_wins_cells}/30 | {row.sae_coverage:.3f} | {row.pca_coverage:.3f} |"
        )
    best = summary.loc[summary["delta_abs_r"].idxmax()]
    report = f"""# Five-scale SAE versus PCA-768

PCA-768 is fit independently for every model-depth cell using only the complete training split after the exact SAE normalization. For each of 49 waveform concepts, the best PC and the best SAE feature are selected on the same fixed 4,096-record semantic training subset and evaluated without reselection on the patient-disjoint test split.

## Results

{chr(10).join(lines)}

The strongest SAE scale relative to PCA is E{int(best.expansion_E)}, with mean paired difference **{best.delta_abs_r:+.4f}** (95% model-cluster CI [{best.delta_abs_r_ci_lower:+.4f}, {best.delta_abs_r_ci_upper:+.4f}]). The E1-minus-E8 SAE difference remains **{contrast['mean_abs_r_difference']:+.4f}** [{contrast['mean_abs_r_ci_lower']:+.4f}, {contrast['mean_abs_r_ci_upper']:+.4f}].

## Interpretation boundary

This is a single-coordinate accessibility comparison. PCA-768 is a complete orthogonal rotation of the same normalized 768-dimensional FM representation; it is fixed across SAE expansion scales. The result does not compare sparse reconstruction budgets, intervention selectivity, or off-target effects.
"""
    (args.output_dir / "report.md").write_text(report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(pca_numerical, indent=2), flush=True)


if __name__ == "__main__":
    main()
