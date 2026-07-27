#!/usr/bin/env python3
"""Compare five common SAE scales with one fixed native dense baseline."""

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
    "CARDIAC-FM",
    "CSFM",
    "ECG-FM",
    "ECG-JEPA",
    "HuBERT-ECG",
    "ST-MEM",
)
DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPANSIONS = (1, 4, 8, 16, 32)
SEEDS = (4311, 4312, 4313)
N_CONCEPTS = 49
THRESHOLD = 0.20
BOOTSTRAP_SEED = 20_260_714


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multiscale-root",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1",
    )
    parser.add_argument(
        "--dense-csv",
        type=Path,
        default=(
            ROOT
            / "results/accessibility_calibration_e8_v2/summary/all_dense_single.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/five_scale_dense_comparison_v1",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_columns(
    values: np.ndarray,
    models: np.ndarray,
    replicates: int,
    seed_offset: int,
) -> tuple[float, float, float]:
    result = clustered_mean_bootstrap(
        values,
        models,
        replicates=replicates,
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    return result.mean, result.lower, result.upper


def load_sae_concepts(manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    expected_concepts: tuple[str, ...] | None = None
    for row in manifest.itertuples(index=False):
        path = Path(row.concept_metrics)
        metrics = pd.read_csv(path)
        test = metrics.loc[metrics["split"] == "test"].copy()
        concepts = tuple(test["concept"].astype(str))
        if len(test) != N_CONCEPTS or len(set(concepts)) != N_CONCEPTS:
            raise ValueError(f"expected {N_CONCEPTS} test concepts in {path}")
        if expected_concepts is None:
            expected_concepts = concepts
        elif set(concepts) != set(expected_concepts):
            raise ValueError(f"concept panel differs in {path}")
        test["concept_metrics_path"] = str(path)
        frames.append(test)
    result = pd.concat(frames, ignore_index=True)
    if len(result) != len(manifest) * N_CONCEPTS:
        raise ValueError("unexpected SAE concept row count")
    return result


def derive_tables(
    manifest: pd.DataFrame,
    sae: pd.DataFrame,
    dense: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sae_keys = [
        "model", "layer", "relative_depth", "expansion_E", "concept", "family"
    ]
    seed_concept = sae[
        sae_keys + ["seed", "abs_eval_correlation", "sign_aligned_eval_correlation"]
    ].copy()
    seed_concept["covered_020"] = (
        seed_concept["abs_eval_correlation"] >= THRESHOLD
    ).astype(int)

    paired = (
        seed_concept.groupby(sae_keys, as_index=False)
        .agg(
            sae_mean_abs_r=("abs_eval_correlation", "mean"),
            sae_sd_abs_r=("abs_eval_correlation", "std"),
            sae_mean_oriented_r=("sign_aligned_eval_correlation", "mean"),
            sae_coverage_probability=("covered_020", "mean"),
            sae_seed_count=("seed", "nunique"),
        )
    )
    dense_keep = dense[
        [
            "model", "layer", "relative_depth", "concept", "family",
            "test_abs_r", "covered_020", "selected_feature",
        ]
    ].rename(
        columns={
            "test_abs_r": "dense_abs_r",
            "covered_020": "dense_covered_020",
            "selected_feature": "dense_selected_feature",
        }
    )
    paired = paired.merge(
        dense_keep,
        on=["model", "layer", "relative_depth", "concept", "family"],
        how="left",
        validate="many_to_one",
    )
    if paired[["dense_abs_r", "dense_covered_020"]].isna().any().any():
        raise ValueError("SAE rows are missing fixed dense matches")
    if not (paired["sae_seed_count"] == len(SEEDS)).all():
        raise ValueError("each SAE concept must contain all three seeds")
    paired["delta_abs_r"] = paired["sae_mean_abs_r"] - paired["dense_abs_r"]
    paired["delta_coverage"] = (
        paired["sae_coverage_probability"] - paired["dense_covered_020"]
    )
    paired["is_final_layer"] = np.isclose(paired["relative_depth"], 1.0)

    cell_keys = ["model", "layer", "relative_depth", "expansion_E"]
    cells = (
        paired.groupby(cell_keys, as_index=False)
        .agg(
            sae_mean_abs_r=("sae_mean_abs_r", "mean"),
            sae_coverage=("sae_coverage_probability", "mean"),
            dense_mean_abs_r=("dense_abs_r", "mean"),
            dense_coverage=("dense_covered_020", "mean"),
            concept_count=("concept", "nunique"),
        )
    )
    cells["delta_abs_r"] = cells["sae_mean_abs_r"] - cells["dense_mean_abs_r"]
    cells["delta_coverage"] = cells["sae_coverage"] - cells["dense_coverage"]
    cells["is_final_layer"] = np.isclose(cells["relative_depth"], 1.0)
    if not (cells["concept_count"] == N_CONCEPTS).all():
        raise ValueError("every model-depth-scale cell must contain 49 concepts")

    expected_manifest = set(
        zip(
            manifest["model"], manifest["relative_depth"],
            manifest["expansion_E"], manifest["seed"],
        )
    )
    validate_complete_factorial(
        expected_manifest, (MODELS, DEPTHS, EXPANSIONS, SEEDS)
    )
    validate_complete_factorial(
        set(zip(cells["model"], cells["relative_depth"], cells["expansion_E"])),
        (MODELS, DEPTHS, EXPANSIONS),
    )
    return seed_concept, paired, cells


def summarize_scales(cells: pd.DataFrame, replicates: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "sae_mean_abs_r", "dense_mean_abs_r", "delta_abs_r",
        "sae_coverage", "dense_coverage", "delta_coverage",
    )
    for expansion in EXPANSIONS:
        subset = cells.loc[cells["expansion_E"] == expansion].copy()
        if len(subset) != len(MODELS) * len(DEPTHS):
            raise ValueError(f"E{expansion} does not contain 30 cells")
        row: dict[str, object] = {
            "expansion_E": expansion,
            "dictionary_width": expansion * 768,
            "model_depth_cells": len(subset),
            "sae_wins_cells": int((subset["delta_abs_r"] > 0).sum()),
        }
        for metric_index, metric in enumerate(metrics):
            seed_offset = (
                10 + metric_index
                if metric in {"dense_mean_abs_r", "dense_coverage"}
                else expansion * 100 + metric_index
            )
            mean, lower, upper = bootstrap_columns(
                subset[metric].to_numpy(),
                subset["model"].to_numpy(),
                replicates,
                seed_offset,
            )
            row[metric] = mean
            row[f"{metric}_ci_lower"] = lower
            row[f"{metric}_ci_upper"] = upper
        rows.append(row)
    return pd.DataFrame(rows)


def scale_contrast(cells: pd.DataFrame, replicates: int) -> dict[str, object]:
    wide_r = cells.pivot(
        index=["model", "layer", "relative_depth"],
        columns="expansion_E",
        values="sae_mean_abs_r",
    )
    wide_c = cells.pivot(
        index=["model", "layer", "relative_depth"],
        columns="expansion_E",
        values="sae_coverage",
    )
    delta_r = (wide_r[1] - wide_r[8]).to_numpy()
    delta_c = (wide_c[1] - wide_c[8]).to_numpy()
    models = wide_r.index.get_level_values("model").to_numpy()
    r = clustered_mean_bootstrap(
        delta_r, models, replicates=replicates, seed=BOOTSTRAP_SEED + 1001
    )
    c = clustered_mean_bootstrap(
        delta_c, models, replicates=replicates, seed=BOOTSTRAP_SEED + 1002
    )
    return {
        "contrast": "E1_minus_E8",
        "mean_abs_r_difference": r.mean,
        "mean_abs_r_ci_lower": r.lower,
        "mean_abs_r_ci_upper": r.upper,
        "coverage_difference": c.mean,
        "coverage_ci_lower": c.lower,
        "coverage_ci_upper": c.upper,
        "cells_E1_above_E8": int(np.sum(delta_r > 0)),
        "cells_total": int(len(delta_r)),
    }


def plot_scale_curves(summary: pd.DataFrame, output: Path) -> None:
    x = summary["expansion_E"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    panels = (
        ("sae_mean_abs_r", "dense_mean_abs_r", "Mean held-out |r|"),
        ("sae_coverage", "dense_coverage", "Concept coverage (|r| >= 0.20)"),
    )
    for axis, (sae_metric, dense_metric, ylabel) in zip(axes, panels):
        sae = summary[sae_metric].to_numpy()
        sae_low = summary[f"{sae_metric}_ci_lower"].to_numpy()
        sae_high = summary[f"{sae_metric}_ci_upper"].to_numpy()
        dense = summary[dense_metric].to_numpy()
        dense_low = summary[f"{dense_metric}_ci_lower"].to_numpy()
        dense_high = summary[f"{dense_metric}_ci_upper"].to_numpy()
        axis.fill_between(x, sae_low, sae_high, color="#D55E00", alpha=0.18)
        axis.plot(x, sae, color="#D55E00", marker="o", linewidth=2, label="SAE")
        axis.fill_between(x, dense_low, dense_high, color="#0072B2", alpha=0.12)
        axis.plot(x, dense, color="#0072B2", linestyle="--", linewidth=2, label="Dense (fixed)")
        axis.set_xscale("log", base=2)
        axis.set_xticks(x, [f"E{int(value)}" for value in x])
        axis.set_xlabel("SAE expansion scale")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Five-scale SAE accessibility versus the fixed dense baseline")
    fig.tight_layout()
    fig.savefig(output / "five_scale_dense_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "five_scale_dense_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_final_heatmap(final: pd.DataFrame, output: Path) -> None:
    matrix = (
        final.pivot(index="model", columns="expansion_E", values="delta_abs_r")
        .reindex(index=MODELS, columns=EXPANSIONS)
    )
    values = matrix.to_numpy()
    limit = max(0.01, float(np.max(np.abs(values))))
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(EXPANSIONS)), [f"E{e}" for e in EXPANSIONS])
    axis.set_yticks(np.arange(len(MODELS)), MODELS)
    axis.set_xlabel("SAE expansion scale")
    axis.set_title("Final-layer SAE minus dense mean held-out |r|")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if abs(values[row, column]) > 0.55 * limit else "black"
            axis.text(column, row, f"{values[row, column]:+.3f}", ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.03)
    colorbar.set_label("SAE - dense")
    fig.tight_layout()
    fig.savefig(output / "final_layer_scale_delta_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "final_layer_scale_delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.multiscale_root / "training_manifest.csv"
    cell_metrics_path = args.multiscale_root / "cell_metrics.csv"
    protocol_path = args.multiscale_root / "protocol.json"
    manifest = pd.read_csv(manifest_path)
    dense = pd.read_csv(args.dense_csv)
    if len(manifest) != 450 or len(dense) != 1470:
        raise ValueError("expected 450 SAE cells and 1470 dense concept rows")
    sae = load_sae_concepts(manifest)
    seed_concept, paired, cells = derive_tables(manifest, sae, dense)

    # Confirm per-concept reconstruction reproduces the released cell summaries.
    reconstructed = (
        seed_concept.groupby(
            ["model", "layer", "relative_depth", "expansion_E", "seed"],
            as_index=False,
        )
        .agg(
            reconstructed_alignment=("abs_eval_correlation", "mean"),
            reconstructed_coverage=("covered_020", "mean"),
        )
    )
    released = pd.read_csv(cell_metrics_path)
    check = reconstructed.merge(
        released,
        on=["model", "layer", "relative_depth", "expansion_E", "seed"],
        validate="one_to_one",
    )
    alignment_error = float(
        np.max(np.abs(check["reconstructed_alignment"] - check["test_semantic_alignment"]))
    )
    coverage_error = float(
        np.max(np.abs(check["reconstructed_coverage"] - check["test_concept_coverage_020"]))
    )
    if alignment_error > 1e-7 or coverage_error > 1e-12:
        raise ValueError("concept rows do not reproduce released cell metrics")

    summary = summarize_scales(cells, args.bootstrap_replicates)
    contrast = scale_contrast(cells, args.bootstrap_replicates)
    final = cells.loc[cells["is_final_layer"]].copy()
    if len(final) != len(MODELS) * len(EXPANSIONS):
        raise ValueError("expected 30 final-layer model-scale rows")

    seed_concept.to_csv(args.output_dir / "sae_seed_concept_scores.csv", index=False)
    paired.to_csv(args.output_dir / "paired_concept_scores.csv", index=False)
    cells.to_csv(args.output_dir / "model_depth_scale_summary.csv", index=False)
    summary.to_csv(args.output_dir / "scale_summary.csv", index=False)
    final.to_csv(args.output_dir / "final_layer_summary.csv", index=False)
    (args.output_dir / "e1_vs_e8_contrast.json").write_text(
        json.dumps(contrast, indent=2) + "\n"
    )
    plot_scale_curves(summary, args.output_dir)
    plot_final_heatmap(final, args.output_dir)

    e1 = summary.loc[summary["expansion_E"] == 1].iloc[0]
    e8 = summary.loc[summary["expansion_E"] == 8].iloc[0]
    e1_wins = final.loc[
        (final["expansion_E"] == 1) & (final["delta_abs_r"] > 0),
        ["model", "delta_abs_r"],
    ].to_dict("records")
    audit = {
        "status": "complete",
        "protocol": "five_scale_dense_comparison_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS),
        "relative_depths": list(DEPTHS),
        "expansions": list(EXPANSIONS),
        "sae_seeds": list(SEEDS),
        "concepts": N_CONCEPTS,
        "threshold": {"operator": ">=", "absolute_r": THRESHOLD},
        "selection": "best feature selected on fixed 4096-record train subset; test frozen",
        "dense_policy": "one fixed native 768-coordinate baseline reused across E",
        "bootstrap": {
            "unit": "model cluster retaining all five depths",
            "replicates": args.bootstrap_replicates,
            "seed": BOOTSTRAP_SEED,
        },
        "counts": {
            "sae_training_cells": len(manifest),
            "sae_seed_concept_rows": len(seed_concept),
            "paired_concept_rows": len(paired),
            "model_depth_scale_rows": len(cells),
            "final_layer_rows": len(final),
            "dense_concept_rows": len(dense),
        },
        "reproduction": {
            "max_abs_alignment_error": alignment_error,
            "max_abs_coverage_error": coverage_error,
        },
        "e1_vs_e8": contrast,
        "e1_final_layer_wins_over_dense": e1_wins,
        "sources": {
            "training_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "cell_metrics": {"path": str(cell_metrics_path), "sha256": sha256(cell_metrics_path)},
            "multiscale_protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "dense_baseline": {"path": str(args.dense_csv), "sha256": sha256(args.dense_csv)},
            "concept_file_count": len(manifest),
            "concept_file_manifest_sha256": hashlib.sha256(
                "\n".join(
                    f"{path}:{sha256(path)}" for path in sorted(
                        (Path(value) for value in manifest["concept_metrics"]), key=str
                    )
                ).encode()
            ).hexdigest(),
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    table_lines = [
        "| E | SAE mean |r| | Dense mean |r| | SAE-dense | SAE wins | SAE coverage | Dense coverage |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        table_lines.append(
            f"| {row.expansion_E} | {row.sae_mean_abs_r:.4f} | "
            f"{row.dense_mean_abs_r:.4f} | {row.delta_abs_r:+.4f} "
            f"[{row.delta_abs_r_ci_lower:+.4f}, {row.delta_abs_r_ci_upper:+.4f}] | "
            f"{row.sae_wins_cells}/30 | {row.sae_coverage:.3f} | {row.dense_coverage:.3f} |"
        )
    report = f"""# Five-scale SAE versus fixed dense baseline

This analysis compares the complete common SAE expansion grid `E={{1,4,8,16,32}}` with one fixed native 768-coordinate dense baseline. It covers six models, five standardized depths, three SAE seeds, and 49 waveform concepts. Feature identity is selected on the same fixed 4,096-record training subset and evaluated without reselection on the patient-disjoint test split.

## Results

{chr(10).join(table_lines)}

The model-cluster paired E1-minus-E8 difference in mean held-out absolute correlation is **{contrast['mean_abs_r_difference']:+.4f}** (95% CI [{contrast['mean_abs_r_ci_lower']:+.4f}, {contrast['mean_abs_r_ci_upper']:+.4f}]); E1 is higher in {contrast['cells_E1_above_E8']}/30 model-depth cells. E8 is therefore not an isolated poor operating point: E4, E8, E16, and E32 form a similar plateau, while E1 is better.

Even at E1, SAE remains below fixed dense by **{e1.delta_abs_r:+.4f}** on average (95% model-cluster CI [{e1.delta_abs_r_ci_lower:+.4f}, {e1.delta_abs_r_ci_upper:+.4f}]) and wins only {int(e1.sae_wins_cells)}/30 cells. At E8 the corresponding difference is **{e8.delta_abs_r:+.4f}**. The only final-layer E1 cells above dense are CARDIAC-FM and ECG-FM.

## Interpretation boundary

The result rejects the explanation that E8 alone caused the dense-baseline deficit. It applies to single-feature waveform-concept accessibility under the fixed-k-over-d BatchTopK arm. It does not establish that dense representations are better for sparse reconstruction, feature stability, causal intervention, or off-target selectivity.
"""
    (args.output_dir / "report.md").write_text(report)
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(contrast, indent=2), flush=True)


if __name__ == "__main__":
    main()
