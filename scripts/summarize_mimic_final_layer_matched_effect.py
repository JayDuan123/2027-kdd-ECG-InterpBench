#!/usr/bin/env python
"""Audit and summarize the MIMIC final-layer Dense-versus-SAE replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.mimic_matched_effect import MODEL_SPECS, PROTOCOL  # noqa: E402
from benchmark_v1.sparse_accessibility import bh_adjust  # noqa: E402
from scripts.run_accessibility_calibration_worker import atomic_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-models", type=int, default=6)
    parser.add_argument("--expected-workers", type=int, default=18)
    parser.add_argument("--expected-readouts", type=int, default=6)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument(
        "--readout-protocol",
        default="",
        help="Protocol stored in reusable readout summaries; defaults to --protocol.",
    )
    parser.add_argument(
        "--quality-source",
        choices=("training_metrics", "worker"),
        default="training_metrics",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/training_manifest.csv",
    )
    parser.add_argument(
        "--readouts-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/readouts",
    )
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/workers",
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/bootstrap",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/mimic_final_layer_matched_effect_v1/summary",
    )
    return parser.parse_args()


def completed(root: Path, protocol: str = PROTOCOL) -> list[dict[str, object]]:
    result = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol") == protocol:
            result.append(payload)
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append("NA" if not np.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_wbi_figure(
    profile: pd.DataFrame,
    output: Path,
    scope: str,
    stem: str,
    title: str,
    nonevaluable_text: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = [row[0] for row in MODEL_SPECS]
    primary = profile[profile.profile_scope == scope]
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), sharey=False)
    for axis, model in zip(axes.ravel(), models):
        values = primary[primary.model == model]
        heights = []
        counts = []
        for method in ("dense", "sae"):
            selected = values[values.method == method]
            heights.append(float(selected.wbi_cross.iloc[0]) if len(selected) else np.nan)
            counts.append(int(selected.concepts_eligible.iloc[0]) if len(selected) else 0)
        if np.all(np.isfinite(heights)) and min(counts) > 0:
            bars = axis.bar([0, 1], heights, color=["#4C78A8", "#E45756"], width=0.68)
            for bar, count, height in zip(bars, counts, heights):
                axis.text(bar.get_x() + bar.get_width() / 2, height, f"n={count}", ha="center", va="bottom", fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                nonevaluable_text,
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.set_xticks([0, 1], ["Dense", "SAE"])
        axis.set_title(model)
        axis.set_ylabel("Cross-family WBI")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_delta_figure(
    paired: pd.DataFrame, output: Path, stem: str, title: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    models = [row[0] for row in MODEL_SPECS]
    metrics = [("off_cross_rms", "Off-target RMS"), ("wbi_cross", "WBI")]
    matrix = np.full((len(models), len(metrics)), np.nan)
    counts = np.zeros_like(matrix, dtype=int)
    for i, model in enumerate(models):
        for j, (metric, _label) in enumerate(metrics):
            selected = paired[(paired.model == model) & (paired.metric == metric)]
            if len(selected):
                matrix[i, j] = float(selected.observed_delta.iloc[0])
                counts[i, j] = int(selected.concepts_paired.iloc[0])
    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(finite.max()) if len(finite) else 1.0, 1e-6)
    fig, axis = plt.subplots(figsize=(6.5, 5.0))
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        aspect="auto",
    )
    for i in range(len(models)):
        for j in range(len(metrics)):
            label = "NA" if not np.isfinite(matrix[i, j]) else f"{matrix[i, j]:+.3f}\nn={counts[i, j]}"
            axis.text(j, i, label, ha="center", va="center", fontsize=8)
    axis.set_xticks(range(len(metrics)), [label for _metric, label in metrics])
    axis.set_yticks(range(len(models)), models)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    readouts = completed(args.readouts_root, args.readout_protocol or args.protocol)
    workers = completed(args.workers_root, args.protocol)
    bootstraps = completed(args.bootstrap_root, args.protocol)
    if len(readouts) != args.expected_readouts:
        raise RuntimeError(f"expected {args.expected_readouts} readouts, found {len(readouts)}")
    if len(workers) != args.expected_workers:
        raise RuntimeError(f"expected {args.expected_workers} workers, found {len(workers)}")
    if len(bootstraps) != args.expected_models:
        raise RuntimeError(f"expected {args.expected_models} bootstraps, found {len(bootstraps)}")

    profiles = pd.concat([pd.read_csv(Path(row["method_profile"])) for row in bootstraps], ignore_index=True)
    paired = pd.concat([pd.read_csv(Path(row["paired_table"])) for row in bootstraps], ignore_index=True)
    paired["q_value_bh"] = np.nan
    for (_scope, _metric), indices in paired.groupby(["inference_scope", "metric"]).groups.items():
        pvalues = paired.loc[indices, "p_value_two_sided"]
        finite = pvalues.notna()
        if finite.any():
            paired.loc[pvalues.index[finite], "q_value_bh"] = bh_adjust(pvalues.loc[finite].to_numpy())
    quality_rows = []
    if args.quality_source == "worker":
        for payload in workers:
            gate = payload["sae_quality_gate"]
            quality_rows.append(
                {
                    "model": payload["model"],
                    "seed": int(payload["sae_seed"]),
                    "validation_reconstruction_r2": payload["sae_validation_quality"]["reconstruction_r2"],
                    "validation_dead_fraction": payload["sae_validation_quality"]["dead_fraction"],
                    "validation_live_features": int(gate["validation_live_features"]),
                    "quality_gate_mode": gate["mode"],
                    "quality_pass": bool(payload["sae_quality_pass"]),
                }
            )
    else:
        manifest = pd.read_csv(args.manifest)
        for row in manifest.itertuples(index=False):
            payload = json.loads(Path(row.metrics).read_text())
            quality_rows.append(
                {
                    "model": row.model,
                    "seed": int(row.seed),
                    "validation_reconstruction_r2": payload["validation"]["reconstruction_r2"],
                    "validation_dead_fraction": payload["validation"]["dead_fraction"],
                    "validation_live_features": int(
                        round(int(payload["N"]) * (1.0 - float(payload["validation"]["dead_fraction"])))
                    ),
                    "quality_gate_mode": "training_metrics",
                    "quality_pass": payload["quality_gate"]["pass"],
                }
            )
    quality = pd.DataFrame(quality_rows)

    args.output_root.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.output_root / "model_method_profile.csv", index=False)
    paired.to_csv(args.output_root / "paired_model_bootstrap_fdr.csv", index=False)
    quality.to_csv(args.output_root / "sae_quality_by_seed.csv", index=False)
    make_wbi_figure(
        profiles,
        args.output_root,
        "quality_and_all_seeds_primary",
        "mimic_final_layer_matched_effect_wbi_k5",
        "MIMIC-IV strict quality-gated intervention",
        "Not evaluable\nunder primary gate",
    )
    make_wbi_figure(
        profiles,
        args.output_root,
        "all_seeds_effect_gate",
        "mimic_final_layer_matched_effect_wbi_k5_sensitivity",
        "MIMIC-IV all-seeds effect-gate sensitivity",
        "No all-seed\neligible concept",
    )

    primary = paired[
        (paired.inference_scope == "quality_and_all_seeds_primary")
        & paired.metric.isin(["off_cross_rms", "wbi_cross"])
    ].copy()
    sensitivity = paired[
        (paired.inference_scope == "all_seeds_effect_gate_sensitivity")
        & paired.metric.isin(["off_cross_rms", "wbi_cross"])
    ].copy()
    make_delta_figure(
        primary,
        args.output_root,
        "mimic_final_layer_matched_effect_deltas",
        "Strict primary: SAE - Dense (negative favors SAE)",
    )
    make_delta_figure(
        sensitivity,
        args.output_root,
        "mimic_final_layer_matched_effect_deltas_sensitivity",
        "Effect-gate sensitivity: SAE - Dense (negative favors SAE)",
    )
    primary["sae_favorable_significant"] = (
        (primary.observed_delta < 0) & (primary.q_value_bh < 0.05)
    )
    evaluable = primary[primary.observed_delta.notna()]
    sensitivity["sae_favorable_significant"] = (
        (sensitivity.observed_delta < 0) & (sensitivity.q_value_bh < 0.05)
    )
    sensitivity_evaluable = sensitivity[sensitivity.observed_delta.notna()]
    primary_profile = profiles[
        (profiles.profile_scope == "quality_and_all_seeds_primary")
        & profiles.method.isin(["dense", "sae"])
    ].copy()
    display = primary_profile[
        ["model", "method", "sae_all_seed_quality_pass", "concepts_eligible", "target_delta", "off_cross_rms", "wbi_cross"]
    ].round(4)
    all_seed_quality_models = sorted(
        model for model, part in quality.groupby("model") if bool(part.quality_pass.all())
    )
    candidate_pools = sorted({str(payload.get("candidate_pool", "all")) for payload in workers})
    quality_gate_descriptions = sorted(
        {
            str(payload.get("sae_quality_gate", {}).get("description", "stored SAE training quality gate"))
            for payload in workers
        }
    )
    favorable_models = 0
    for _model, part in evaluable.groupby("model"):
        favorable_models += int(len(part) == 2 and bool(part.sae_favorable_significant.all()))
    audit = {
        "status": "complete",
        "protocol": args.protocol,
        "models": args.expected_models,
        "readout_cells": len(readouts),
        "training_and_worker_cells": len(workers),
        "bootstrap_cells": len(bootstraps),
        "concepts": 7,
        "methods": ["dense", "sae"],
        "candidate_count_per_method": 768,
        "candidate_pools": candidate_pools,
        "k": 5,
        "sae_seeds": 3,
        "sae_all_seed_quality_models": all_seed_quality_models,
        "primary_evaluable_models": sorted(evaluable.model.unique().tolist()),
        "primary_nonevaluable_models": sorted(set(row[0] for row in MODEL_SPECS) - set(evaluable.model.unique())),
        "primary_evaluable_metric_tests": len(evaluable),
        "sae_favorable_significant_primary_tests": int(evaluable.sae_favorable_significant.sum()),
        "models_sae_favorable_significant_on_both_primary_metrics": favorable_models,
        "sensitivity_evaluable_models": sorted(sensitivity_evaluable.model.unique().tolist()),
        "sensitivity_evaluable_metric_tests": len(sensitivity_evaluable),
        "sensitivity_sae_favorable_significant_tests": int(
            sensitivity_evaluable.sae_favorable_significant.sum()
        ),
        "bootstrap_draws": int(bootstraps[0]["bootstrap_draws"]),
        "fdr_family": "six model tests separately for each endpoint",
        "primary_gate": (
            "all three SAE quality gates and all three matched-effect feasibility gates; "
            + "; ".join(quality_gate_descriptions)
        ),
        "claim_boundary": "frozen linear-readout response on a seven-concept MIMIC waveform panel, not waveform or biological causality",
    }
    atomic_json(args.output_root / "audit.json", audit)
    report = [
        "# MIMIC-IV final-layer validation-matched intervention",
        "",
        "This external replication compares Dense and MIMIC-trained SAE coordinates at matched candidate count (768) and matched validation target-readout effect. Feature selection, centroids, SAE fitting, and readout fitting use training data only; intervention doses are fixed on validation and evaluated on the patient-disjoint test split. SAE candidate pool(s): " + ", ".join(candidate_pools) + ".",
        "",
        "## Primary profile",
        "",
        markdown_table(display),
        "",
        "## SAE-minus-Dense patient bootstrap",
        "",
        markdown_table(primary[["model", "metric", "concepts_paired", "observed_delta", "ci_low", "ci_high", "p_value_two_sided", "q_value_bh"]].round(4)),
        "",
        "## All-seeds effect-gate sensitivity (not quality-gated)",
        "",
        markdown_table(sensitivity[["model", "metric", "concepts_paired", "observed_delta", "ci_low", "ci_high", "p_value_two_sided", "q_value_bh"]].round(4)),
        "",
        f"Models passing the SAE quality gate in all three seeds: {', '.join(all_seed_quality_models) if all_seed_quality_models else 'none'}.",
        f"SAE is favorable and FDR-significant in {int(evaluable.sae_favorable_significant.sum())}/{len(evaluable)} evaluable primary metric tests and on both endpoints in {favorable_models}/{evaluable.model.nunique()} evaluable models.",
        f"The non-primary all-seeds effect-gate sensitivity has {len(sensitivity_evaluable)} evaluable metric tests; it does not override SAE quality failures.",
        "Ineligible concepts and quality-gate failures remain in the fixed denominator. Results measure frozen linear-readout spillover and do not establish waveform causality or clinical utility.",
    ]
    (args.output_root / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
