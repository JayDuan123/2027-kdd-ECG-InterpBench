#!/usr/bin/env python
"""Audit and summarize the final-layer matched-effect intervention release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.sparse_accessibility import bh_adjust  # noqa: E402
try:
    from paper_figure_style import configure_paper_fonts  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts.paper_figure_style import configure_paper_fonts  # noqa: E402
from scripts.run_accessibility_calibration_worker import atomic_json  # noqa: E402


PROTOCOL = "final_layer_matched_effect_v1"
METHOD_ORDER = ("dense", "pca", "sae", "random_rotation")
DISPLAY_METHODS = ("dense", "sae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-models", type=int, default=6)
    parser.add_argument("--expected-workers", type=int, default=18)
    parser.add_argument("--expected-readouts", type=int, default=6)
    parser.add_argument(
        "--readouts-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/readouts",
    )
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/workers",
    )
    parser.add_argument(
        "--bootstrap-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/bootstrap",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_matched_effect_v1/summary",
    )
    return parser.parse_args()


def completed(root: Path) -> list[tuple[Path, dict]]:
    result = []
    for path in sorted(root.glob("*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and payload.get("protocol") == PROTOCOL:
            result.append((path, payload))
    return result


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, float):
                cells.append("nan" if not np.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_figure(profile: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_paper_fonts()
    import matplotlib.pyplot as plt

    style = {
        "dense": ("Dense", "#4C78A8", "o"),
        "sae": ("SAE", "#E45756", "^"),
    }
    primary_methods = DISPLAY_METHODS
    matched = profile[
        (profile.candidate_arm == "matched_768")
        & (profile.k == 5)
        & (profile.profile_scope == "all_seeds_primary")
    ]
    models = [
        model
        for model in matched.model.drop_duplicates()
        if model != "ECG-FM"
    ]
    fig, axes = plt.subplots(1, len(models), figsize=(13.6, 3.8), sharex=True)
    for axis, model in zip(np.atleast_1d(axes), models):
        values = matched[matched.model == model]
        heights = []
        colors = []
        labels_local = []
        counts = []
        for method in primary_methods:
            selected = values[values.method == method]
            label, color, _ = style[method]
            heights.append(float(selected.wbi_cross.iloc[0]) if len(selected) else np.nan)
            colors.append(color)
            labels_local.append(label)
            counts.append(int(selected.concepts_eligible.iloc[0]) if len(selected) else 0)
        bars = axis.bar(np.arange(len(primary_methods)), heights, color=colors, width=0.72)
        for bar, count, height in zip(bars, counts, heights):
            if np.isfinite(height):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(model)
        axis.set_xticks(np.arange(len(primary_methods)), labels_local, rotation=20, ha="right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if not np.any(np.isfinite(heights)):
            axis.text(
                0.5,
                0.5,
                "No concepts passed\nthe all-seed gate",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
        axis.grid(False)
    np.atleast_1d(axes)[0].set_ylabel("Cross-family WBI (lower is better)")
    fig.suptitle("Matched-768, k=5, all-seed eligible concepts")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "final_layer_matched_effect_wbi_k5.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "final_layer_matched_effect_wbi_k5.pdf", bbox_inches="tight")
    plt.close(fig)


def make_delta_figure(paired: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_paper_fonts()
    import matplotlib.pyplot as plt

    selected = paired[
        (paired.candidate_arm == "matched_768")
        & (paired.k == 5)
        & (paired.comparison == "sae_minus_dense")
        & paired.metric.isin(["off_cross_rms", "wbi_cross"])
        & (paired.model != "ECG-FM")
    ]
    models = list(dict.fromkeys(selected.model))
    counts = (
        selected.drop_duplicates("model")
        .set_index("model")
        .concepts_paired.astype(int)
        .to_dict()
    )
    labels = [f"{model} (n={counts[model]})" for model in models]
    y = np.arange(len(models))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.8),
        sharey=True,
        constrained_layout=True,
    )
    for axis, metric, title in zip(
        axes,
        ("off_cross_rms", "wbi_cross"),
        (r"$\Delta$ off-target RMS", r"$\Delta$ WBI"),
    ):
        rows = (
            selected[selected.metric == metric]
            .set_index("model")
            .reindex(models)
        )
        estimates = rows.observed_delta.to_numpy(dtype=float)
        ci_low = rows.ci_low.to_numpy(dtype=float)
        ci_high = rows.ci_high.to_numpy(dtype=float)
        axis.errorbar(
            estimates,
            y,
            xerr=np.vstack((estimates - ci_low, ci_high - estimates)),
            fmt="o",
            color="#E45756",
            ecolor="#E45756",
            elinewidth=1.8,
            capsize=3,
            markersize=6,
            zorder=3,
        )
        axis.axvline(0, color="#333333", linewidth=1, linestyle="--")
        span = max(abs(float(ci_low.min())), 1e-6)
        axis.set_xlim(float(ci_low.min()) - 0.08 * span, 0.08 * span)
        axis.set_title(title)
        axis.set_xlabel("SAE - Dense")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.savefig(output / "final_layer_matched_effect_deltas.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "final_layer_matched_effect_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    readouts = completed(args.readouts_root)
    workers = completed(args.workers_root)
    bootstraps = completed(args.bootstrap_root)
    if len(readouts) != args.expected_readouts:
        raise RuntimeError(f"expected {args.expected_readouts} readouts, found {len(readouts)}")
    if len(workers) != args.expected_workers:
        raise RuntimeError(f"expected {args.expected_workers} workers, found {len(workers)}")
    if len(bootstraps) != args.expected_models:
        raise RuntimeError(f"expected {args.expected_models} bootstraps, found {len(bootstraps)}")
    worker_frame = pd.DataFrame(payload for _, payload in workers)
    if worker_frame.model.nunique() != args.expected_models:
        raise RuntimeError("worker model coverage is incomplete")

    design = pd.concat(
        [pd.read_csv(Path(payload["design_cells"])) for _, payload in workers],
        ignore_index=True,
    )
    profiles = pd.concat(
        [pd.read_csv(Path(payload["method_profile"])) for _, payload in bootstraps],
        ignore_index=True,
    )
    paired = pd.concat(
        [pd.read_csv(Path(payload["paired_table"])) for _, payload in bootstraps],
        ignore_index=True,
    )
    paired["q_value_bh"] = np.nan
    for metric, indices in paired.groupby("metric").groups.items():
        pvalues = paired.loc[indices, "p_value_two_sided"]
        finite = pvalues.notna()
        if finite.any():
            paired.loc[pvalues.index[finite], "q_value_bh"] = bh_adjust(
                pvalues.loc[finite].to_numpy()
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    design.to_csv(args.output_root / "design_cells.csv", index=False)
    profiles.to_csv(args.output_root / "model_method_profile.csv", index=False)
    paired.to_csv(args.output_root / "paired_model_bootstrap_fdr.csv", index=False)
    make_figure(profiles, args.output_root)
    make_delta_figure(paired, args.output_root)
    primary = paired[paired.metric.isin(["off_cross_rms", "wbi_cross"])].copy()
    primary["sae_favorable_significant"] = (
        (primary.observed_delta < 0) & (primary.q_value_bh < 0.05)
    )
    eligible = design.groupby(
        ["model", "candidate_arm", "k", "method"], as_index=False
    ).agg(
        design_rows=("status", "size"),
        eligible_rows=("status", lambda values: int(np.sum(values == "eligible"))),
    )
    eligible.to_csv(args.output_root / "eligibility_profile.csv", index=False)
    matched_dense = primary[
        (primary.candidate_arm == "matched_768")
        & (primary.k == 5)
        & (primary.comparison == "sae_minus_dense")
    ]
    matched_pca = primary[
        (primary.candidate_arm == "matched_768")
        & (primary.k == 5)
        & (primary.comparison == "sae_minus_pca")
    ]
    dense_evaluable = matched_dense[matched_dense.observed_delta.notna()]
    pca_evaluable = matched_pca[matched_pca.observed_delta.notna()]

    def models_favorable_on_both(values: pd.DataFrame) -> int:
        return int(
            sum(
                len(group) == 2 and group.sae_favorable_significant.all()
                for _, group in values.groupby("model")
            )
        )

    audit = {
        "status": "complete",
        "protocol": PROTOCOL,
        "models": int(profiles.model.nunique()),
        "readout_cells": len(readouts),
        "worker_cells": len(workers),
        "bootstrap_cells": len(bootstraps),
        "design_rows": len(design),
        "concepts": 49,
        "methods": list(METHOD_ORDER),
        "candidate_arms": sorted(profiles.candidate_arm.unique().tolist()),
        "ks": sorted(int(value) for value in profiles.k.unique()),
        "bootstrap_draws": int(bootstraps[0][1]["bootstrap_draws"]),
        "primary_tests": len(primary),
        "sae_favorable_significant_primary_tests": int(
            primary.sae_favorable_significant.sum()
        ),
        "matched_k5_sae_vs_dense_favorable_tests": int(
            dense_evaluable.sae_favorable_significant.sum()
        ),
        "matched_k5_sae_vs_dense_evaluable_tests": len(dense_evaluable),
        "matched_k5_sae_vs_dense_models_favorable_on_both_metrics": models_favorable_on_both(
            dense_evaluable
        ),
        "matched_k5_sae_vs_dense_evaluable_models": int(dense_evaluable.model.nunique()),
        "matched_k5_sae_vs_pca_favorable_tests": int(
            pca_evaluable.sae_favorable_significant.sum()
        ),
        "matched_k5_sae_vs_pca_evaluable_tests": len(pca_evaluable),
        "matched_k5_sae_vs_pca_models_favorable_on_both_metrics": models_favorable_on_both(
            pca_evaluable
        ),
        "matched_k5_sae_vs_pca_evaluable_models": int(pca_evaluable.model.nunique()),
        "matched_k5_nonevaluable_models": sorted(
            set(profiles.model.unique()) - set(dense_evaluable.model.unique())
        ),
        "fdr_family": "all model x arm x k x comparison tests, separately by metric",
        "seed_eligibility_gate": "all three SAE seeds for primary paired inference",
        "claim_boundary": "frozen linear-readout intervention response, not waveform or biological causality",
    }
    atomic_json(args.output_root / "audit.json", audit)
    k5 = profiles[
        (profiles.candidate_arm == "matched_768")
        & (profiles.k == 5)
        & (profiles.profile_scope == "all_seeds_primary")
        & profiles.method.isin(DISPLAY_METHODS)
    ].copy()
    report_primary = primary[
        (primary.candidate_arm == "matched_768")
        & (primary.k == 5)
        & (primary.comparison == "sae_minus_dense")
    ].copy()
    k5.to_csv(args.output_root / "final_layer_matched_effect_dense_sae_profile.csv", index=False)
    report_primary.to_csv(
        args.output_root / "final_layer_matched_effect_sae_vs_dense_fdr.csv", index=False
    )
    report = [
        "# Final-layer validation-matched intervention",
        "",
        "Features and centroids are selected on training data. Dense and SAE doses "
        "are calibrated to the same validation target-readout effect (up to 0.25 SD) and "
        "then frozen on the patient-disjoint test split.",
        "Primary paired inference requires a concept to pass the validation effect gate "
        "for all three SAE seeds; any-seed profiles are retained as sensitivity outputs.",
        "",
        "## Matched-768, k=5",
        "",
        markdown_table(
            k5[
                [
                    "model",
                    "method",
                    "concepts_eligible",
                    "target_delta",
                    "target_retention",
                    "off_cross_rms",
                    "wbi_cross",
                    "activation_l2",
                ]
            ].round(4)
        ),
        "",
        "## Paired patient bootstrap with BH-FDR",
        "",
        markdown_table(report_primary.round(4)),
        "",
        f"Across all-seed eligible concepts, SAE is favorable and FDR-significant versus "
        f"Dense on {int(dense_evaluable.sae_favorable_significant.sum())}/{len(dense_evaluable)} "
        f"evaluable metric tests and on both primary metrics in "
        f"{models_favorable_on_both(dense_evaluable)}/{dense_evaluable.model.nunique()} "
        f"evaluable models.",
        "ECG-FM has no concept passing the validation effect gate in all three SAE seeds; "
        "its any-seed sensitivity output is retained but excluded from primary inference.",
        "",
        "Lower off-target RMS and WBI are favorable. Ineligible concepts remain in the "
        "fixed denominator and are reported separately from conditional selectivity.",
    ]
    (args.output_root / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
