#!/usr/bin/env python
"""Plot final-layer held-out concept counts for native dense and SAE features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("CSFM", "CARDIAC-FM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM")
METHODS = ("dense_native_768", "sae_full_6144")
LABELS = {
    "dense_native_768": "Dense native (768)",
    "sae_full_6144": "SAE full (6144)",
}
COLORS = {
    "dense_native_768": "#3569a8",
    "sae_full_6144": "#d55e32",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "results/dictionary_accessibility_e8_v1/summary/all_target_profiles.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT
        / "results/dictionary_accessibility_e8_v1/summary/final_layer_sae_dense_concept_count",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data[
        np.isclose(data.relative_depth, 1.0)
        & data.model.isin(MODELS)
        & data.method.isin(METHODS)
    ].copy()
    expected_rows = len(MODELS) * (49 + 9) * (1 + 3)
    if len(data) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} final-layer target rows, found {len(data)}")

    counts = (
        data.groupby(
            ["model", "target_type", "method", "replicate"],
            as_index=False,
        )
        .agg(concept_count=("covered_primary", "sum"), targets=("target", "nunique"))
    )
    expected_targets = {"waveform": 49, "diagnosis": 9}
    for target_type, expected in expected_targets.items():
        observed = counts.loc[counts.target_type == target_type, "targets"]
        if not (observed == expected).all():
            raise RuntimeError(f"{target_type} target count mismatch: {observed.tolist()}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.35))
    x = np.arange(len(MODELS), dtype=float)
    width = 0.36

    for axis, target_type in zip(axes, ("waveform", "diagnosis")):
        panel = counts[counts.target_type == target_type]
        for method, offset in zip(METHODS, (-width / 2, width / 2)):
            method_rows = panel[panel.method == method]
            values = []
            replicate_values: list[np.ndarray] = []
            for model in MODELS:
                model_values = method_rows.loc[
                    method_rows.model == model, "concept_count"
                ].to_numpy(dtype=float)
                expected_replicates = 1 if method == "dense_native_768" else 3
                if len(model_values) != expected_replicates:
                    raise RuntimeError(
                        f"{model}/{target_type}/{method}: expected "
                        f"{expected_replicates} replicates, found {len(model_values)}"
                    )
                values.append(float(model_values.mean()))
                replicate_values.append(model_values)
            positions = x + offset
            bars = axis.bar(
                positions,
                values,
                width=width,
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                label=LABELS[method],
                zorder=2,
            )
            if method == "sae_full_6144":
                jitter = np.array([-0.055, 0.0, 0.055])
                for position, model_values in zip(positions, replicate_values):
                    axis.scatter(
                        position + jitter,
                        model_values,
                        s=19,
                        facecolor="white",
                        edgecolor="#8f351c",
                        linewidth=0.8,
                        zorder=3,
                    )
            labels = [f"{value:.1f}" if value % 1 else f"{int(value)}" for value in values]
            axis.bar_label(bars, labels=labels, padding=2, fontsize=8)

        maximum = expected_targets[target_type]
        axis.set_ylim(0, maximum * 1.13)
        axis.set_yticks(np.arange(0, maximum + 1, 10 if target_type == "waveform" else 2))
        axis.set_xticks(x, MODELS, rotation=24, ha="right")
        axis.set_ylabel(f"Covered concepts (out of {maximum})")
        axis.set_title(
            "Waveform concepts ($|r|\\geq0.20$)"
            if target_type == "waveform"
            else "Diagnosis concepts (AUROC $\\geq0.70$)"
        )
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(
        "Final-layer held-out concept coverage (CSFM L6; other encoders L12)",
        y=1.08,
        fontsize=13,
    )
    fig.tight_layout()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
