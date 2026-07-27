#!/usr/bin/env python
"""Plot held-out feature peaks for dense, SAE, and random dictionaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("dense_native_768", "sae_full_6144", "random_full_6144")
LABELS = {
    "dense_native_768": "Dense native (768)",
    "sae_full_6144": "SAE full (6144)",
    "random_full_6144": "Random full (6144)",
}
COLORS = {
    "dense_native_768": "#3569a8",
    "sae_full_6144": "#d55e32",
    "random_full_6144": "#4f8b63",
}
MARKERS = {
    "dense_native_768": "o",
    "sae_full_6144": "s",
    "random_full_6144": "^",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT
        / "results/dictionary_accessibility_e8_v1/summary/depth_method_summary.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=ROOT
        / "results/dictionary_accessibility_e8_v1/summary/dictionary_feature_peak_comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    data = data[data.method.isin(METHODS)].copy()
    expected = 2 * 6 * 5 * len(METHODS)
    if len(data) != expected or data.mean_feature_max.isna().any():
        raise RuntimeError(f"expected {expected} complete method rows, found {len(data)}")

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
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharex=True)
    depths = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    for axis, target_type in zip(axes, ("waveform", "diagnosis")):
        panel = data[data.target_type == target_type]
        for method in METHODS:
            values = panel[panel.method == method]
            pivot = values.pivot(index="relative_depth", columns="model", values="mean_feature_max")
            pivot = pivot.reindex(depths)
            for column in pivot.columns:
                axis.plot(
                    depths,
                    pivot[column],
                    color=COLORS[method],
                    linewidth=0.65,
                    alpha=0.16,
                    zorder=1,
                )
                axis.scatter(
                    depths,
                    pivot[column],
                    color=COLORS[method],
                    s=11,
                    alpha=0.24,
                    edgecolors="none",
                    zorder=2,
                )
            mean = pivot.mean(axis=1)
            axis.plot(
                depths,
                mean,
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=5.5,
                linewidth=2.2,
                label=LABELS[method],
                zorder=3,
            )

        dense = panel[panel.method == "dense_native_768"].set_index(["model", "relative_depth"])
        sae = panel[panel.method == "sae_full_6144"].set_index(["model", "relative_depth"])
        random = panel[panel.method == "random_full_6144"].set_index(["model", "relative_depth"])
        sae_random = sae.mean_feature_max - random.mean_feature_max
        sae_dense = sae.mean_feature_max - dense.mean_feature_max
        annotation = (
            f"SAE > random: {(sae_random > 0).sum()}/30"
            f"  (mean delta {sae_random.mean():+.3f})\n"
            f"SAE > dense: {(sae_dense > 0).sum()}/30"
            f"  (mean delta {sae_dense.mean():+.3f})"
        )
        axis.text(
            0.03,
            0.04,
            annotation,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.7,
            bbox={"facecolor": "white", "edgecolor": "#b8b8b8", "alpha": 0.94, "pad": 4},
        )
        axis.set_title("Waveform concepts" if target_type == "waveform" else "Diagnosis concepts")
        axis.set_xlabel("Relative encoder depth")
        axis.set_xticks(depths)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_ylabel("Held-out feature peak |r|")
    axes[1].set_ylabel("Held-out feature peak AUROC")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Peak concept accessibility across ECG foundation-model depth", y=1.08, fontsize=13)
    fig.tight_layout()

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        path = args.output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
