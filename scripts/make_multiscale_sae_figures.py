#!/usr/bin/env python
"""Create paper-ready figures and tables for the multi-scale SAE benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from paper_figure_style import configure_paper_fonts
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts.paper_figure_style import configure_paper_fonts


ROOT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["CARDIAC-FM", "CSFM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)


def add_panel_labels(axes, *, y: float = 1.04) -> None:
    for letter, axis in zip("abcdefghijklmnopqrstuvwxyz", axes):
        axis.text(
            -0.10,
            y,
            f"({letter})",
            transform=axis.transAxes,
            fontsize=10.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )


def workflow_figure(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    fig, axis = plt.subplots(figsize=(11.2, 4.4))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    def box(x, y, width, height, title, lines, edge, face="#FFFFFF"):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.2,
            edgecolor=edge,
            facecolor=face,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height - 0.045, title, ha="center", va="center", fontsize=9, fontweight="bold")
        axis.text(x + width / 2, y + height / 2 - 0.025, "\n".join(lines), ha="center", va="center", fontsize=7.5, linespacing=1.35)
        return patch

    def arrow(x1, y1, x2, y2):
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#555555",
            )
        )

    box(
        0.02,
        0.30,
        0.14,
        0.48,
        "PTB-XL source",
        ["21,799 ECGs", "patient-disjoint", "train / val / test", "49 concepts"],
        "#0072B2",
        "#EEF6FB",
    )
    box(
        0.20,
        0.22,
        0.17,
        0.64,
        "Six frozen ECG FMs",
        ["CARDIAC-FM", "CSFM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"],
        "#009E73",
        "#EEF9F5",
    )
    arrow(0.16, 0.54, 0.20, 0.54)

    grid_x, grid_y, grid_w, grid_h = 0.42, 0.28, 0.25, 0.54
    axis.add_patch(
        FancyBboxPatch(
            (grid_x, grid_y),
            grid_w,
            grid_h,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.2,
            edgecolor="#CC79A7",
            facecolor="#FFF7FB",
        )
    )
    axis.text(grid_x + grid_w / 2, grid_y + grid_h - 0.045, "Matched depth-scale SAE atlas", ha="center", va="center", fontsize=9, fontweight="bold")
    axis.text(
        grid_x + grid_w / 2,
        grid_y + grid_h - 0.125,
        "same grid for every FM\nhighlighted row: compare E=8 with E=8 only",
        ha="center",
        va="center",
        fontsize=6.9,
        linespacing=1.2,
    )
    depths = ["0", ".25", ".5", ".75", "1"]
    expansions = ["1", "4", "8", "16", "32"]
    cell = 0.034
    start_x = grid_x + 0.060
    start_y = grid_y + 0.105
    for row_index, expansion in enumerate(expansions):
        for column_index, depth in enumerate(depths):
            face = "#F3CFE4" if row_index == 2 else "#F9EAF3"
            axis.add_patch(
                Rectangle(
                    (start_x + column_index * cell, start_y + row_index * cell),
                    cell - 0.003,
                    cell - 0.003,
                    facecolor=face,
                    edgecolor="#FFFFFF",
                    linewidth=0.6,
                )
            )
        axis.text(start_x - 0.014, start_y + row_index * cell + cell / 2, expansion, ha="right", va="center", fontsize=6.8)
    for column_index, depth in enumerate(depths):
        axis.text(start_x + column_index * cell + cell / 2, start_y - 0.014, depth, ha="center", va="top", fontsize=6.8)
    axis.text(start_x - 0.043, start_y + 2.5 * cell, "$E=N/d_{FM}$", rotation=90, ha="center", va="center", fontsize=7.3)
    axis.text(start_x + 2.5 * cell, start_y - 0.052, "relative depth", ha="center", va="center", fontsize=7.3)
    arrow(0.37, 0.54, 0.42, 0.54)

    box(
        0.72,
        0.28,
        0.26,
        0.54,
        "Matched FM inference",
        [
            "fidelity: reconstruction / dead features",
            "semantics: train-selected |r| / coverage",
            "reproducibility: seed matching / subspace",
            "paired patient bootstrap at each common E",
            "same-five-scale AUC (no best-scale ranking)",
        ],
        "#D55E00",
        "#FFF4EC",
    )
    arrow(0.67, 0.54, 0.72, 0.54)

    axis.add_patch(
        FancyBboxPatch(
            (0.20, 0.04),
            0.78,
            0.11,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.0,
            linestyle="--",
            edgecolor="#666666",
            facecolor="#F6F6F6",
        )
    )
    axis.text(0.59, 0.095, "Stage 2 (validation-selected points only): external transport  |  controlled intervention  |  waveform grounding", ha="center", va="center", fontsize=8.0)
    arrow(0.85, 0.28, 0.85, 0.15)
    axis.text(0.5, 0.94, "ECG-FM-InterpBench: the FM is the object; matched-scale SAE is the measurement instrument", ha="center", va="center", fontsize=11, fontweight="bold")
    save_figure(fig, out)
    plt.close(fig)


def atlas_figure(surface: pd.DataFrame, metric: str, label: str, cmap: str, out: Path) -> None:
    import matplotlib.pyplot as plt

    depths = sorted(surface.relative_depth.unique())
    expansions = sorted(surface.expansion_E.unique())
    values = surface[metric].to_numpy(dtype=float)
    vmin = 0.0 if "dead_fraction" in metric else float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.4), constrained_layout=True, sharex=True, sharey=True)
    image = None
    for axis, model in zip(axes.ravel(), MODEL_ORDER):
        part = surface[surface.model == model]
        matrix = np.full((len(expansions), len(depths)), np.nan, dtype=float)
        for row in part.itertuples(index=False):
            matrix[expansions.index(row.expansion_E), depths.index(row.relative_depth)] = float(
                getattr(row, metric)
            )
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(model, fontsize=9)
        axis.set_xticks(range(len(depths)), [f"{depth:g}" for depth in depths])
        axis.set_yticks(range(len(expansions)), [f"{int(expansion)}" for expansion in expansions])
        axis.tick_params(labelsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("Relative depth", fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Expansion N/d", fontsize=9)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
    save_figure(fig, out)
    plt.close(fig)


def scale_curves(surface: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("validation_recon_R2_mean", "Validation reconstruction $R^2$"),
        ("validation_semantic_alignment_mean", "Train-selected validation |r|"),
        ("validation_dead_fraction_mean", "Validation dead-feature fraction"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, panels):
        for model, color in zip(MODEL_ORDER, COLORS):
            part = surface[surface.model == model]
            curve = part.groupby("expansion_E", as_index=False)[metric].mean().sort_values("expansion_E")
            axis.plot(
                curve.expansion_E,
                curve[metric],
                marker="o",
                linewidth=1.5,
                markersize=3.5,
                color=color,
                label=model,
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 8, 16, 32], ["1", "4", "8", "16", "32"])
        axis.set_xlabel("Expansion N/d", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        clean_axis(axis)
        axis.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    save_figure(fig, out)
    plt.close(fig)


def stability_curves(stability: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=False)
    panels = [
        ("stability_above_random_mean", "Matched cosine above random"),
        ("subspace_overlap_mean", "Top-feature subspace overlap"),
    ]
    for axis, (metric, ylabel) in zip(axes, panels):
        for model, color in zip(MODEL_ORDER, COLORS):
            part = stability[stability.model == model]
            curve = part.groupby("expansion_E", as_index=False)[metric].mean().sort_values("expansion_E")
            axis.plot(curve.expansion_E, curve[metric], marker="o", linewidth=1.5, color=color, label=model)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 8, 16, 32], ["1", "4", "8", "16", "32"])
        axis.set_xlabel("Expansion N/d", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        clean_axis(axis)
        axis.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], marker="o", color=color, linewidth=1.5, markersize=4, label=model)
        for model, color in zip(MODEL_ORDER, COLORS)
    ]
    fig.legend(handles=handles, fontsize=7.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.10, right=0.995, top=0.96, bottom=0.31, wspace=0.28)
    save_figure(fig, out)
    plt.close(fig)


def patient_scale_curves(profile: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("recon_R2", "Test reconstruction $R^2$"),
        ("semantic_alignment", "Train-selected test |r|"),
        ("concept_coverage_020", "Concept coverage, |r| $\\geq$ 0.20"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.45), constrained_layout=False)
    for axis, (metric, ylabel) in zip(axes, panels):
        metric_rows = profile[profile.metric == metric]
        for model, color in zip(MODEL_ORDER, COLORS):
            part = metric_rows[metric_rows.model == model].sort_values("expansion_E")
            x = part.expansion_E.to_numpy(dtype=float)
            y = part.observed.to_numpy(dtype=float)
            low = part.ci_low.to_numpy(dtype=float)
            high = part.ci_high.to_numpy(dtype=float)
            axis.plot(x, y, marker="o", linewidth=1.5, markersize=3.5, color=color, label=model)
            axis.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 8, 16, 32], ["1", "4", "8", "16", "32"])
        axis.set_xlabel("Matched expansion $E=N/d_{FM}$", fontsize=9)
        axis.set_ylabel(ylabel, fontsize=9)
        clean_axis(axis)
        axis.tick_params(labelsize=8)
    add_panel_labels(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.25, wspace=0.34)
    save_figure(fig, out)
    plt.close(fig)


def patient_profile_forest(profile: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [
        ("recon_R2", "Reconstruction AUC"),
        ("semantic_alignment", "Semantic AUC"),
        ("concept_coverage_020", "Coverage AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.75), constrained_layout=False, sharey=True)
    y = np.arange(len(MODEL_ORDER))[::-1]
    for axis, (metric, xlabel) in zip(axes, panels):
        part = profile[profile.metric == metric].set_index("model").loc[MODEL_ORDER]
        point = part.observed_common_scale_auc.to_numpy(dtype=float)
        low = part.ci_low.to_numpy(dtype=float)
        high = part.ci_high.to_numpy(dtype=float)
        for index, color in enumerate(COLORS):
            axis.errorbar(
                point[index],
                y[index],
                xerr=[[point[index] - low[index]], [high[index] - point[index]]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2,
                markersize=4,
                linewidth=1.2,
            )
        axis.set_xlabel(xlabel, fontsize=9)
        clean_axis(axis)
        axis.tick_params(labelsize=8)
    add_panel_labels(axes)
    axes[0].set_yticks(y, MODEL_ORDER)
    handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="None", markersize=4, label=model)
        for model, color in zip(MODEL_ORDER, COLORS)
    ]
    fig.legend(handles=handles, fontsize=7, frameon=False, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.12, right=0.995, top=0.96, bottom=0.24, wspace=0.16)
    save_figure(fig, out)
    plt.close(fig)


def semantic_summary_figure(
    surface: pd.DataFrame,
    model_profiles: pd.DataFrame,
    concept_profiles: pd.DataFrame,
    concept_registry: pd.DataFrame,
    out: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    family_labels = {
        "RATE_RHYTHM": "Rate /\nrhythm",
        "INTERVAL": "Intervals",
        "AXIS": "Axis",
        "AMPLITUDE": "Amplitude",
        "ST_T": "ST-T",
        "MORPHOLOGY": "Morphology",
    }
    family_order = [family for family in family_labels if family in set(concept_registry.family)]
    concept_to_family = concept_registry.set_index("concept_id")["family"].to_dict()
    concept_profiles = concept_profiles.copy()
    concept_profiles["family"] = concept_profiles.concept.map(concept_to_family)
    concept_profiles = concept_profiles[
        concept_profiles.family.isin(family_order) & concept_profiles.model.isin(MODEL_ORDER)
    ]

    depths = sorted(surface.relative_depth.unique())
    expansions = sorted(surface.expansion_E.unique())
    values = surface.test_semantic_alignment_mean.to_numpy(dtype=float)
    vmin = max(0.0, float(np.nanmin(values)) - 0.01)
    vmax = float(np.nanmax(values))

    fig = plt.figure(figsize=(11.2, 7.45), constrained_layout=True)
    grid = GridSpec(4, 6, figure=fig, height_ratios=[1.0, 1.0, 0.12, 0.98])
    heat_grid = grid[:2, :].subgridspec(2, 3, wspace=0.18, hspace=0.10)
    heat_axes = [fig.add_subplot(heat_grid[row, col]) for row in range(2) for col in range(3)]
    image = None
    for axis, model in zip(heat_axes, MODEL_ORDER):
        part = surface[surface.model == model]
        matrix = np.full((len(expansions), len(depths)), np.nan, dtype=float)
        for row in part.itertuples(index=False):
            matrix[expansions.index(row.expansion_E), depths.index(row.relative_depth)] = float(
                row.test_semantic_alignment_mean
            )
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="Blues_r", vmin=vmin, vmax=vmax)
        peak = np.unravel_index(np.nanargmax(matrix), matrix.shape)
        axis.scatter(
            peak[1],
            peak[0],
            marker="*",
            s=90,
            facecolors="#FFD166",
            edgecolors="#111111",
            linewidths=0.8,
            zorder=4,
        )
        axis.set_title(model, fontsize=8.5, pad=2)
        axis.set_xticks(range(len(depths)), [f"{depth:g}" for depth in depths])
        axis.set_yticks(range(len(expansions)), [f"{int(expansion)}" for expansion in expansions])
        axis.tick_params(labelsize=7.5, length=2)
    for axis in heat_axes[3:]:
        axis.set_xlabel("Relative depth", fontsize=8)
    for axis in (heat_axes[0], heat_axes[3]):
        axis.set_ylabel("Expansion $E$", fontsize=8)
    if image is not None:
        colorbar = fig.colorbar(image, ax=heat_axes, shrink=0.80, pad=0.012)
        colorbar.set_label("Test mean |r|", fontsize=8)
        colorbar.ax.tick_params(labelsize=7.5)
    forest_axis = fig.add_subplot(grid[3, :3])
    semantic = model_profiles[model_profiles.metric == "semantic_alignment"].set_index("model").loc[MODEL_ORDER]
    y_positions = np.arange(len(MODEL_ORDER))[::-1]
    points = semantic.observed_common_scale_auc.to_numpy(dtype=float)
    lows = semantic.ci_low.to_numpy(dtype=float)
    highs = semantic.ci_high.to_numpy(dtype=float)
    for index, (model, color) in enumerate(zip(MODEL_ORDER, COLORS)):
        forest_axis.errorbar(
            points[index],
            y_positions[index],
            xerr=[[points[index] - lows[index]], [highs[index] - points[index]]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
            markersize=4.2,
            linewidth=1.2,
        )
    forest_axis.set_yticks(y_positions, MODEL_ORDER)
    forest_axis.set_xlabel("Common-scale semantic AUC", fontsize=8.5)
    clean_axis(forest_axis)
    forest_axis.tick_params(labelsize=7.5)
    family_axis = fig.add_subplot(grid[3, 3:])
    family_summary = (
        concept_profiles.groupby(["model", "family"], as_index=False)
        .observed_common_scale_auc_abs_correlation.mean()
    )
    x_positions = np.arange(len(family_order))
    offsets = np.linspace(-0.25, 0.25, len(MODEL_ORDER))
    for model_index, (model, color) in enumerate(zip(MODEL_ORDER, COLORS)):
        part = family_summary[family_summary.model == model].set_index("family")
        y_values = [part.loc[family, "observed_common_scale_auc_abs_correlation"] if family in part.index else np.nan for family in family_order]
        family_axis.scatter(
            x_positions + offsets[model_index],
            y_values,
            s=18,
            color=color,
            label=model,
            alpha=0.92,
        )
    family_axis.set_xticks(x_positions, [family_labels[family] for family in family_order])
    family_axis.set_ylabel("Mean concept AUC", fontsize=8.5)
    clean_axis(family_axis)
    family_axis.tick_params(labelsize=7.5)
    family_axis.legend(fontsize=6.4, frameon=False, ncol=2, loc="upper right")
    fig.canvas.draw()

    def panel_heading(axes, letter: str, title: str, y_pad: float = 0.028) -> None:
        axis_list = axes if isinstance(axes, list) else [axes]
        boxes = [axis.get_position() for axis in axis_list]
        x0 = min(box.x0 for box in boxes)
        x1 = max(box.x1 for box in boxes)
        y1 = max(box.y1 for box in boxes)
        y = y1 + y_pad
        fig.text(
            x0 - 0.026,
            y,
            f"({letter})",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            (x0 + x1) / 2,
            y,
            title,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
        )

    panel_heading(heat_axes, "a", "Depth-scale accessibility surfaces", y_pad=0.074)
    panel_heading(forest_axis, "b", "Model summary", y_pad=0.015)
    panel_heading(family_axis, "c", "Concept-family breakdown", y_pad=0.015)

    save_figure(fig, out)
    plt.close(fig)


def sparsity_sensitivity_figure(profiles: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [
        ("reconstruction", "Reconstruction AUC"),
        ("semantic_alignment", "Semantic AUC"),
        ("concept_coverage", "Coverage AUC"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.55), constrained_layout=False)
    for axis, (metric, label) in zip(axes, panels):
        part = profiles[profiles.metric == metric]
        pivot = part.pivot(index="model", columns="sparsity_arm", values="observed_common_scale_auc")
        pivot = pivot.loc[MODEL_ORDER]
        x = pivot["fixed_k_over_d"].to_numpy(dtype=float)
        y = pivot["fixed_k_over_n"].to_numpy(dtype=float)
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        padding = max((high - low) * 0.08, 1e-4)
        axis.plot([low - padding, high + padding], [low - padding, high + padding], color="#666666", linestyle="--", linewidth=0.9)
        for index, (model, color) in enumerate(zip(MODEL_ORDER, COLORS)):
            axis.scatter(x[index], y[index], color=color, s=26, label=model, zorder=3)
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        axis.set_xlabel("Fixed $k/d$", fontsize=9)
        axis.set_ylabel("Fixed $k/N$", fontsize=9)
        axis.set_title(label, fontsize=9)
        clean_axis(axis)
        axis.tick_params(labelsize=8)
    handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="None", markersize=4, label=model)
        for model, color in zip(MODEL_ORDER, COLORS)
    ]
    fig.legend(handles=handles, fontsize=7.2, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.30, wspace=0.32)
    save_figure(fig, out)
    plt.close(fig)


def robustness_five_panel_figure(stability: pd.DataFrame, profiles: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(12.4, 2.85), constrained_layout=False)
    stability_panels = [
        ("stability_above_random_mean", "Matched cosine\nabove random", "Matched cosine"),
        ("subspace_overlap_mean", "Top-feature\nsubspace overlap", "Subspace overlap"),
    ]
    for axis, (metric, ylabel, title) in zip(axes[:2], stability_panels):
        for model, color in zip(MODEL_ORDER, COLORS):
            part = stability[stability.model == model]
            curve = part.groupby("expansion_E", as_index=False)[metric].mean().sort_values("expansion_E")
            axis.plot(curve.expansion_E, curve[metric], marker="o", linewidth=1.1, markersize=2.6, color=color)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 8, 16, 32], ["1", "4", "8", "16", "32"])
        axis.set_xlabel("Expansion N/d", fontsize=8.6)
        axis.set_ylabel(ylabel, fontsize=8.6)
        axis.set_title(title, fontsize=9.2)
        clean_axis(axis)
        axis.tick_params(labelsize=7.8, length=2)

    sparsity_panels = [
        ("reconstruction", "Reconstruction AUC"),
        ("semantic_alignment", "Semantic AUC"),
        ("concept_coverage", "Coverage AUC"),
    ]
    for axis, (metric, title) in zip(axes[2:], sparsity_panels):
        part = profiles[profiles.metric == metric]
        pivot = part.pivot(index="model", columns="sparsity_arm", values="observed_common_scale_auc")
        pivot = pivot.loc[MODEL_ORDER]
        x = pivot["fixed_k_over_d"].to_numpy(dtype=float)
        y = pivot["fixed_k_over_n"].to_numpy(dtype=float)
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        padding = max((high - low) * 0.08, 1e-4)
        axis.plot([low - padding, high + padding], [low - padding, high + padding], color="#666666", linestyle="--", linewidth=0.75)
        for index, color in enumerate(COLORS):
            axis.scatter(x[index], y[index], color=color, s=15, zorder=3)
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        axis.set_xlabel("Fixed $k/d$", fontsize=8.6)
        axis.set_ylabel("Fixed $k/N$", fontsize=8.6)
        axis.set_title(title, fontsize=9.2)
        clean_axis(axis)
        axis.tick_params(labelsize=7.8, length=2)
    add_panel_labels(axes, y=1.08)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.82, bottom=0.24, wspace=0.55)
    save_figure(fig, out)
    plt.close(fig)


def model_legend_figure(out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(7.4, 0.45))
    handles = [
        Line2D([0], [0], marker="o", color=color, linewidth=1.2, markersize=4.0, label=model)
        for model, color in zip(MODEL_ORDER, COLORS)
    ]
    legend = fig.legend(handles=handles, fontsize=8.4, frameon=False, ncol=6, loc="center")
    fig.canvas.draw()
    legend.set_in_layout(True)
    save_figure(fig, out)
    plt.close(fig)


def latex_table(profiles: pd.DataFrame, path: Path) -> None:
    ordered = profiles.set_index("model").loc[MODEL_ORDER].reset_index()
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & Recon AUC & Semantic AUC & Coverage AUC & Dead & Fidelity \\",
        r"\midrule",
    ]
    for row in ordered.itertuples(index=False):
        lines.append(
            f"{row.model} & {row.test_multiscale_recon_auc:.3f} & "
            f"{row.test_multiscale_semantic_auc:.3f} & {row.test_multiscale_coverage_auc:.3f} & "
            f"{row.test_mean_dead_fraction:.3f} & {row.validation_fidelity_pass_fraction:.2f} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def latex_patient_table(profiles: pd.DataFrame, path: Path) -> None:
    labels = [
        ("recon_R2", "Recon AUC"),
        ("semantic_alignment", "Semantic AUC"),
        ("concept_coverage_020", "Coverage AUC"),
    ]
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        "Model & " + " & ".join(label for _, label in labels) + r" \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        cells = []
        for metric, _ in labels:
            row = profiles[(profiles.model == model) & (profiles.metric == metric)].iloc[0]
            cells.append(
                f"{row.observed_common_scale_auc:.3f} "
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
            )
        lines.append(model + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=ROOT / "results/multiscale_sae_fixed_k_over_n_middepth_v1",
    )
    args = parser.parse_args()
    out = args.out or (args.root / "figures")
    audit = json.loads((args.root / "audit.json").read_text())
    if not audit.get("audit_pass"):
        raise RuntimeError(f"cannot render incomplete multi-scale results: {audit}")

    import matplotlib

    matplotlib.use("Agg")
    configure_paper_fonts()
    surface = pd.read_csv(args.root / "layer_scale_surface.csv")
    profiles = pd.read_csv(args.root / "model_profiles.csv")
    workflow_figure(out / "multiscale_benchmark_workflow")
    atlas_figure(
        surface,
        "validation_recon_R2_mean",
        "Validation reconstruction $R^2$",
        "Blues_r",
        out / "multiscale_reconstruction_atlas",
    )
    atlas_figure(
        surface,
        "validation_dead_fraction_mean",
        "Validation dead-feature fraction",
        "Blues_r",
        out / "multiscale_dead_feature_atlas",
    )
    scale_curves(surface, out / "multiscale_model_curves")
    stability_path = args.root / "stability_layer_scale.csv"
    stability_df = None
    if stability_path.exists():
        stability_df = pd.read_csv(stability_path)
        stability_curves(stability_df, out / "multiscale_stability_curves")
    latex_table(profiles, args.root / "paper_table_multiscale.tex")
    patient_scale_path = args.root / "test_patient_bootstrap_matched_scales.csv"
    patient_profile_path = args.root / "test_patient_bootstrap_model_profiles.csv"
    patient_audit_path = args.root / "test_patient_bootstrap_audit.json"
    patient_audit = None
    if patient_scale_path.exists() or patient_profile_path.exists() or patient_audit_path.exists():
        if not all(path.exists() for path in (patient_scale_path, patient_profile_path, patient_audit_path)):
            raise RuntimeError("partial patient-bootstrap figure inputs")
        patient_audit = json.loads(patient_audit_path.read_text())
        if patient_audit.get("status") != "complete" or patient_audit.get("verified_tasks") != 450:
            raise RuntimeError(f"patient bootstrap is not complete: {patient_audit}")
        patient_scale_curves(
            pd.read_csv(patient_scale_path), out / "multiscale_patient_matched_scale_curves"
        )
        patient_profiles = pd.read_csv(patient_profile_path)
        concept_profile_path = args.root / "test_patient_bootstrap_concepts.csv"
        concept_registry_path = ROOT / "configs/concepts.csv"
        if not concept_profile_path.exists() or not concept_registry_path.exists():
            raise RuntimeError("missing semantic-summary figure inputs")
        semantic_summary_figure(
            surface,
            patient_profiles,
            pd.read_csv(concept_profile_path),
            pd.read_csv(concept_registry_path),
            out / "multiscale_semantic_atlas",
        )
        patient_profile_forest(
            patient_profiles, out / "multiscale_patient_common_scale_auc"
        )
        latex_patient_table(
            patient_profiles, args.root / "paper_table_multiscale_patient.tex"
        )
    sensitivity_path = args.sensitivity_root / "test_sparsity_auc_profiles.csv"
    sensitivity_audit_path = args.sensitivity_root / "test_sparsity_sensitivity_audit.json"
    sensitivity_audit = None
    sensitivity_df = None
    if sensitivity_path.exists() or sensitivity_audit_path.exists():
        if not sensitivity_path.exists() or not sensitivity_audit_path.exists():
            raise RuntimeError("partial sparsity-sensitivity figure inputs")
        sensitivity_audit = json.loads(sensitivity_audit_path.read_text())
        if sensitivity_audit.get("status") != "complete":
            raise RuntimeError(f"sparsity sensitivity is not complete: {sensitivity_audit}")
        sensitivity_df = pd.read_csv(sensitivity_path)
        sparsity_sensitivity_figure(
            sensitivity_df, out / "multiscale_sparsity_sensitivity"
        )
    if stability_df is not None and sensitivity_df is not None:
        robustness_five_panel_figure(stability_df, sensitivity_df, out / "multiscale_robustness_five_panel")
        model_legend_figure(out / "multiscale_robustness_legend")
    metadata = {
        "status": "complete",
        "source_audit": audit,
        "output_dir": str(out),
        "figures": sorted(path.name for path in out.glob("*.png")),
        "latex_table": str(args.root / "paper_table_multiscale.tex"),
        "patient_bootstrap_audit": patient_audit,
        "sparsity_sensitivity_audit": sensitivity_audit,
    }
    (args.root / "figure_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
