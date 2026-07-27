#!/usr/bin/env python
"""Build lightweight README figures and checksums from final summary artifacts."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd

try:
    from paper_figure_style import configure_paper_fonts
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts.paper_figure_style import configure_paper_fonts


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
FINAL = ROOT / "results" / "external_benchmark_v1" / "final"
SUMMARY = ROOT / "results" / "external_benchmark_v1" / "summary"
SOURCE = ROOT / "results" / "sae_reconciliation" / "matched_scale_v1"

MODELS = ["CSFM", "CARDIAC-FM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]
COHORTS = ["chapman_f", "cpsc_f", "ningbo_f", "mimic_f"]


def style() -> None:
    configure_paper_fonts()
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 140,
        "savefig.dpi": 180,
    })


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def workflow() -> None:
    stages = [
        ("Frozen ECG\nencoder", "6 models"),
        ("Activation\naudit", "30 model-cohort pairs"),
        ("Readout +\nprobes", "held-out evaluation"),
        ("Matched-scale\nSAE", "source / local / adapted"),
        ("Interventions", "steering / LEACE\n/ closure"),
        ("Bootstrap +\nFDR", "gated final report"),
    ]
    colors = ["#35618f", "#3f7d72", "#6b7f3f", "#b06b35", "#8a5377", "#555c66"]
    fig, ax = plt.subplots(figsize=(13, 2.45))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.45)
    ax.axis("off")
    width, gap, y = 1.78, 0.38, 0.62
    for i, ((title, subtitle), color) in enumerate(zip(stages, colors)):
        x = 0.12 + i * (width + gap)
        ax.add_patch(Rectangle((x, y), width, 1.12, facecolor="#f8fafb", edgecolor=color, linewidth=2))
        ax.text(x + width / 2, y + 0.72, title, ha="center", va="center", fontsize=9,
                weight="bold", color="#20262d", linespacing=1.05)
        ax.text(x + width / 2, y + 0.32, subtitle, ha="center", va="center", fontsize=8.3,
                color="#59636e", linespacing=1.05)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + width + 0.04, y + 0.56),
                                         (x + width + gap - 0.04, y + 0.56),
                                         arrowstyle="-|>", mutation_scale=12,
                                         linewidth=1.4, color="#78828c"))
    ax.text(0.12, 2.12, "Benchmark workflow", fontsize=13, weight="bold", color="#20262d")
    save(fig, "benchmark_workflow.png")


def source_fidelity() -> None:
    frame = pd.read_csv(SOURCE / "matched_scale_model_profile.csv").set_index("model").loc[MODELS]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    eligible = frame["matched_scale_primary_eligible"].astype(bool).to_numpy()
    colors = np.where(eligible, "#3f7d72", "#aeb6bf")
    axes[0].bar(x, frame["recon_R2_mean"], color=colors)
    axes[0].axhline(0.90, color="#a14343", linestyle="--", linewidth=1.3, label="gate = 0.90")
    axes[0].set_ylim(0.84, 1.00)
    axes[0].set_ylabel("Reconstruction explained variance")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].bar(x, frame["dead_fraction_max"], color=colors)
    axes[1].axhline(0.20, color="#a14343", linestyle="--", linewidth=1.3, label="gate < 0.20")
    axes[1].set_ylim(0, max(0.40, frame["dead_fraction_max"].max() * 1.08))
    axes[1].set_ylabel("Maximum dead-feature fraction")
    axes[1].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.set_xticks(x, frame.index, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.22)
    fig.suptitle("Source matched-scale SAE fidelity (green = primary eligible)", weight="bold")
    fig.tight_layout()
    save(fig, "source_sae_fidelity.png")


def steering_tiers() -> None:
    frame = pd.read_csv(FINAL / "steering_protocol_counts.csv").set_index("protocol")
    order = ["frozen_atom", "local_atom", "cohort_adapted_atom"]
    labels = ["Frozen", "Local re-rank", "Cohort-adapted"]
    tiers = ["tier0", "tier1", "tier2", "tier3"]
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    x = np.arange(len(order)); width = 0.19
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    for i, (tier, color) in enumerate(zip(tiers, colors)):
        rates = frame.loc[order, tier].to_numpy() / frame.loc[order, "cells"].to_numpy()
        bars = ax.bar(x + (i - 1.5) * width, rates, width, label=tier.upper(), color=color)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in rates], fontsize=11, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_ylabel("Pass rate among 270 seed-level cells", fontsize=13)
    ax.set_title("External steering gates by atom protocol", weight="bold", fontsize=15)
    ax.grid(False)
    ax.legend(ncol=4, frameon=False, loc="upper left", fontsize=12)
    fig.tight_layout()
    save(fig, "external_steering_tiers.png")


def frozen_gate_heatmap() -> None:
    frame = pd.read_csv(SUMMARY / "external_steering_target_profile.csv")
    frame = frame[frame["protocol"].eq("frozen_atom")].copy()
    grouped = frame.groupby(["model", "cohort"]).agg(passed=("tier0_pass", "sum"), total=("seeds", "sum"))
    values = (grouped["passed"] / grouped["total"]).unstack("cohort").reindex(index=MODELS, columns=COHORTS)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    image = ax.imshow(values.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            ax.text(j, i, "n/a" if pd.isna(value) else f"{value:.0%}", ha="center", va="center",
                    color="white" if value >= 0.55 else "#20262d", weight="bold")
    ax.set_xticks(range(4), ["Chapman", "CPSC", "Ningbo", "MIMIC"])
    ax.set_yticks(range(6), MODELS)
    ax.set_title("Frozen-Atom Tier 0 eligibility by model and cohort", weight="bold")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("Eligible seed-cell fraction")
    fig.tight_layout()
    save(fig, "frozen_atom_tier0_heatmap.png")


def checksums() -> None:
    paths = [
        FINAL / "completion_audit.csv",
        FINAL / "completion_audit.json",
        FINAL / "external_pair_profile.csv",
        FINAL / "steering_protocol_counts.csv",
        SUMMARY / "external_steering_cells.csv",
        SUMMARY / "external_steering_target_profile.csv",
        SOURCE / "matched_scale_model_profile.csv",
        SOURCE / "matched_scale_stable_main_targets.csv",
    ]
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    (FINAL / "readme_artifact_checksums.sha256").write_text("\n".join(lines) + "\n")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    workflow()
    source_fidelity()
    steering_tiers()
    frozen_gate_heatmap()
    checksums()
    print(f"Wrote 4 figures to {FIGURES} and final artifact checksums to {FINAL}")


if __name__ == "__main__":
    main()
