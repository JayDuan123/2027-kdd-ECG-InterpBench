#!/usr/bin/env python
"""Create paper-facing tables, shared-atom audit, figure, and detailed report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=BASE)
    return p.parse_args()


def main() -> None:
    a = parse_args(); summary = a.base / "summary"
    cells = pd.read_csv(summary / "steering_cells.csv")
    registry = pd.read_csv(a.base / "target_registry.csv")
    heads = json.loads((a.base / "frozen_heads.metrics.json").read_text())
    detail_rows = []; sensitivity_rows = []; atom_rows = []
    for path in sorted((a.base / "tasks").glob("seed*/*/result.json")):
        result = json.loads(path.read_text()); target = result["target"]; seed = int(result["seed"])
        metric = result["metrics"]["top5"][target]
        row = {"target": target, "seed": seed}
        for key in ("auroc_drop", "sens_drop_at_95spec", "decision_flip_rate", "brier_change", "ece_change",
                    "r2_drop", "mae_change"):
            row[key] = metric.get(key, np.nan)
        detail_rows.append(row)
        for topk in ("top1", "top5", "top10"):
            tm = result["metrics"][topk][target]
            sensitivity_rows.append({"target": target, "seed": seed, "intervention": topk,
                                     "primary_behavior_effect": tm.get("sens_drop_at_95spec", tm.get("r2_drop", np.nan)),
                                     "auroc_drop": tm.get("auroc_drop", np.nan), "decision_flip_rate": tm.get("decision_flip_rate", np.nan),
                                     "brier_change": tm.get("brier_change", np.nan), "ece_change": tm.get("ece_change", np.nan),
                                     "mae_change": tm.get("mae_change", np.nan)})
        atom_rows.append({"target": target, "seed": seed,
                          **{f"{topk}_atoms": " ".join(map(str, result["selected_atoms"][topk])) for topk in ("top1", "top5", "top10")}})
    cells = cells.merge(pd.DataFrame(detail_rows), on=["target", "seed"], how="left", validate="one_to_one")
    cells["behavior_interpretation"] = np.where(
        cells.tier3_behavior_changing,
        np.where(cells.target_type.eq("binary"), "operating_point_discrimination_change", "continuous_readout_change"),
        np.where(cells.target_type.eq("binary") & ((cells.brier_change.abs() >= 0.005) | (cells.ece_change.abs() >= 0.01)),
                 "calibration_shift_candidate_only", "no_detectable_behavior_change"))
    cells["readout_quality"] = cells.target.map(lambda t: heads[t].get("test_auroc", heads[t].get("test_r2", np.nan)))
    cells.to_csv(summary / "steering_cells_enriched.csv", index=False)
    pd.DataFrame(sensitivity_rows).to_csv(summary / "topk_intervention_sensitivity.csv", index=False)
    atoms = pd.DataFrame(atom_rows); atoms.to_csv(summary / "selected_atoms.csv", index=False)

    pairs = []
    for seed, group in atoms.groupby("seed"):
        records = list(group.itertuples(index=False))
        for left in records:
            for right in records:
                aa = set(map(int, left.top5_atoms.split())); bb = set(map(int, right.top5_atoms.split()))
                pairs.append({"seed": seed, "target_a": left.target, "target_b": right.target,
                              "intersection": len(aa & bb), "jaccard": len(aa & bb) / len(aa | bb)})
    overlap = pd.DataFrame(pairs); overlap.to_csv(summary / "shared_atom_audit.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = registry.target.tolist(); grouped = cells.groupby("target", as_index=False).agg(
        ste=("ste", "mean"), otd=("otd_mean", "mean"), tier1=("tier1_sparse_attribution", "sum"),
        tier2=("tier2_selective_steering", "sum"), tier3=("tier3_behavior_changing", "sum"))
    grouped["order"] = grouped.target.map({t: i for i, t in enumerate(order)}); grouped = grouped.sort_values("order")
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8), layout="constrained")
    roles = registry.set_index("target").analysis_role.to_dict(); colors = {"main": "#1f77b4", "positive_control": "#2ca02c", "nuisance_control": "#7f7f7f"}
    offsets = {"avb1": (5, 5), "hr_ventricular": (-5, -16), "lafb": (5, 8), "lbbb": (5, 7),
               "pvc": (-8, -14), "qrs_duration": (5, 14), "qrst_angle": (5, -15),
               "qtc_fridericia": (-4, -16), "rbbb": (5, -13), "st_amp_global": (5, 14)}
    for r in grouped.itertuples(index=False):
        axes[0].scatter(r.ste, r.otd, s=55, color=colors[roles[r.target]])
        if r.target in offsets:
            axes[0].annotate(r.target, (r.ste, r.otd), xytext=offsets[r.target], textcoords="offset points", fontsize=8)
    ylim = grouped.otd.max() * 1.35
    axes[0].plot([0, ylim], [0, ylim], "--", color="black", lw=1)
    axes[0].set_ylim(-.01, ylim)
    axes[0].set_xlim(-.12, grouped.ste.max() * 1.12)
    for role, color in colors.items():
        axes[0].scatter([], [], s=45, color=color, label=role)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[0].set(xlabel="Standardized target effect (STE)", ylabel="Mean off-target damage (OTD)", title="Top-5 code-space selectivity")
    x = np.arange(len(grouped)); width = .25
    axes[1].bar(x - width, grouped.tier1, width, label="Tier 1"); axes[1].bar(x, grouped.tier2, width, label="Tier 2"); axes[1].bar(x + width, grouped.tier3, width, label="Tier 3")
    axes[1].set_xticks(x, grouped.target, rotation=70, ha="right"); axes[1].set_ylim(0, 3.3); axes[1].set_ylabel("Passing SAE seeds (of 3)"); axes[1].set_title("Hierarchical robustness profile"); axes[1].legend(frameon=False)
    matrix = overlap.groupby(["target_a", "target_b"]).jaccard.mean().unstack().reindex(index=order, columns=order)
    image = axes[2].imshow(matrix, vmin=0, vmax=1, cmap="magma")
    axes[2].set_xticks(range(len(order)), order, rotation=90, fontsize=7); axes[2].set_yticks(range(len(order)), order, fontsize=7)
    axes[2].set_title("Mean top-5 atom Jaccard across seeds"); fig.colorbar(image, ax=axes[2], fraction=.046)
    fig.savefig(summary / "steering_benchmark_profile.png", dpi=220); fig.savefig(summary / "steering_benchmark_profile.pdf")

    novelty = cells[cells.analysis_role.eq("main")]
    by_target = cells.groupby(["target", "analysis_role"], as_index=False).agg(
        tier1_pass=("tier1_sparse_attribution", "sum"), tier2_pass=("tier2_selective_steering", "sum"),
        tier3_pass=("tier3_behavior_changing", "sum"), ste_mean=("ste", "mean"), otd_mean=("otd_mean", "mean"),
        wrong_ste_mean=("max_any_wrong_ste", "mean"), readout_quality=("readout_quality", "mean"))
    lines = ["# SAE Steering Benchmark: Detailed Results", "", "## Executive summary", "",
             f"Across {len(novelty)} main target-seed cells, Tier 1 passed {int(novelty.tier1_sparse_attribution.sum())}, Tier 2 passed {int(novelty.tier2_selective_steering.sum())}, and Tier 3 passed {int(novelty.tier3_behavior_changing.sum())}.",
             "Tier 2 is the decisive selective-steering gate: it requires superiority to matched random clamps and to the strongest other-target top-5 intervention.", "",
             "## Per-target profile", "", by_target.to_csv(index=False), "",
             "## Interpretation", "",
             "- A Tier 1-only result means the selected sparse code affects its target more than matched random codes, but does not establish selective control.",
             "- A Tier 2 result supports selective code-space steering of frozen readouts. It remains a representation intervention, not a generated ECG.",
             "- Tier 3 reports held-out readout behavior at a fixed operating point (binary) or held-out R2 damage (continuous).",
             "- AFIB is a definition-proximal positive control; age, sex, and baseline drift are nuisance controls and are excluded from the novelty denominator.",
             "- Tier 4 waveform causality remains false for every cell.", "",
             "## Files", "",
             "- `steering_cells_enriched.csv`: complete seed-level statistics and behavior labels.",
             "- `topk_intervention_sensitivity.csv`: top-1/top-5/top-10 behavior sensitivity.",
             "- `shared_atom_audit.csv`: seed-level pairwise atom overlap.",
             "- `steering_benchmark_profile.png/.pdf`: paper-facing profile figure."]
    (summary / "steering_benchmark_detailed_report.md").write_text("\n".join(lines) + "\n")
    print(by_target.to_string(index=False))


if __name__ == "__main__":
    main()
