#!/usr/bin/env python
"""Build family-level paper artifacts for the expanded 55-target audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    return parser.parse_args()


def markdown(frame: pd.DataFrame) -> str:
    cols = frame.columns.tolist(); lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    args = parse_args(); base = args.base
    out = base / "summary"; cells = pd.read_csv(out / "multimodel_steering_cells.csv")
    profile = pd.read_csv(out / "multimodel_target_profile.csv"); gate = pd.read_csv(base / "model_target_gate.csv")
    main = cells[cells.analysis_role.eq("main") & cells.headline_eligible]
    family = main.groupby(["model", "family"], as_index=False).agg(
        cells=("target", "size"), targets=("target", "nunique"), tier1_pass=("tier1_sparse_attribution", "sum"),
        tier2_pass=("tier2_selective_steering", "sum"), tier3_pass=("tier3_behavior_changing", "sum"),
        ste_mean=("ste", "mean"), otd_mean=("otd_mean", "mean"), quality_warnings=("readout_quality_warning", "sum"))
    for tier in (1, 2, 3): family[f"tier{tier}_rate"] = family[f"tier{tier}_pass"] / family.cells
    family["otd_ste_ratio"] = family.otd_mean / np.maximum(family.ste_mean, 1e-8)
    robust = profile[(profile.analysis_role == "main") & profile.headline_eligible & (profile.tier2_pass == 3)].copy()
    robust_count = robust.groupby("target").model.nunique().rename("robust_models").sort_values(ascending=False)
    family.to_csv(out / "expanded_family_profile.csv", index=False)
    robust.to_csv(out / "expanded_robust_model_targets.csv", index=False)
    robust_count.to_csv(out / "expanded_target_consistency.csv")

    models = ["CSFM", "CARDIAC-FM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]
    families = sorted(family.family.unique())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    matrices = [("tier2_rate", "Tier 2 pass rate", "viridis", 0, 1),
                ("tier3_rate", "Tier 3 behavior-change rate", "magma", 0, 1),
                ("otd_ste_ratio", "Family off-target / target effect", "coolwarm", 0, 1.5)]
    for ax, (column, title, cmap, lo, hi) in zip(axes, matrices):
        matrix = family.pivot(index="model", columns="family", values=column).reindex(index=models, columns=families)
        image = ax.imshow(matrix, vmin=lo, vmax=hi, cmap=cmap)
        ax.set_xticks(range(len(families)), families, rotation=55, ha="right"); ax.set_yticks(range(len(models)), models)
        ax.set_title(title); fig.colorbar(image, ax=ax, fraction=.046)
    fig.savefig(out / "expanded_family_steering_profile.png", dpi=220); fig.savefig(out / "expanded_family_steering_profile.pdf")

    sensitivity = []; atoms = []
    for path in sorted((base / "models").glob("*/tasks/seed*/*/result.json")):
        result = json.loads(path.read_text())
        for topk in ("top1", "top5", "top10"):
            metric = result["metrics"][topk][result["target"]]
            sensitivity.append({"model": result["model"], "target": result["target"], "family": result["family"],
                "seed": result["seed"], "intervention": topk,
                "primary_behavior_effect": metric.get("sens_drop_at_95spec", metric.get("r2_drop", np.nan)),
                "auroc_drop": metric.get("auroc_drop", np.nan), "decision_flip_rate": metric.get("decision_flip_rate", np.nan),
                "brier_change": metric.get("brier_change", np.nan), "ece_change": metric.get("ece_change", np.nan),
                "mae_change": metric.get("mae_change", np.nan)})
        atoms.append({"model": result["model"], "target": result["target"], "family": result["family"], "seed": result["seed"],
                      "top5_atoms": " ".join(map(str, result["selected_atoms"]["top5"]))})
    pd.DataFrame(sensitivity).to_csv(out / "expanded_topk_sensitivity.csv", index=False)
    atoms = pd.DataFrame(atoms); atoms.to_csv(out / "expanded_selected_atoms.csv", index=False)
    overlap = []
    for (model, seed), group in atoms.groupby(["model", "seed"]):
        records = list(group.itertuples(index=False))
        for left in records:
            for right in records:
                aa = set(map(int, left.top5_atoms.split())); bb = set(map(int, right.top5_atoms.split()))
                overlap.append({"model": model, "seed": seed, "target_a": left.target, "target_b": right.target,
                                "family_a": left.family, "family_b": right.family,
                                "intersection": len(aa & bb), "jaccard": len(aa & bb) / len(aa | bb)})
    pd.DataFrame(overlap).to_csv(out / "expanded_shared_atom_audit.csv", index=False)

    qualified = main[~main.readout_quality_warning]
    consistent = robust_count.reset_index().head(20)
    lines = ["# Expanded Multimodel SAE Steering Benchmark", "", "## Frozen scope", "",
             f"- Candidate targets: {gate.target.nunique()}; eligible model-target cells: {int(gate.cell_eligible.sum())}/{len(gate)}.",
             f"- Completed headline main cells: {len(main)}; quality-qualified: {len(qualified)}.",
             f"- Tier 1/Tier 2/Tier 3: {int(main.tier1_sparse_attribution.sum())}/{int(main.tier2_selective_steering.sum())}/{int(main.tier3_behavior_changing.sum())}.",
             f"- Quality-qualified Tier 2: {int(qualified.tier2_selective_steering.sum())}/{len(qualified)}.", "",
             "## Most cross-model-consistent robust targets", "", markdown(consistent), "",
             "## Family profile", "", markdown(family), "",
             "## Claim discipline", "",
             "- Family-wise BH-FDR is applied within each model.",
             "- Tier 2 requires matched-random, WBI, and strongest wrong-target superiority.",
             "- Tier 3 without Tier 2 is nonselective behavior change.",
             "- Diagnostic-definition and redundancy controls are excluded from the main novelty denominator.",
             "- Tier 4 waveform causality is not claimed."]
    (out / "expanded_steering_report.md").write_text("\n".join(lines) + "\n")
    print(consistent.to_string(index=False))


if __name__ == "__main__": main()
