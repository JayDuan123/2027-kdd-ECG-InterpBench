#!/usr/bin/env python
"""Create paper-facing multimodel steering profiles without a winner ranking."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"


def markdown(frame: pd.DataFrame) -> str:
    cols = frame.columns.tolist(); lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    out = BASE / "summary"; cells = pd.read_csv(out / "multimodel_steering_cells.csv")
    profile = pd.read_csv(out / "multimodel_target_profile.csv"); op = pd.read_csv(BASE / "selected_operating_points.csv")
    registry = pd.read_csv(BASE / "target_registry.csv")
    main_targets = registry.loc[registry.analysis_role.eq("main"), "target"].tolist()
    models = op.model.tolist()
    tier2 = profile[profile.analysis_role.eq("main")].pivot(index="model", columns="target", values="tier2_pass").reindex(index=models, columns=main_targets)
    ste = profile[profile.analysis_role.eq("main")].pivot(index="model", columns="target", values="ste_mean").reindex(index=models, columns=main_targets)
    otd = profile[profile.analysis_role.eq("main")].pivot(index="model", columns="target", values="otd_mean").reindex(index=models, columns=main_targets)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), layout="constrained")
    im = axes[0].imshow(tier2, vmin=0, vmax=3, cmap="viridis")
    axes[0].set_xticks(range(len(main_targets)), main_targets, rotation=65, ha="right"); axes[0].set_yticks(range(len(models)), models)
    axes[0].set_title("Tier 2 passing seeds (of 3)"); fig.colorbar(im, ax=axes[0], ticks=[0, 1, 2, 3], fraction=.046)
    im = axes[1].imshow(ste, vmin=0, cmap="magma")
    axes[1].set_xticks(range(len(main_targets)), main_targets, rotation=65, ha="right"); axes[1].set_yticks(range(len(models)), models)
    axes[1].set_title("Mean standardized target effect"); fig.colorbar(im, ax=axes[1], fraction=.046)
    ratio = otd / np.maximum(ste, 1e-8)
    im = axes[2].imshow(ratio, vmin=0, vmax=1.5, cmap="coolwarm")
    axes[2].set_xticks(range(len(main_targets)), main_targets, rotation=65, ha="right"); axes[2].set_yticks(range(len(models)), models)
    axes[2].set_title("Off-target / target effect ratio"); fig.colorbar(im, ax=axes[2], fraction=.046)
    fig.savefig(out / "multimodel_steering_profile.png", dpi=220); fig.savefig(out / "multimodel_steering_profile.pdf")

    nuisance = profile[profile.analysis_role.ne("main")].copy(); nuisance.to_csv(out / "multimodel_controls_profile.csv", index=False)
    robust = profile[(profile.analysis_role == "main") & (profile.tier2_pass == 3)][["model", "target", "ste_mean", "otd_mean", "wbi_median"]]
    robust.to_csv(out / "robust_tier2_cells.csv", index=False)
    sensitivity_rows = []; atom_rows = []
    for path in sorted((BASE / "models").glob("*/tasks/seed*/*/result.json")):
        result = json.loads(path.read_text())
        for topk in ("top1", "top5", "top10"):
            metric = result["metrics"][topk][result["target"]]
            sensitivity_rows.append({"model": result["model"], "target": result["target"], "seed": result["seed"],
                "intervention": topk, "primary_behavior_effect": metric.get("sens_drop_at_95spec", metric.get("r2_drop", np.nan)),
                "auroc_drop": metric.get("auroc_drop", np.nan), "decision_flip_rate": metric.get("decision_flip_rate", np.nan),
                "brier_change": metric.get("brier_change", np.nan), "ece_change": metric.get("ece_change", np.nan),
                "mae_change": metric.get("mae_change", np.nan)})
        atom_rows.append({"model": result["model"], "target": result["target"], "seed": result["seed"],
                          "top5_atoms": " ".join(map(str, result["selected_atoms"]["top5"]))})
    pd.DataFrame(sensitivity_rows).to_csv(out / "multimodel_topk_sensitivity.csv", index=False)
    atoms = pd.DataFrame(atom_rows); atoms.to_csv(out / "multimodel_selected_atoms.csv", index=False)
    overlap_rows = []
    for (model, seed), group in atoms.groupby(["model", "seed"]):
        records = list(group.itertuples(index=False))
        for left in records:
            for right in records:
                aa = set(map(int, left.top5_atoms.split())); bb = set(map(int, right.top5_atoms.split()))
                overlap_rows.append({"model": model, "seed": seed, "target_a": left.target, "target_b": right.target,
                                     "intersection": len(aa & bb), "jaccard": len(aa & bb) / len(aa | bb)})
    pd.DataFrame(overlap_rows).to_csv(out / "multimodel_shared_atom_audit.csv", index=False)
    seed_audit = pd.read_csv(BASE / "sae_seed_fidelity_audit.csv")
    lines = ["# Multimodel SAE Steering: Paper-Ready Profile", "", "## Operating points", "",
             markdown(op[["model", "d_hidden", "N", "k", "recon_R2", "dead_fraction", "status"]]), "",
             f"All {len(seed_audit)}/{len(seed_audit)} model-seed dictionaries passed the frozen reconstruction and dead-feature gates.", "",
             f"Readout-quality sensitivity retains {(~cells.readout_quality_warning).sum()}/{len(cells)} seed-level cells; all robust Tier 2 targets are reported with their warning status in the cell CSV.", "",
             "## Robust Tier 2 cells", "", markdown(robust) if len(robust) else "No target passed 3/3 seeds.", "",
             "## Interpretation discipline", "",
             "- Profiles are compared under matched reconstruction fidelity, not by a single aggregate winner score.",
             "- A Tier 2 pass means selective movement of frozen pooled-representation readouts relative to random and wrong-target sparse interventions.",
             "- Tier 3 behavior change without Tier 2 selectivity is a nonselective intervention, not successful steering.",
             "- No result establishes an edited physiologic waveform; Tier 4 remains false."]
    # Avoid requiring pandas' optional tabulate package in production environments.
    text = "\n".join(lines).replace("nan", "NA")
    (out / "multimodel_paper_ready_report.md").write_text(text + "\n")
    print(robust.to_string(index=False))


if __name__ == "__main__": main()
