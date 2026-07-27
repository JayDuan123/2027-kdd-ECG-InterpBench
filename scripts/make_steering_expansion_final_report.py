#!/usr/bin/env python
"""Consolidate expanded, fine-grained, joint, and dose steering results."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
V21 = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_1_finegrained"
OUT = V2 / "summary/final_expansion"


def markdown(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(V2 / "summary/multimodel_steering_cells.csv")
    profile = pd.read_csv(V2 / "summary/multimodel_target_profile.csv")
    registry = pd.read_csv(V2 / "target_registry.csv")
    model_profile = pd.read_csv(V2 / "summary/multimodel_model_profile.csv")
    main = cells[cells.analysis_role.eq("main") & cells.headline_eligible.astype(bool)]
    quality = main[~main.readout_quality_warning.astype(bool)]
    robust = profile[
        profile.analysis_role.eq("main")
        & profile.headline_eligible.astype(bool)
        & profile.tier2_pass.eq(3)
    ].merge(registry[["target", "target_type", "family"]], on="target", how="left")
    consistency = (
        robust.groupby(["target", "target_type", "family"], as_index=False)
        .agg(robust_models=("model", "nunique"))
        .sort_values(["robust_models", "target"], ascending=[False, True])
    )
    robust_kind = robust.groupby(["target_type", "family"], as_index=False).size()
    robust_kind = robust_kind.rename(columns={"size": "robust_model_target_pairs"})

    strict = pd.read_csv(
        V21 / "summary/full_wrong_target_audit/strict_multimodel_target_profile.csv"
    )
    strict_robust = strict[strict.strict_robust_3_of_3.astype(bool)][
        ["model", "target", "family", "strict_margin_mean"]
    ].copy()
    strict_robust["strict_margin_mean"] = strict_robust.strict_margin_mean.round(3)
    joint = pd.read_csv(V2 / "joint_steering/summary/joint_steering_profile.csv")
    joint_summary = joint.groupby(["group_type", "scheme"], as_index=False).agg(
        groups=("group_id", "size"),
        robust_groups=("robust_selective", "sum"),
        target_effect_mean=("target_effect_mean", "mean"),
        offtarget_damage_mean=("offtarget_damage_mean", "mean"),
        wbi_median=("wbi_median", "median"),
    )
    for column in ("target_effect_mean", "offtarget_damage_mean", "wbi_median"):
        joint_summary[column] = joint_summary[column].round(3)
    dose = pd.read_csv(V2 / "summary/dose_direction/dose_direction_profile.csv")
    dose_summary = dose.groupby(["direction", "dose"], as_index=False).agg(
        model_targets=("target", "size"),
        robust_selective=("robust_selective", "sum"),
        absolute_signed_change=(
            "signed_target_change_mean",
            lambda values: float(np.mean(np.abs(values))),
        ),
        behavior_excess_mean=("behavior_excess_mean", "mean"),
    )
    dose_summary[["absolute_signed_change", "behavior_excess_mean"]] = dose_summary[
        ["absolute_signed_change", "behavior_excess_mean"]
    ].round(3)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    fig.patch.set_facecolor("white")
    target_plot = consistency.sort_values("robust_models", ascending=True)
    axes[0].barh(target_plot.target, target_plot.robust_models, color="#2878B5")
    axes[0].set_xlabel("Models with robust Tier-2 steering")
    axes[0].set_xlim(0, 6.2)
    axes[0].set_title("Cross-model target consistency")
    for direction, group in dose_summary.groupby("direction"):
        label = "Neutralize" if direction == "neutralize" else "Enhance"
        axes[1].plot(group.dose, group.robust_selective, marker="o", linewidth=2, label=label)
    axes[1].set_xlabel("Intervention dose")
    axes[1].set_ylabel("Robust model-target pairs")
    axes[1].set_ylim(0, 32)
    axes[1].set_title("Dose-response selectivity")
    axes[1].legend(frameon=False)
    kind_plot = robust_kind.sort_values("robust_model_target_pairs", ascending=True)
    labels = [f"{kind}: {family}" for kind, family in zip(kind_plot.target_type, kind_plot.family)]
    colors = ["#D95F02" if kind == "binary" else "#1B9E77" for kind in kind_plot.target_type]
    axes[2].barh(labels, kind_plot.robust_model_target_pairs, color=colors)
    axes[2].set_xlabel("Robust model-target pairs")
    axes[2].set_title("Where sparse steering survives")
    fig.savefig(OUT / "steering_expansion_summary.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "steering_expansion_summary.pdf", facecolor="white")

    model_table = model_profile[
        [
            "model",
            "targets",
            "tier1_pass",
            "tier2_pass",
            "tier3_pass",
            "robust_tier2_targets",
            "quality_qualified_cells",
            "qualified_tier2_pass",
        ]
    ]
    lines = [
        "# Expanded Six-Model SAE Steering Benchmark: Final Report",
        "",
        "## Frozen primary denominator",
        "",
        f"- Main targets: {main.target.nunique()} across six models.",
        f"- Eligible model-target pairs: {main[['model', 'target']].drop_duplicates().shape[0]}; seed-level cells: {len(main)}.",
        f"- Tier-1 / Tier-2 / Tier-3 passes: {int(main.tier1_sparse_attribution.sum())} / {int(main.tier2_selective_steering.sum())} / {int(main.tier3_behavior_changing.sum())} of {len(main)}.",
        f"- Quality-qualified Tier-2: {int(quality.tier2_selective_steering.sum())}/{len(quality)}.",
        f"- Robust Tier-2 model-target pairs (3/3 seeds): {len(robust)}.",
        "",
        "## Model profiles (not a leaderboard)",
        "",
        markdown(model_table),
        "",
        "## Robust target consistency",
        "",
        markdown(consistency),
        "",
        "Thirty of 31 robust model-target pairs are binary diagnostic phenotypes. "
        "The only robust continuous measurement pair is ECG-JEPA with RR-IQR. "
        "Expanded task coverage therefore reveals sparse control of diagnostic readouts, not broad sparse control of ECG measurement concepts.",
        "",
        "## Fine-grained SCP extension with full-registry controls",
        "",
        "The six-target extension completed 108/108 seed-level cells. A restricted wrong-target set produced 14 robust pairs, "
        "but the full 61-target wrong-atom audit retained only four:",
        "",
        markdown(strict_robust),
        "",
        "CLBBB and CRBBB were rejected as subtype-specific findings because LBBB and RBBB atoms respectively matched or exceeded them. "
        "This is consistent with the explicit label-nesting audit.",
        "",
        "## Joint steering",
        "",
        markdown(joint_summary),
        "",
        f"All {int(joint.robust_selective.sum())}/{len(joint)} family/cross-family group-scheme profiles were selective in all three seeds. "
        "Atom unions were deduplicated, and Top-5 and Top-10 unions each used same-size frequency/magnitude-matched random groups.",
        "",
        "## Dose and direction",
        "",
        markdown(dose_summary),
        "",
        "Signed changes were negative for every neutralization row and positive for every enhancement row. "
        "Absolute selectivity is direction-symmetric under the frozen linear readout, while behavior effects are direction-dependent. "
        "At 25%, 50%, 75%, and 100% neutralization, 13, 22, 28, and 31 of 31 robust pairs remained robust, respectively.",
        "",
        "## Interpretation and claim boundary",
        "",
        "- Diagnostic rhythm, conduction, and ectopy readouts expose the clearest sparse steering structure.",
        "- Most continuous measurement concepts remain encoded/behavior-relevant but fail selective sparse steering.",
        "- Tier-3 behavior change without Tier-2 selectivity is not evidence of concept-specific control.",
        "- Joint steering establishes code-space multi-readout control, not waveform-level physiological validity.",
        "- No result is Tier-4; no edited ECG waveform or clinical treatment effect is claimed.",
        "",
        "## Primary artifacts",
        "",
        "- `../multimodel_steering_cells.csv`: 954 patient-bootstrap seed-level cells.",
        "- `../multimodel_target_profile.csv`: 318 model-target profiles.",
        "- `../expanded_steering_report.md`: family-level v2 report.",
        "- `../../joint_steering/summary/joint_steering_profile.csv`: joint Top-5/Top-10 results.",
        "- `../dose_direction/dose_direction_profile.csv`: dose/direction profiles.",
        "- `../../../steering_benchmark_multimodel_v2_1_finegrained/summary/finegrained_strict_report.md`: strict fine-grained extension.",
    ]
    report = OUT / "steering_expansion_final_report.md"
    report.write_text("\n".join(lines) + "\n")
    consistency.to_csv(OUT / "robust_target_consistency.csv", index=False)
    joint_summary.to_csv(OUT / "joint_summary.csv", index=False)
    dose_summary.to_csv(OUT / "dose_summary.csv", index=False)
    print(report)


if __name__ == "__main__":
    main()
