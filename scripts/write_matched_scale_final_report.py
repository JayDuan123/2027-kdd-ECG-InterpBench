#!/usr/bin/env python
"""Write the final matched-scale SAE audit report from verified outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/matched_scale_v1"
STEER = BASE / "steering/summary"


def markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    fidelity = pd.read_csv(BASE / "matched_scale_model_profile.csv")
    model = pd.read_csv(STEER / "multimodel_model_profile.csv")
    targets = pd.read_csv(STEER / "multimodel_target_profile.csv")
    registry = pd.read_csv(BASE / "steering/target_registry.csv")[["target", "family"]]
    targets = targets.merge(registry, on="target", how="left", validate="many_to_one")
    if targets.family.isna().any():
        raise RuntimeError("Missing target family after registry join")
    paired = pd.read_csv(STEER / "scale_comparison/paired_target_robustness.csv")
    merged = targets.merge(
        paired[["model", "target", "robust_recon_matched", "robust_scale_matched"]],
        on=["model", "target"], validate="one_to_one"
    )
    stable = merged[
        merged.analysis_role.eq("main")
        & merged.robust_recon_matched
        & merged.robust_scale_matched
    ].copy()
    stable = stable[[
        "model", "target", "family", "ste_mean", "otd_mean", "wbi_median",
        "sae_retention_mean", "quality_warning_seeds"
    ]].sort_values(["model", "target"])
    stable.to_csv(BASE / "matched_scale_stable_main_targets.csv", index=False)

    fidelity_table = fidelity[[
        "model", "recon_R2_mean", "recon_R2_min", "dead_fraction_max",
        "readout_retention_median", "fidelity_pass_seeds", "matched_scale_primary_eligible"
    ]].copy()
    model_table = model[[
        "model", "cells", "targets", "tier1_pass", "tier2_pass", "tier3_pass",
        "robust_tier2_targets", "quality_qualified_cells", "qualified_tier2_pass"
    ]].copy()
    stable_counts = stable.groupby("model").size().rename("stable_main_targets_across_protocols").reset_index()
    model_table = model_table.merge(stable_counts, on="model", how="left").fillna(
        {"stable_main_targets_across_protocols": 0}
    )
    model_table["stable_main_targets_across_protocols"] = model_table.stable_main_targets_across_protocols.astype(int)

    scale_only = merged[
        merged.analysis_role.eq("main") & merged.robust_scale_matched & ~merged.robust_recon_matched
    ][["model", "target"]]
    old_only = merged[
        merged.analysis_role.eq("main") & ~merged.robust_scale_matched & merged.robust_recon_matched
    ][["model", "target"]]
    transport = pd.read_csv(
        ROOT / "results/multicohort/pooled_sae_transport/pooled_transport_model_cohort_gate.csv"
    )
    transport_table = transport[[
        "model", "cohort", "external_recon_R2_mean", "external_recon_R2_min",
        "finite_activation_fraction_min", "dead_fraction_shift_max",
        "source_fidelity_eligible", "primary_transport_eligible", "strict_090_eligible"
    ]].copy()
    passed_transport = transport_table[transport_table.primary_transport_eligible]

    lines = [
        "# Matched-Scale SAE Benchmark Final Report",
        "",
        "## Frozen scale protocol",
        "",
        "All six ECG foundation models were trained with the same relative SAE scale:",
        "",
        "- expansion `E = N/d = 8`;",
        "- active-code budget `k/d = 1/8`;",
        "- active dictionary fraction `k/N = 1/64`;",
        "- BatchTopK training for 8,000 steps;",
        "- three seeds: 4311, 4312, and 4313;",
        "- train-only per-dimension activation normalization.",
        "",
        "Absolute `N` differs only when hidden width differs: `N=6144,k=96` for `d=768`, "
        "and `N=4096,k=64` for CARDIAC-FM (`d=512`). This is matched relative capacity, "
        "not an assertion that all encoders have identical intrinsic complexity.",
        "",
        "## Fidelity gate",
        "",
        markdown(fidelity_table),
        "",
        "Primary steering includes only CSFM, ECG-FM, and ECG-JEPA because all three seeds "
        "satisfy reconstruction R2 >= 0.90, dead fraction < 0.20, and median frozen-readout "
        "retention >= 0.95. CARDIAC-FM, HuBERT-ECG, and ST-MEM remain explicit sensitivity "
        "or quality-warning models rather than being silently removed.",
        "",
        "## Matched-scale steering",
        "",
        "The primary denominator contains 399 main seed-level cells (3 models, 45 unique main targets). "
        "Every cell uses the same top-5 population-centroid intervention, matched random atoms, "
        "2,000 patient-level bootstrap resamples, and model-wise BH-FDR.",
        "",
        markdown(model_table),
        "",
        "Across all main cells, Tier 1 / Tier 2 / Tier 3 pass counts are 373/80/330 of 399. "
        "The quality-qualified sensitivity contains 256 cells, of which 78 pass Tier 2.",
        "",
        "## Stable findings across SAE scale protocols",
        "",
        "Only targets robust in all three seeds under both the earlier reconstruction-matched "
        "protocol and the new scale-matched protocol are treated as operating-point-stable.",
        "",
        markdown(stable),
        "",
        f"Stable main model-target findings: {len(stable)}. Scale-matched-only findings "
        f"({', '.join(scale_only.model + ':' + scale_only.target) or 'none'}) and reconstruction-matched-only "
        f"findings ({', '.join(old_only.model + ':' + old_only.target) or 'none'}) are sensitivity results, "
        "not stable model properties.",
        "",
        "## External pooled-activation transport gate",
        "",
        "A deterministic 512-record smoke sample was evaluated for each of four external cohorts. "
        "Dead-feature shift is compared against a deterministic 512-record PTB-XL reference, so "
        "the comparison has matched sample size. All activations must be finite, all three seeds must "
        "reach external reconstruction R2 >= 0.85, dead-fraction shift must be <= 0.20, and the source "
        "model must pass the PTB fidelity gate.",
        "",
        markdown(transport_table),
        "",
        f"Primary transport passes: {len(passed_transport)}/24 model-cohort pairs: "
        f"{', '.join(passed_transport.model + ':' + passed_transport.cohort)}.",
        "",
        "This gate permits a downstream external steering analysis; it is not itself an external "
        "steering result. Chapman, CPSC, and Ningbo lack native patient identifiers and therefore "
        "support record-level sensitivity only. MIMIC supports patient-level resampling after linkage "
        "to the ICD label matrix.",
        "",
        "## Interpretation boundary",
        "",
        "The matched-scale arm supports a controlled comparison of sparse code-space steering. "
        "It does not establish waveform-level causality, feature monosemanticity, or a model leaderboard. "
        "Tier 2 means that a frozen-readout intervention is more selective than matched random and wrong-atom "
        "controls under the frozen statistical protocol. Tier 4 remains unclaimed.",
        "",
        "## Authoritative artifacts",
        "",
        "- `matched_scale_fidelity_audit.csv`: all 18 seed-level SAE fidelity checks.",
        "- `matched_scale_model_profile.csv`: six-model fidelity gate.",
        "- `steering/summary/multimodel_steering_cells.csv`: patient-bootstrap/FDR cell results.",
        "- `steering/summary/scale_comparison/paired_seed_cells.csv`: paired operating-point sensitivity.",
        "- `matched_scale_stable_main_targets.csv`: stable main target set.",
        "- `results/multicohort/pooled_sae_transport/`: matched-sample external transport gate.",
    ]
    (BASE / "matched_scale_final_report.md").write_text("\n".join(lines) + "\n")
    print(f"stable_main_targets={len(stable)}")
    print(stable.groupby("model").size().to_string())


if __name__ == "__main__":
    main()
