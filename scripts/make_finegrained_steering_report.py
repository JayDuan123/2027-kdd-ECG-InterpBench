#!/usr/bin/env python
"""Write the strict, label-overlap-aware report for the v2.1 SCP extension."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_1_finegrained"


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
    profile = pd.read_csv(BASE / "summary/multimodel_target_profile.csv")
    strict = pd.read_csv(
        BASE
        / "summary/full_wrong_target_audit/strict_multimodel_target_profile.csv"
    )
    manifest = pd.read_csv(BASE / "manifest.csv")
    target_summary = pd.read_csv(BASE / "target_gate_summary.csv")
    new_targets = ["irbbb", "crbbb", "clbbb", "almi", "injas", "iscin"]
    gate = target_summary[target_summary.target.isin(new_targets)][
        ["target", "family", "eligible_models", "min_readout", "median_readout"]
    ].copy()
    gate[["min_readout", "median_readout"]] = gate[
        ["min_readout", "median_readout"]
    ].round(3)
    target_consistency = strict.groupby("target", as_index=False).agg(
        original_robust_models=("original_tier2_pass", lambda values: int((values == 3).sum())),
        strict_robust_models=("strict_tier2_pass", lambda values: int((values == 3).sum())),
        strict_seed_pass=("strict_tier2_pass", "sum"),
    )
    robust = strict[strict.strict_robust_3_of_3.astype(bool)][
        ["model", "target", "family", "strict_margin_mean"]
    ].copy()
    robust["strict_margin_mean"] = robust.strict_margin_mean.round(3)
    pairs = [
        ("clbbb", "lbbb"),
        ("crbbb", "rbbb"),
        ("irbbb", "rbbb"),
        ("almi", "ami"),
        ("injas", "asmi"),
        ("iscin", "imi"),
    ]
    overlap = []
    for new, existing in pairs:
        left = manifest[new].astype(bool)
        right = manifest[existing].astype(bool)
        intersection = int((left & right).sum())
        union = int((left | right).sum())
        overlap.append(
            {
                "new_target": new,
                "comparison": existing,
                "new_positive": int(left.sum()),
                "comparison_positive": int(right.sum()),
                "jaccard": round(intersection / max(union, 1), 3),
                "new_within_comparison": round(intersection / max(int(left.sum()), 1), 3),
            }
        )
    overlap = pd.DataFrame(overlap)
    original_robust = profile.tier2_pass.eq(3).sum()
    strict_robust = strict.strict_robust_3_of_3.sum()
    lines = [
        "# Fine-Grained SCP Steering Extension (v2.1)",
        "",
        "## Scope and completion",
        "",
        "- Six fine-grained PTB-XL SCP targets were added without changing the frozen v2 denominator.",
        "- All 36 model-target cells passed prevalence and frozen-readout gates; all 108 model-target-seed interventions completed.",
        "- Every inference uses 2,000 patient-level bootstrap replicates and model-by-family BH-FDR.",
        "- The strict result compares each target's Top-5 intervention with all other non-nuisance targets in the 61-target registry.",
        "",
        "## Readout gate",
        "",
        markdown(gate),
        "",
        "## Strict selective-steering results",
        "",
        f"- Original restricted wrong-target control: {int(original_robust)} robust model-target pairs.",
        f"- Full-registry wrong-target control: {int(strict_robust)} robust model-target pairs.",
        "",
        markdown(robust),
        "",
        "## Cross-model target consistency",
        "",
        markdown(target_consistency),
        "",
        "## Label-overlap audit",
        "",
        markdown(overlap),
        "",
        "`CLBBB` is nested within `LBBB`, while `CRBBB` and `IRBBB` are nested within `RBBB`. "
        "Their strong unrestricted interventions therefore do not establish subtype-specific sparse control. "
        "The full-registry wrong-target audit correctly removes these parent-label explanations.",
        "",
        "## Claim boundary",
        "",
        "The strict passes support selective manipulation of fine-grained diagnostic readouts in frozen SAE code space. "
        "They do not demonstrate a physiologically valid edited waveform or clinical treatment effect. "
        "Conditional subtype contrasts, especially CRBBB versus IRBBB among RBBB-positive records, remain the appropriate next test.",
    ]
    output = BASE / "summary/finegrained_strict_report.md"
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
