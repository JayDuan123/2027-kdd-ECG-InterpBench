#!/usr/bin/env python
"""Reclassify SAE L0-clamp steering results with ratio-stability awareness.

The original strict pass rule required both a positive difference-based
ExcessSelectivity CI and a positive ratio-based WBIImprovement CI. The WBI ratio
is not interpretable when the concept or random target effect is non-positive,
so this post-hoc audit separates full corroborated passes from
difference-positive but ratio-inconclusive cells.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("results/sae_extension/six_model_sae_audit")
OUT = ROOT / "l0clamp_reclassified"

INPUTS = {
    "main_recon_0.90": ROOT / "l0clamp_summary" / "sae_l0clamp_combined_results.csv",
    "sensitivity_recon_0.95": ROOT
    / "l0clamp_sensitivity95_summary"
    / "sae_l0clamp_combined_results.csv",
}


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    excess_low = pd.to_numeric(df["excess_selectivity_ci_low"], errors="coerce")
    wbi_low = pd.to_numeric(df["wbi_improvement_ci_low"], errors="coerce")
    stable = _bool_series(df["wbi_ratio_stable"])

    df["difference_significant"] = excess_low > 0.0
    df["ratio_stable"] = stable
    df["ratio_corrob_positive"] = stable & (wbi_low > 0.0)
    df["full_selective_pass"] = (
        df["difference_significant"] & df["ratio_corrob_positive"]
    )

    status = pd.Series("not_selective_by_difference", index=df.index, dtype=object)
    status.loc[df["full_selective_pass"]] = "full_selective"
    status.loc[
        df["difference_significant"]
        & ~df["full_selective_pass"]
        & ~df["ratio_stable"]
    ] = "difference_positive_ratio_unstable"
    status.loc[
        df["difference_significant"]
        & ~df["full_selective_pass"]
        & df["ratio_stable"]
        & ~df["ratio_corrob_positive"]
    ] = "difference_positive_ratio_not_corrob"
    df["steering_reclass"] = status
    return df


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    out = df.copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
    lines = [
        "| " + " | ".join(out.columns) + " |",
        "| " + " | ".join(["---"] * len(out.columns)) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for analysis, path in INPUTS.items():
        df = pd.read_csv(path)
        df["analysis"] = analysis
        frames.append(classify(df))
    cells = pd.concat(frames, ignore_index=True, sort=False)

    model_summary = (
        cells.groupby(["analysis", "model"])
        .agg(
            cells=("model", "size"),
            full_selective=("full_selective_pass", "sum"),
            difference_significant=("difference_significant", "sum"),
            difference_ratio_unstable=(
                "steering_reclass",
                lambda s: int((s == "difference_positive_ratio_unstable").sum()),
            ),
            difference_ratio_not_corrob=(
                "steering_reclass",
                lambda s: int((s == "difference_positive_ratio_not_corrob").sum()),
            ),
            not_selective_by_difference=(
                "steering_reclass",
                lambda s: int((s == "not_selective_by_difference").sum()),
            ),
            ratio_stable=("ratio_stable", "sum"),
            median_excess=("excess_selectivity", "median"),
            min_excess_ci_low=("excess_selectivity_ci_low", "min"),
            median_wbi_improvement=("wbi_improvement", "median"),
        )
        .reset_index()
    )
    overall_summary = (
        cells.groupby("analysis")
        .agg(
            cells=("model", "size"),
            models=("model", "nunique"),
            full_selective=("full_selective_pass", "sum"),
            difference_significant=("difference_significant", "sum"),
            difference_ratio_unstable=(
                "steering_reclass",
                lambda s: int((s == "difference_positive_ratio_unstable").sum()),
            ),
            difference_ratio_not_corrob=(
                "steering_reclass",
                lambda s: int((s == "difference_positive_ratio_not_corrob").sum()),
            ),
            not_selective_by_difference=(
                "steering_reclass",
                lambda s: int((s == "not_selective_by_difference").sum()),
            ),
            ratio_stable=("ratio_stable", "sum"),
        )
        .reset_index()
    )
    diff_cells = cells[cells["difference_significant"]].copy()
    diff_by_model = (
        diff_cells.groupby(["analysis", "model"])
        .agg(
            difference_significant=("model", "size"),
            median_excess=("excess_selectivity", "median"),
            min_excess=("excess_selectivity", "min"),
            max_excess=("excess_selectivity", "max"),
        )
        .reset_index()
    )

    effect_rows = []
    for analysis, part in cells.groupby("analysis"):
        diff_part = part[part["difference_significant"]]
        effect_rows.append(
            {
                "analysis": analysis,
                "all_target_abs_median": float(part["target_effect"].abs().median()),
                "all_target_abs_min": float(part["target_effect"].abs().min()),
                "all_target_abs_max": float(part["target_effect"].abs().max()),
                "diff_positive_median_excess": float(diff_part["excess_selectivity"].median())
                if len(diff_part)
                else float("nan"),
                "diff_positive_min_excess": float(diff_part["excess_selectivity"].min())
                if len(diff_part)
                else float("nan"),
                "diff_positive_max_excess": float(diff_part["excess_selectivity"].max())
                if len(diff_part)
                else float("nan"),
                "wbi_ci_low_min": float(part["wbi_improvement_ci_low"].min()),
                "wbi_ci_low_median": float(part["wbi_improvement_ci_low"].median()),
                "wbi_ci_low_max": float(part["wbi_improvement_ci_low"].max()),
            }
        )
    effect_summary = pd.DataFrame(effect_rows)

    low_coupling = pd.read_csv(ROOT / "phase0_low_coupling_cells.csv")
    candidate_models = sorted(low_coupling["model"].unique())
    main_models = sorted(cells.loc[cells["analysis"] == "main_recon_0.90", "model"].unique())
    sens_models = sorted(
        cells.loc[cells["analysis"] == "sensitivity_recon_0.95", "model"].unique()
    )
    no_main = sorted(set(candidate_models) - set(main_models))
    no_sens = sorted(set(candidate_models) - set(sens_models))

    cells.to_csv(OUT / "sae_l0clamp_reclassified_cells.csv", index=False)
    model_summary.to_csv(OUT / "sae_l0clamp_reclassified_model_summary.csv", index=False)
    overall_summary.to_csv(OUT / "sae_l0clamp_reclassified_overall_summary.csv", index=False)
    diff_by_model.to_csv(OUT / "sae_l0clamp_difference_positive_by_model.csv", index=False)
    effect_summary.to_csv(OUT / "sae_l0clamp_effect_size_summary.csv", index=False)

    diff_cols = [
        "analysis",
        "model",
        "concept",
        "task",
        "excess_selectivity",
        "excess_selectivity_ci_low",
        "ratio_stable",
        "wbi_improvement",
        "wbi_improvement_ci_low",
        "steering_reclass",
    ]

    lines = [
        "# SAE L0-Clamp Reclassified Steering Audit",
        "",
        "## Classification Rule",
        "",
        "- `full_selective`: `ExcessSelectivity` patient-bootstrap CI lower bound > 0, WBI ratio is stable, and `WBIImprovement` CI lower bound > 0.",
        "- `difference_positive_ratio_unstable`: `ExcessSelectivity` CI lower bound > 0, but the WBI ratio is unstable because a target-effect denominator is non-positive.",
        "- `difference_positive_ratio_not_corrob`: `ExcessSelectivity` CI lower bound > 0 and WBI ratio is stable, but `WBIImprovement` CI lower bound is not > 0.",
        "- `not_selective_by_difference`: `ExcessSelectivity` CI lower bound is not > 0.",
        "",
        "WBI is treated as corroboration only when stable; unstable WBI is not used as a universal veto.",
        "",
        "## Overall Counts",
        "",
        md_table(overall_summary),
        "",
        "## Model Counts",
        "",
        md_table(model_summary),
        "",
        "## Difference-Positive Concentration",
        "",
        md_table(diff_by_model),
        "",
        "Difference-positive cells are not uniformly distributed across models: in the main 0.90 analysis, 5/6 are in CSFM and 1/6 is in CARDIAC-FM; in the 0.95 sensitivity analysis, 4/4 are in CSFM.",
        "",
        "## Effect Size and Ratio Fragility",
        "",
        md_table(effect_summary),
        "",
        "The main 0.90 difference-positive effects are statistically above random but small in magnitude (median ExcessSelectivity about 0.013; CSFM's model-level median is about 0.010). The 0.95 difference-positive effects are larger among the selected cells, but they remain uncorroborated by WBIImprovement and are concentrated entirely in CSFM.",
        "",
        "The WBI ratio is fragile in this regime: target effects are small, 8/27 main cells have non-positive concept or random target-effect denominators, and all model-level WBIImprovement CI lower bounds remain negative. This is consistent with the broader finding that SAE feature clamps do not create robust, selective target movement.",
        "",
        "## Difference-Positive Cells",
        "",
        md_table(diff_cells[diff_cols].sort_values(["analysis", "model", "concept", "task"])),
        "",
        "## Operating-Point Coverage",
        "",
        f"- Candidate models from low-coupling LEACE cells: {len(candidate_models)}/6 ({', '.join(candidate_models)}).",
        f"- Main 0.90 in-band steering denominator: {len(main_models)}/6 models ({', '.join(main_models)}).",
        f"- Sensitivity 0.95 in-band steering denominator: {len(sens_models)}/6 models ({', '.join(sens_models)}).",
        f"- No main 0.90 in-band cells: {', '.join(no_main) if no_main else 'none'}.",
        f"- No sensitivity 0.95 in-band cells: {', '.join(no_sens) if no_sens else 'none'}.",
        "",
        "## Reading",
        "",
        "The fully corroborated selective-steering count remains 0 in both analyses. However, the stable difference-based audit identifies difference-positive cells: 6/27 in the 0.90 main analysis and 4/24 in the 0.95 sensitivity analysis. These are not full SAE steering passes because the WBI ratio either is unstable or fails to corroborate the difference-based signal.",
        "",
        "The correct paper wording is therefore: no cell achieved fully corroborated selective steering, while a small subset showed difference-based steering signals that require ratio-stability qualification.",
    ]
    (OUT / "sae_l0clamp_reclassified_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(OUT / "sae_l0clamp_reclassified_report.md")


if __name__ == "__main__":
    main()
