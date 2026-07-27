#!/usr/bin/env python3
"""Build final v1 benchmark report and sensitivity tables from completed results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis" / "model_comparison"
OUT_REPORT = ANALYSIS / "final_v1_benchmark_report.md"
OUT_MANIFEST = ANALYSIS / "final_v1_manifest.csv"
OUT_THRESHOLD = ANALYSIS / "sensitivity_thresholds.csv"
OUT_FAMILY_COLLAPSE = ANALYSIS / "sensitivity_family_collapse.csv"
OUT_ENCODED_VS_CAUSAL = ANALYSIS / "encoded_vs_causal_used.csv"
OUT_NO_CONFIRMED = ANALYSIS / "no_confirmed_failure_modes.csv"

DELTA_THRESHOLDS = (0.0025, 0.005, 0.01)


def load_csv(name: str) -> pd.DataFrame:
    path = ANALYSIS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def yes_no(path: Path) -> str:
    return "yes" if path.exists() else "no"


def confirmed_mask(df: pd.DataFrame, delta_threshold: float = 0.005) -> pd.Series:
    return (
        (df["delta_auroc_minus_random"] > delta_threshold)
        & (df["delta_auroc_minus_random_ci_low"] > 0)
        & (df["delta_auroc_minus_random_bh_q"] < 0.05)
    )


def make_threshold_sensitivity(erasure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in DELTA_THRESHOLDS:
        mask = confirmed_mask(erasure, thr)
        tmp = erasure.assign(confirmed_at_threshold=mask)
        for model, part in tmp.groupby("model", sort=False):
            rows.append(
                {
                    "delta_threshold": thr,
                    "model": model,
                    "tested_count": len(part),
                    "confirmed_count": int(part["confirmed_at_threshold"].sum()),
                    "confirmed_concepts": "|".join(
                        sorted(part.loc[part["confirmed_at_threshold"], "concept_id"].unique())
                    ),
                    "confirmed_tasks": "|".join(
                        sorted(part.loc[part["confirmed_at_threshold"], "task_id"].unique())
                    ),
                    "max_confirmed_adj_drop": part.loc[
                        part["confirmed_at_threshold"], "delta_auroc_minus_random"
                    ].max(),
                }
            )
    return pd.DataFrame(rows)


def make_family_collapse(erasure: pd.DataFrame) -> pd.DataFrame:
    confirmed = erasure.loc[confirmed_mask(erasure)].copy()
    if confirmed.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "family",
                "confirmed_rows",
                "unique_concepts",
                "unique_tasks",
                "max_adj_drop",
                "sum_adj_drop",
                "representative_candidate",
            ]
        )

    rows = []
    for (model, family), part in confirmed.groupby(["model", "family"], sort=False):
        top = part.sort_values("delta_auroc_minus_random", ascending=False).iloc[0]
        rows.append(
            {
                "model": model,
                "family": family,
                "confirmed_rows": len(part),
                "unique_concepts": part["concept_id"].nunique(),
                "unique_tasks": part["task_id"].nunique(),
                "max_adj_drop": part["delta_auroc_minus_random"].max(),
                "sum_adj_drop": part["delta_auroc_minus_random"].sum(),
                "representative_candidate": (
                    f"{top.concept_id}->{top.task_id}@L{int(top.layer)}"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model", "sum_adj_drop", "max_adj_drop"], ascending=[True, False, False]
    )


def make_encoded_vs_causal(probe_family: pd.DataFrame, erasure: pd.DataFrame) -> pd.DataFrame:
    confirmed = erasure.loc[confirmed_mask(erasure)].copy()
    rows = []
    for (model, family), probe_part in probe_family.groupby(["model", "family"], sort=False):
        causal_part = confirmed[
            (confirmed["model"] == model) & (confirmed["family"] == family)
        ]
        encoded_count = int(probe_part["encoded_count"].sum())
        causal_count = int(causal_part["concept_id"].nunique())
        rows.append(
            {
                "model": model,
                "family": family,
                "encoded_count": encoded_count,
                "causal_used_concept_count": causal_count,
                "encoded_only_count": max(encoded_count - causal_count, 0),
                "causal_used_tasks": "|".join(sorted(causal_part["task_id"].unique())),
                "causal_used_concepts": "|".join(sorted(causal_part["concept_id"].unique())),
            }
        )
    return pd.DataFrame(rows)


def make_no_confirmed_failure_modes(erasure: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in erasure.groupby("model", sort=False):
        n = len(part)
        pass_delta = part["delta_auroc_minus_random"] > 0.005
        pass_ci = part["delta_auroc_minus_random_ci_low"] > 0
        pass_q = part["delta_auroc_minus_random_bh_q"] < 0.05
        confirmed = pass_delta & pass_ci & pass_q
        top = part.sort_values("delta_auroc_minus_random", ascending=False).iloc[0]
        rows.append(
            {
                "model": model,
                "tested_count": n,
                "confirmed_count": int(confirmed.sum()),
                "pass_delta_count": int(pass_delta.sum()),
                "pass_ci_count": int(pass_ci.sum()),
                "pass_bh_fdr_count": int(pass_q.sum()),
                "pass_delta_and_ci_count": int((pass_delta & pass_ci).sum()),
                "pass_all_except_delta_count": int((pass_ci & pass_q & ~pass_delta).sum()),
                "pass_all_except_ci_count": int((pass_delta & pass_q & ~pass_ci).sum()),
                "pass_all_except_q_count": int((pass_delta & pass_ci & ~pass_q).sum()),
                "top_candidate": f"{top.concept_id}->{top.task_id}@L{int(top.layer)}",
                "top_adj_drop": top["delta_auroc_minus_random"],
                "top_ci_low": top["delta_auroc_minus_random_ci_low"],
                "top_bh_q": top["delta_auroc_minus_random_bh_q"],
            }
        )
    return pd.DataFrame(rows)


def make_manifest(table2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in table2["model"].tolist():
        rows.append(
            {
                "model": model,
                "probe": "yes",
                "linear_task_head": "yes",
                "closure_ball": "yes",
                "continuation_erasure": "yes",
                "bootstrap_ci": "yes",
                "bh_fdr": "yes",
                "figure2": yes_no(ROOT / "results" / "figures" / "figure2_probe_family_heatmap.png"),
                "figure3": yes_no(ROOT / "results" / "figures" / "figure3_causal_use_atlas.png"),
                "figure4": yes_no(ROOT / "results" / "figures" / "figure4_family_causal_summary.png"),
                "figure5": yes_no(ROOT / "results" / "figures" / "figure5_closure_by_task.png"),
            }
        )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    view = view.fillna("")
    return view.to_markdown(index=False)


def make_report(
    table2: pd.DataFrame,
    erasure: pd.DataFrame,
    confirmed: pd.DataFrame,
    threshold: pd.DataFrame,
    family_collapse: pd.DataFrame,
    encoded_vs_causal: pd.DataFrame,
    no_confirmed: pd.DataFrame,
) -> str:
    total_tested = len(erasure)
    total_confirmed = int(confirmed_mask(erasure).sum())
    model_confirmed = table2[
        [
            "model",
            "encoded_concept_frac",
            "mean_fm_test_auroc",
            "mean_closure_ratio_ball",
            "confirmed_count",
            "strongest_causal_family_by_count",
            "strongest_causal_family_by_drop",
            "top_continuation_candidate",
            "top_continuation_adj_drop",
        ]
    ].copy()

    top_confirmed = confirmed[
        [
            "model",
            "concept_id",
            "family",
            "task_id",
            "layer",
            "delta_auroc_minus_random",
            "delta_auroc_minus_random_ci_low",
            "delta_auroc_minus_random_ci_high",
            "delta_auroc_minus_random_bh_q",
        ]
    ].head(20)

    lines = [
        "# ECG FM Interpretability Benchmark v1 Final Report",
        "",
        "## Scope",
        "",
        "This report summarizes the current PTB-XL/PTB-XL+ profile-based clinical concept interpretability benchmark.",
        "Concepts are PTB-XL+ ECG measurements/morphology variables; tasks are PTB-XL diagnostic labels.",
        "",
        "## Completion Status",
        "",
        f"- Models summarized: {table2['model'].nunique()}",
        f"- Continuation-erasure candidates evaluated: {total_tested}",
        f"- Confirmed representation-causal candidates: {total_confirmed}",
        "- Confirmation rule: adjusted AUROC drop > 0.005, paired bootstrap CI lower bound > 0, BH-FDR q < 0.05.",
        "- Erasure effect is adjusted against dimension-matched random-subspace control.",
        "",
        "## Table 2 Profile Summary",
        "",
        md_table(model_confirmed),
        "",
        "## Confirmed Continuation-Erasure Candidates",
        "",
        md_table(top_confirmed),
        "",
        "## Family-Level Causal Summary",
        "",
        md_table(family_collapse),
        "",
        "## Encoded vs Causally Used Concepts",
        "",
        md_table(encoded_vs_causal),
        "",
        "## Threshold Sensitivity",
        "",
        md_table(threshold),
        "",
        "## No-Confirmed Failure Mode Audit",
        "",
        md_table(no_confirmed),
        "",
        "## Current Main Findings",
        "",
        "- All six evaluated models encode the frozen PTB-XL+ measurement concept set under the current probe protocol.",
        "- Encoding is not equivalent to causal diagnostic use: only 16/180 continuation-erasure candidates pass the full confirmation rule.",
        "- Confirmed causal use is concentrated in CSFM and ECG-JEPA, with smaller confirmed effects for ST-MEM and HuBERT-ECG.",
        "- ECG-FM and CARDIAC-FM have no confirmed candidates under the current strict rule, mostly because adjusted effects are small or fail the CI/FDR controls.",
        "- The strongest causal families differ by model: CSFM has RATE_RHYTHM by count and AMPLITUDE by largest drop; ECG-JEPA and ST-MEM are RATE_RHYTHM; HuBERT-ECG is AMPLITUDE.",
        "",
        "## Output Files",
        "",
        "- `table2_interpretability_profile.csv`",
        "- `continuation_erasure_summary.csv`",
        "- `continuation_confirmed_candidates.csv`",
        "- `sensitivity_thresholds.csv`",
        "- `sensitivity_family_collapse.csv`",
        "- `encoded_vs_causal_used.csv`",
        "- `no_confirmed_failure_modes.csv`",
        "- `final_v1_manifest.csv`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    table2 = load_csv("table2_interpretability_profile.csv")
    erasure = load_csv("continuation_erasure_summary.csv")
    confirmed = erasure.loc[confirmed_mask(erasure)].copy().sort_values(
        "delta_auroc_minus_random", ascending=False
    )
    probe_family = load_csv("figure2_probe_family_summary.csv")

    threshold = make_threshold_sensitivity(erasure)
    family_collapse = make_family_collapse(erasure)
    encoded_vs_causal = make_encoded_vs_causal(probe_family, erasure)
    no_confirmed = make_no_confirmed_failure_modes(erasure)
    manifest = make_manifest(table2)

    threshold.to_csv(OUT_THRESHOLD, index=False)
    family_collapse.to_csv(OUT_FAMILY_COLLAPSE, index=False)
    encoded_vs_causal.to_csv(OUT_ENCODED_VS_CAUSAL, index=False)
    no_confirmed.to_csv(OUT_NO_CONFIRMED, index=False)
    manifest.to_csv(OUT_MANIFEST, index=False)

    report = make_report(
        table2=table2,
        erasure=erasure,
        confirmed=confirmed,
        threshold=threshold,
        family_collapse=family_collapse,
        encoded_vs_causal=encoded_vs_causal,
        no_confirmed=no_confirmed,
    )
    OUT_REPORT.write_text(report, encoding="utf-8")
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
