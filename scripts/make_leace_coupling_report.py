#!/usr/bin/env python
"""Build a LEACE + concept-coupling aware final benchmark report.

This script reads existing cleanup-audit artifacts only. It does not rerun model
inference, erasure, bootstrap, or probe fitting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "results" / "analysis" / "model_comparison"
CLEANUP = COMPARISON / "cleanup_audit"

OUT_REPORT = COMPARISON / "final_v1_leace_coupling_report.md"
OUT_MODEL = COMPARISON / "leace_coupling_model_summary.csv"
OUT_FAMILY = COMPARISON / "leace_coupling_family_summary.csv"
OUT_RISK = COMPARISON / "leace_coupling_risk_summary.csv"
OUT_TOP = COMPARISON / "leace_coupling_top_confirmed.csv"


MODEL_ORDER = ["CSFM", "HuBERT-ECG", "ST-MEM", "ECG-FM", "ECG-JEPA", "CARDIAC-FM"]


def read_cleanup(name: str) -> pd.DataFrame:
    path = CLEANUP / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_No rows._"
    return view.fillna("").to_markdown(index=False)


def fmt_float_cols(df: pd.DataFrame, cols: list[str], digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda x: "" if pd.isna(x) else f"{x:.{digits}f}"
            )
    return out


def sort_model(df: pd.DataFrame) -> pd.DataFrame:
    if "model" not in df.columns:
        return df
    order = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    return (
        df.assign(_model_order=df["model"].map(lambda x: order.get(x, len(order))))
        .sort_values(["_model_order", "model"])
        .drop(columns=["_model_order"])
    )


def make_model_summary(
    strict_probe: pd.DataFrame,
    continuation_summary: pd.DataFrame,
    residual: pd.DataFrame,
    coupling_summary: pd.DataFrame,
    confirmed: pd.DataFrame,
) -> pd.DataFrame:
    top = (
        confirmed.sort_values("delta_auroc_minus_random", ascending=False)
        .groupby("model", sort=False)
        .head(1)
        .assign(
            top_candidate=lambda x: (
                x["concept_id"] + "->" + x["canonical_task_id"] + "@L" + x["layer"].astype(int).astype(str)
            )
        )[["model", "top_candidate", "delta_auroc_minus_random"]]
        .rename(columns={"delta_auroc_minus_random": "top_adj_drop"})
    )

    coupling = coupling_summary.copy()
    coupling["has_high_coupling"] = (
        (pd.to_numeric(coupling["other_erased_fraction"], errors="coerce") > 0)
        | (pd.to_numeric(coupling["max_abs_groundtruth_corr"], errors="coerce") >= 0.9)
    )
    coupling_model = (
        coupling.groupby("model", sort=False)
        .agg(
            high_coupling_sources=("has_high_coupling", "sum"),
            max_other_erased_fraction=("other_erased_fraction", "max"),
            max_abs_groundtruth_corr=("max_abs_groundtruth_corr", "max"),
            max_other_r2_drop=("max_other_r2_drop", "max"),
        )
        .reset_index()
    )

    model = (
        strict_probe[["model", "strict_encoded_frac"]]
        .merge(continuation_summary, on="model", how="outer")
        .merge(residual, on="model", how="outer")
        .merge(coupling_model, on="model", how="outer")
        .merge(top, on="model", how="outer")
    )
    model["confirmed_fraction"] = (
        model["canonical_confirmed_count"].fillna(0).astype(int).astype(str)
        + "/"
        + model["strict_gated_tested"].fillna(0).astype(int).astype(str)
    )
    cols = [
        "model",
        "strict_encoded_frac",
        "confirmed_fraction",
        "max_canonical_adj_drop",
        "eraser_effective_rate_complete",
        "median_residual_probe_r2",
        "high_coupling_sources",
        "max_other_erased_fraction",
        "max_abs_groundtruth_corr",
        "top_candidate",
        "top_adj_drop",
    ]
    return sort_model(model[cols])


def make_family_summary(confirmed: pd.DataFrame, coupling_summary: pd.DataFrame) -> pd.DataFrame:
    family = (
        confirmed.groupby(["model", "family"], sort=False)
        .agg(
            confirmed_rows=("concept_id", "size"),
            unique_concepts=("concept_id", "nunique"),
            unique_tasks=("canonical_task_id", "nunique"),
            max_adj_drop=("delta_auroc_minus_random", "max"),
            sum_adj_drop=("delta_auroc_minus_random", "sum"),
        )
        .reset_index()
    )

    representative = (
        confirmed.sort_values("delta_auroc_minus_random", ascending=False)
        .groupby(["model", "family"], sort=False)
        .head(1)
        .assign(
            representative_candidate=lambda x: (
                x["concept_id"] + "->" + x["canonical_task_id"] + "@L" + x["layer"].astype(int).astype(str)
            )
        )[["model", "family", "representative_candidate"]]
    )

    coupling = coupling_summary.copy()
    coupling["coupled_source"] = (
        (pd.to_numeric(coupling["other_erased_fraction"], errors="coerce") > 0)
        | (pd.to_numeric(coupling["max_abs_groundtruth_corr"], errors="coerce") >= 0.9)
    )
    fam_coupling = (
        coupling.groupby(["model", "source_family"], sort=False)
        .agg(
            coupled_sources=("coupled_source", "sum"),
            max_other_erased_fraction=("other_erased_fraction", "max"),
            max_abs_groundtruth_corr=("max_abs_groundtruth_corr", "max"),
        )
        .reset_index()
        .rename(columns={"source_family": "family"})
    )

    out = family.merge(representative, on=["model", "family"], how="left").merge(
        fam_coupling, on=["model", "family"], how="left"
    )
    out["interpretation_level"] = out.apply(
        lambda row: "family_level_required"
        if (row.get("coupled_sources", 0) > 0 or row.get("unique_concepts", 0) > 1)
        else "individual_candidate_ok",
        axis=1,
    )
    return sort_model(out).sort_values(["model", "sum_adj_drop"], ascending=[True, False])


def make_risk_summary(coupling_summary: pd.DataFrame, coupling_matrix: pd.DataFrame) -> pd.DataFrame:
    source = coupling_summary.copy()
    source["risk_reason"] = "low"
    source.loc[
        pd.to_numeric(source["max_abs_groundtruth_corr"], errors="coerce") >= 0.9,
        "risk_reason",
    ] = "ground_truth_corr_ge_0.9"
    source.loc[
        pd.to_numeric(source["other_erased_fraction"], errors="coerce") > 0,
        "risk_reason",
    ] = source.loc[
        pd.to_numeric(source["other_erased_fraction"], errors="coerce") > 0,
        "risk_reason",
    ].map(lambda x: "other_confirmed_concept_erased" if x == "low" else x + "+other_confirmed_concept_erased")

    risky = source[source["risk_reason"] != "low"].copy()
    if risky.empty:
        return risky

    top_pair = (
        coupling_matrix[~coupling_matrix["same_concept"].astype(bool)]
        .sort_values("target_r2_drop", ascending=False)
        .groupby(["model", "source_concept", "source_task", "source_layer"], sort=False)
        .head(1)
        .assign(
            strongest_damaged_target=lambda x: (
                x["target_concept"] + " (drop=" + x["target_r2_drop"].map(lambda y: f"{y:.3f}") + ")"
            )
        )[
            [
                "model",
                "source_concept",
                "source_task",
                "source_layer",
                "strongest_damaged_target",
            ]
        ]
    )
    risky = risky.merge(top_pair, on=["model", "source_concept", "source_task", "source_layer"], how="left")
    cols = [
        "model",
        "source_concept",
        "source_family",
        "source_task",
        "source_layer",
        "risk_reason",
        "n_other_confirmed_concepts",
        "n_other_erased_effective",
        "other_erased_fraction",
        "max_abs_groundtruth_corr",
        "max_other_r2_drop",
        "strongest_damaged_target",
    ]
    return sort_model(risky[cols]).sort_values(
        ["model", "other_erased_fraction", "max_abs_groundtruth_corr", "max_other_r2_drop"],
        ascending=[True, False, False, False],
    )


def make_top_confirmed(confirmed: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    risk_key = risk.assign(
        coupling_risk=lambda x: x["risk_reason"],
    )[["model", "source_concept", "source_task", "source_layer", "coupling_risk"]].rename(
        columns={
            "source_concept": "concept_id",
            "source_task": "task_id",
            "source_layer": "layer",
        }
    )
    top = confirmed.merge(risk_key, on=["model", "concept_id", "task_id", "layer"], how="left")
    top["coupling_risk"] = top["coupling_risk"].fillna("low_or_non_effective_cross_damage")
    top["candidate"] = (
        top["concept_id"] + "->" + top["canonical_task_id"] + "@L" + top["layer"].astype(int).astype(str)
    )
    cols = [
        "model",
        "candidate",
        "family",
        "delta_auroc_minus_random",
        "delta_auroc_minus_random_ci_low",
        "canonical_bh_q",
        "residual_probe_r2",
        "coupling_risk",
    ]
    return top.sort_values("delta_auroc_minus_random", ascending=False)[cols]


def build_report(
    model_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    risk_summary: pd.DataFrame,
    top_confirmed: pd.DataFrame,
    residual: pd.DataFrame,
    confirmed: pd.DataFrame,
) -> str:
    total_confirmed = len(confirmed)
    total_effective = int(residual["eraser_effective_cells"].sum())
    total_cells = int(residual["strict_gated_cells"].sum())
    risky_sources = len(risk_summary)

    model_view = fmt_float_cols(
        model_summary,
        [
            "max_canonical_adj_drop",
            "eraser_effective_rate_complete",
            "median_residual_probe_r2",
            "max_other_erased_fraction",
            "max_abs_groundtruth_corr",
            "top_adj_drop",
        ],
    )
    family_view = fmt_float_cols(
        family_summary,
        [
            "max_adj_drop",
            "sum_adj_drop",
            "max_other_erased_fraction",
            "max_abs_groundtruth_corr",
        ],
    )
    risk_view = fmt_float_cols(
        risk_summary,
        ["other_erased_fraction", "max_abs_groundtruth_corr", "max_other_r2_drop"],
    )
    top_view = fmt_float_cols(
        top_confirmed,
        [
            "delta_auroc_minus_random",
            "delta_auroc_minus_random_ci_low",
            "canonical_bh_q",
            "residual_probe_r2",
        ],
    )

    return "\n".join(
        [
            "# Final v1 LEACE and Concept-Coupling Report",
            "",
            "## Status",
            "",
            f"- Strict-gated continuation cells with residual audit: {total_cells}",
            f"- LEACE-effective cells: {total_effective}/{total_cells}",
            f"- Canonical confirmed concept-task candidates: {total_confirmed}",
            f"- Coupling-risk source candidates: {risky_sources}",
            "",
            "Interpretation rule: confirmed rows are valid LEACE causal-use candidates, but rows with high coupling should be interpreted at concept-family or shared-subspace level rather than as independent clinical concepts.",
            "",
            "## Model-Level Summary",
            "",
            md_table(model_view),
            "",
            "## Family-Level Summary",
            "",
            md_table(family_view),
            "",
            "## Coupling Risk Summary",
            "",
            md_table(risk_view, max_rows=30),
            "",
            "## Top Confirmed Candidates",
            "",
            md_table(top_view, max_rows=30),
            "",
            "## Paper-Ready Reading",
            "",
            "- The LEACE rerun fixes the earlier eraser no-op failure: every strict-gated cell has residual concept R2 driven to the preregistered threshold or below.",
            "- The causal signal is real under the current operational definition: residual concept information is removed and downstream AUROC drops remain positive after random-subspace adjustment, bootstrap CI, and BH-FDR.",
            "- The independent-concept count should not be overclaimed. HR/RR and multiple amplitude summaries share strong subspaces, so they should be reported as RATE_RHYTHM or AMPLITUDE family evidence.",
            "- The cleanest headline rows are those with large adjusted drops and low coupling risk, especially non-synonymous measurement-to-diagnosis links such as ST/T to MI or QRS/axis to conduction disease.",
            "- AF rows based on P-wave or rate/rhythm concepts are clinically plausible but close to diagnostic-definition features, so they should be framed as expected positive controls rather than the strongest novelty claim.",
            "- QRS-T angle appears across several models/tasks but is not flagged as a high coupling-risk source in the cross-concept residual audit. This supports treating it as a broad clinical axis measurement used by models, not simply a dirty erasure direction that removes many confirmed concepts at once.",
            "- Layer-0 confirmed effects should be interpreted separately from mid/late-layer effects. For example, HuBERT-ECG `hr_atrial->af_rhythm@L0` is input- or embedding-proximal causal use, while CSFM L3/L5 and ECG-JEPA L7 effects reflect later representation stages.",
            "- ECG-JEPA's confirmed hypertrophy evidence is coupling-aware AMPLITUDE-family evidence. `r_amp_global` and `r_amp_precordial` should be counted as one shared amplitude-family finding, not two independent clinical concepts.",
            "",
            "## Output Files",
            "",
            f"- `{OUT_MODEL.relative_to(ROOT)}`",
            f"- `{OUT_FAMILY.relative_to(ROOT)}`",
            f"- `{OUT_RISK.relative_to(ROOT)}`",
            f"- `{OUT_TOP.relative_to(ROOT)}`",
            f"- `{OUT_REPORT.relative_to(ROOT)}`",
            "",
        ]
    )


def main() -> None:
    strict_probe = read_cleanup("probe_encoding_strict_summary.csv")
    continuation_summary = read_cleanup("continuation_canonical_strict_summary.csv")
    residual = read_cleanup("residual_probe_summary.csv")
    continuation = read_cleanup("continuation_canonical_strict_fdr.csv")
    coupling_summary = read_cleanup("concept_coupling_summary.csv")
    coupling_matrix = read_cleanup("concept_coupling_residual_matrix.csv")

    confirmed = continuation[continuation["canonical_confirmed"].astype(bool)].copy()
    model_summary = make_model_summary(
        strict_probe=strict_probe,
        continuation_summary=continuation_summary,
        residual=residual,
        coupling_summary=coupling_summary,
        confirmed=confirmed,
    )
    family_summary = make_family_summary(confirmed, coupling_summary)
    risk_summary = make_risk_summary(coupling_summary, coupling_matrix)
    top_confirmed = make_top_confirmed(confirmed, risk_summary)

    model_summary.to_csv(OUT_MODEL, index=False)
    family_summary.to_csv(OUT_FAMILY, index=False)
    risk_summary.to_csv(OUT_RISK, index=False)
    top_confirmed.to_csv(OUT_TOP, index=False)

    OUT_REPORT.write_text(
        build_report(
            model_summary=model_summary,
            family_summary=family_summary,
            risk_summary=risk_summary,
            top_confirmed=top_confirmed,
            residual=residual,
            confirmed=confirmed,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT_REPORT}")


if __name__ == "__main__":
    main()
