#!/usr/bin/env python3
"""Post-hoc cleanup audit for v1 interpretability benchmark results.

This script does not rerun model inference or overwrite primary result files. It
builds stricter probe gates, canonical task summaries, concept correlation
audits, and a residual-probe rerun manifest from existing artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results" / "analysis"
COMPARISON = ANALYSIS / "model_comparison"
PROBE_ROOT = ROOT / "results" / "probe"
MANIFEST = ROOT / "results" / "manifest"
OUT = COMPARISON / "cleanup_audit"

VAL_R2_MIN = 0.04
CONTROL_MARGIN_MIN = 0.01
PEAK_GAP_MIN = 0.002
DOMINANCE_MAX = 0.90
CONFIRM_DELTA_MIN = 0.005

MODEL_SUFFIX_TO_NAME = {
    "csfm_cu118_commons": "CSFM",
    "ecg_fm_cu118_commons": "ECG-FM",
    "ecg_jepa_cu118_commons": "ECG-JEPA",
    "st_mem_cu118_commons": "ST-MEM",
    "hubert_ecg_cu118_commons": "HuBERT-ECG",
    "cardiac_fm_cu118_commons": "CARDIAC-FM",
}

TASK_CANONICAL = {
    "ptbxl_norm": "ptbxl_norm",
    "ptbxl_mi": "ptbxl_mi",
    "ptbxl_sttc": "ptbxl_sttc",
    "ptbxl_cd": "ptbxl_cd",
    "ptbxl_hyp": "ptbxl_hyp",
    "mi_ischemia": "mi_ischemia",
    "bbb_conduction": "ptbxl_cd",
    "hypertrophy": "ptbxl_hyp",
    "af_rhythm": "af_rhythm",
}


def bh_fdr(p_values: pd.Series) -> pd.Series:
    p = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return pd.Series(out, index=p_values.index)
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return pd.Series(out, index=p_values.index)


def strict_probe_for_suffix(suffix: str, concepts: pd.DataFrame) -> pd.DataFrame:
    scores_path = PROBE_ROOT / suffix / "probe_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(scores_path)
    scores = pd.read_csv(scores_path)
    scores["val_r2"] = pd.to_numeric(scores["val_r2"], errors="coerce")
    scores["test_r2"] = pd.to_numeric(scores["test_r2"], errors="coerce")
    scores["val_r2_shuffled"] = pd.to_numeric(scores["val_r2_shuffled"], errors="coerce")
    scores["val_r2_gaussian"] = pd.to_numeric(scores["val_r2_gaussian"], errors="coerce")

    rows = []
    for concept_id, part in scores.groupby("concept_id", sort=False):
        part = part.sort_values("val_r2", ascending=False).reset_index(drop=True)
        peak = part.iloc[0]
        second_val = float(part.iloc[1]["val_r2"]) if len(part) > 1 else np.nan
        peak_val = float(peak["val_r2"])
        shuffled_margin = peak_val - float(peak["val_r2_shuffled"])
        gaussian_margin = peak_val - float(peak["val_r2_gaussian"])
        peak_gap = peak_val - second_val if np.isfinite(second_val) else np.nan
        dominance = np.nan
        strict = (
            peak_val >= VAL_R2_MIN
            and shuffled_margin >= CONTROL_MARGIN_MIN
            and gaussian_margin >= CONTROL_MARGIN_MIN
            and (not np.isfinite(peak_gap) or peak_gap >= PEAK_GAP_MIN)
            and (not np.isfinite(dominance) or dominance <= DOMINANCE_MAX)
        )
        rows.append(
            {
                "model": MODEL_SUFFIX_TO_NAME.get(suffix, suffix),
                "suffix": suffix,
                "concept_id": concept_id,
                "peak_feature": peak["feature"],
                "peak_val_r2": peak_val,
                "second_best_val_r2": second_val,
                "peak_gap": peak_gap,
                "test_r2_at_peak": float(peak["test_r2"]),
                "val_r2_shuffled_at_peak": float(peak["val_r2_shuffled"]),
                "val_r2_gaussian_at_peak": float(peak["val_r2_gaussian"]),
                "shuffled_margin": shuffled_margin,
                "gaussian_margin": gaussian_margin,
                "dominance": dominance,
                "dominance_status": "not_applicable_scalar_concept",
                "strict_encoded": strict,
                "fail_val_r2": peak_val < VAL_R2_MIN,
                "fail_shuffled_margin": shuffled_margin < CONTROL_MARGIN_MIN,
                "fail_gaussian_margin": gaussian_margin < CONTROL_MARGIN_MIN,
                "fail_peak_gap": bool(np.isfinite(peak_gap) and peak_gap < PEAK_GAP_MIN),
                "fail_dominance": False,
            }
        )
    df = pd.DataFrame(rows)
    return df.merge(concepts[["concept_id", "family", "display_name"]], on="concept_id", how="left")


def make_strict_probe(concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [strict_probe_for_suffix(suffix, concepts) for suffix in MODEL_SUFFIX_TO_NAME]
    strict = pd.concat(frames, ignore_index=True)
    summary = (
        strict.groupby("model", sort=False)
        .agg(
            concept_count=("concept_id", "nunique"),
            strict_encoded_count=("strict_encoded", "sum"),
            mean_peak_val_r2=("peak_val_r2", "mean"),
            min_peak_gap=("peak_gap", "min"),
            fail_peak_gap_count=("fail_peak_gap", "sum"),
            fail_val_r2_count=("fail_val_r2", "sum"),
            fail_shuffled_margin_count=("fail_shuffled_margin", "sum"),
            fail_gaussian_margin_count=("fail_gaussian_margin", "sum"),
        )
        .reset_index()
    )
    summary["strict_encoded_frac"] = (
        summary["strict_encoded_count"].astype(int).astype(str)
        + "/"
        + summary["concept_count"].astype(int).astype(str)
    )
    return strict, summary


def task_overlap_audit() -> pd.DataFrame:
    tasks_path = MANIFEST / "tasks_matrix.csv"
    tasks = pd.read_csv(tasks_path)
    task_cols = [c for c in tasks.columns if c != "ecg_id"]
    rows = []
    for i, a in enumerate(task_cols):
        av = pd.to_numeric(tasks[a], errors="coerce").fillna(0).astype(int).to_numpy()
        for b in task_cols[i + 1 :]:
            bv = pd.to_numeric(tasks[b], errors="coerce").fillna(0).astype(int).to_numpy()
            identical = bool(np.array_equal(av, bv))
            a_pos = av == 1
            b_pos = bv == 1
            inter = int(np.sum(a_pos & b_pos))
            a_n = int(np.sum(a_pos))
            b_n = int(np.sum(b_pos))
            rows.append(
                {
                    "task_a": a,
                    "task_b": b,
                    "canonical_a": TASK_CANONICAL.get(a, a),
                    "canonical_b": TASK_CANONICAL.get(b, b),
                    "n_a": a_n,
                    "n_b": b_n,
                    "n_intersection": inter,
                    "jaccard": inter / max(int(np.sum(a_pos | b_pos)), 1),
                    "a_subset_b": bool(a_n > 0 and inter == a_n),
                    "b_subset_a": bool(b_n > 0 and inter == b_n),
                    "identical": identical,
                    "canonical_duplicate": TASK_CANONICAL.get(a, a) == TASK_CANONICAL.get(b, b),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["identical", "canonical_duplicate", "jaccard"], ascending=[False, False, False]
    )


def canonical_continuation(erasure: pd.DataFrame, strict: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strict_keys = set(
        strict.loc[strict["strict_encoded"], ["model", "concept_id"]].itertuples(index=False, name=None)
    )
    df = erasure.copy()
    json_fields = [
        "base_auroc",
        "erased_auroc",
        "random_auroc",
        "delta_auroc",
        "delta_auroc_minus_random",
        "delta_auroc_minus_random_ci_low",
        "delta_auroc_minus_random_ci_high",
        "delta_auroc_minus_random_p_one_sided",
        "base_auprc",
        "erased_auprc",
        "random_auprc",
        "delta_auprc",
        "bootstrap_samples",
        "bootstrap_valid_samples",
        "original_probe_r2",
        "residual_probe_r2",
        "residual_probe_r2_drop",
        "residual_probe_threshold",
        "eraser_effective_flag",
        "eraser_method",
        "eraser_rank",
        "leace_ridge",
        "leace_ridge_abs",
    ]
    for idx, row in df.iterrows():
        path = ROOT / str(row["out_json"])
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("eraser_method") != "leace":
            continue
        for field in json_fields:
            if field in data:
                df.loc[idx, field] = data[field]
    df["canonical_task_id"] = df["task_id"].map(lambda x: TASK_CANONICAL.get(x, x))
    df["strict_encoded_gate"] = [
        (row.model, row.concept_id) in strict_keys for row in df.itertuples(index=False)
    ]

    # Collapse exact canonical duplicates before FDR. Keep the strongest adjusted
    # effect per model/concept/canonical-task/layer.
    sort_cols = ["delta_auroc_minus_random", "delta_auroc_minus_random_ci_low"]
    dedup = (
        df.sort_values(sort_cols, ascending=[False, False])
        .drop_duplicates(["model", "concept_id", "canonical_task_id", "layer"], keep="first")
        .copy()
    )
    dedup["canonical_bh_q"] = np.nan
    for (_model, task), idx in dedup.groupby(["model", "canonical_task_id"]).groups.items():
        dedup.loc[idx, "canonical_bh_q"] = bh_fdr(dedup.loc[idx, "delta_auroc_minus_random_p_one_sided"])
    dedup["canonical_confirmed"] = (
        dedup["strict_encoded_gate"]
        & (dedup.get("eraser_method", "") == "leace")
        & (dedup.get("eraser_effective_flag", False).fillna(False).astype(bool))
        & (dedup["delta_auroc_minus_random"] > CONFIRM_DELTA_MIN)
        & (dedup["delta_auroc_minus_random_ci_low"] > 0)
        & (dedup["canonical_bh_q"] < 0.05)
    )
    summary = (
        dedup.groupby("model", sort=False)
        .agg(
            tested_after_task_dedup=("concept_id", "size"),
            strict_gated_tested=("strict_encoded_gate", "sum"),
            canonical_confirmed_count=("canonical_confirmed", "sum"),
            max_canonical_adj_drop=("delta_auroc_minus_random", "max"),
        )
        .reset_index()
    )
    return dedup, summary


def concept_correlation(concepts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(MANIFEST / "concepts_matrix.csv")
    concept_ids = [c for c in matrix.columns if c != "ecg_id"]
    values = matrix[concept_ids].apply(pd.to_numeric, errors="coerce")
    corr = values.corr(method="spearman", min_periods=100)
    corr.index.name = "concept_id"
    corr_long = (
        corr.reset_index()
        .melt(id_vars="concept_id", var_name="concept_id_b", value_name="spearman_r")
        .rename(columns={"concept_id": "concept_id_a"})
    )
    corr_long = corr_long[corr_long["concept_id_a"] < corr_long["concept_id_b"]].copy()
    corr_long["abs_spearman_r"] = corr_long["spearman_r"].abs()
    family = concepts.set_index("concept_id")["family"].to_dict()
    corr_long["family_a"] = corr_long["concept_id_a"].map(family)
    corr_long["family_b"] = corr_long["concept_id_b"].map(family)
    corr_long["same_family"] = corr_long["family_a"] == corr_long["family_b"]
    high = corr_long.sort_values("abs_spearman_r", ascending=False)
    return corr, high


def residual_manifest(canonical: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "model_key",
        "suffix",
        "concept_id",
        "family",
        "task_id",
        "canonical_task_id",
        "feature",
        "layer",
        "out_json",
    ]
    manifest = canonical.loc[canonical["strict_encoded_gate"], cols].copy()
    residual_rows = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.out_json
        data = {}
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
        complete = (
            "residual_probe_r2" in data
            and "eraser_effective_flag" in data
            and data.get("eraser_method") == "leace"
        )
        residual_rows.append(
            {
                "eraser_method": data.get("eraser_method"),
                "original_probe_r2": data.get("original_probe_r2"),
                "residual_probe_r2": data.get("residual_probe_r2"),
                "residual_probe_r2_drop": data.get("residual_probe_r2_drop"),
                "residual_probe_threshold_value": data.get("residual_probe_threshold"),
                "eraser_effective_flag": data.get("eraser_effective_flag"),
                "residual_probe_status": "complete" if complete else "needs_rerun",
            }
        )
    manifest = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(residual_rows)], axis=1)
    manifest["residual_probe_threshold"] = "R_resid < max(0.02, 0.35*max(R_probe,0.04))"
    return manifest


def residual_summary(residual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in residual.groupby("model", sort=False):
        complete = part[part["residual_probe_status"] == "complete"]
        if len(complete):
            effective = complete["eraser_effective_flag"].fillna(False).astype(bool)
            effective_rate = float(effective.mean())
            effective_count = int(effective.sum())
            median_resid = pd.to_numeric(complete["residual_probe_r2"], errors="coerce").median()
        else:
            effective_rate = np.nan
            effective_count = 0
            median_resid = np.nan
        rows.append(
            {
                "model": model,
                "strict_gated_cells": len(part),
                "residual_complete_cells": len(complete),
                "eraser_effective_cells": effective_count,
                "eraser_effective_rate_complete": effective_rate,
                "median_residual_probe_r2": median_resid,
            }
        )
    return pd.DataFrame(rows)


def residual_commands(residual: pd.DataFrame) -> list[str]:
    commands = []
    for row in residual.itertuples(index=False):
        if row.residual_probe_status == "complete":
            continue
        commands.append(
            " ".join(
                [
                    "scripts/run_one_continuation_candidate.sh",
                    str(row.model_key),
                    str(row.suffix),
                    str(row.concept_id),
                    str(row.task_id),
                    str(int(row.layer)),
                ]
            )
        )
    return commands


def gate_update(canonical_summary: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    continuation_verified = {
        "CSFM": "yes",
        "ECG-JEPA": "yes_needs_residual_probe",
        "ECG-FM": "needs_residual_probe_and_continuation_audit",
        "ST-MEM": "needs_residual_probe_and_continuation_audit",
        "HuBERT-ECG": "needs_residual_probe_and_continuation_audit",
        "CARDIAC-FM": "needs_residual_probe_and_continuation_audit",
    }
    residual_by_model = residual_summary(residual).set_index("model")
    for row in canonical_summary.itertuples(index=False):
        verified = continuation_verified.get(row.model, "needs_review")
        residual_row = residual_by_model.loc[row.model] if row.model in residual_by_model.index else None
        complete = int(residual_row["residual_complete_cells"]) if residual_row is not None else 0
        total = int(residual_row["strict_gated_cells"]) if residual_row is not None else 0
        effective_rate = float(residual_row["eraser_effective_rate_complete"]) if residual_row is not None else np.nan
        if complete < total:
            main_status = "pending_residual_probe"
        elif verified.startswith("yes") and effective_rate >= 0.8:
            main_status = "main_causal_candidate"
        elif verified.startswith("yes"):
            main_status = "continuation_verified_but_eraser_ineffective"
        else:
            main_status = "extended_until_audit"
        rows.append(
            {
                "model": row.model,
                "continuation_audit_status": verified,
                "canonical_confirmed_count_pre_residual": int(row.canonical_confirmed_count),
                "residual_complete_cells": complete,
                "strict_gated_cells": total,
                "eraser_effective_rate_complete": effective_rate,
                "main_status_after_cleanup": main_status,
                "note": "residual-probe gate still required for causal claims",
            }
        )
    return pd.DataFrame(rows)


def report_text(
    strict_summary: pd.DataFrame,
    canonical_summary: pd.DataFrame,
    task_audit: pd.DataFrame,
    residual: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    identical = task_audit[task_audit["identical"] | task_audit["canonical_duplicate"]]
    resid_summary = residual_summary(residual)
    return "\n".join(
        [
            "# V1 Cleanup Audit",
            "",
            "## Fixed Criteria",
            "",
            f"- strict encoding: val R2 >= {VAL_R2_MIN}, shuffled margin >= {CONTROL_MARGIN_MIN}, "
            f"Gaussian margin >= {CONTROL_MARGIN_MIN}, peak gap >= {PEAK_GAP_MIN}.",
            "- dominance check is marked not applicable because the current registry uses scalar concepts.",
            "- canonical continuation confirmation reruns BH-FDR within model x canonical-task panels after task de-duplication.",
            "- residual probe values are not present in existing JSON outputs; rerun manifest is generated instead of fabricating them.",
            "",
            "## Strict Probe Summary",
            "",
            strict_summary.fillna("").to_markdown(index=False),
            "",
            "## Canonical Continuation Summary",
            "",
            canonical_summary.fillna("").to_markdown(index=False),
            "",
            "## Task Duplicate / Overlap Flags",
            "",
            identical.fillna("").to_markdown(index=False),
            "",
            "## Residual Probe Summary",
            "",
            resid_summary.fillna("").to_markdown(index=False),
            "",
            "## Gate Update",
            "",
            gate.fillna("").to_markdown(index=False),
            "",
        ]
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    concepts = pd.read_csv(ROOT / "configs" / "concepts.csv")
    erasure = pd.read_csv(COMPARISON / "continuation_erasure_summary.csv")

    strict, strict_summary = make_strict_probe(concepts)
    task_audit = task_overlap_audit()
    canonical, canonical_summary = canonical_continuation(erasure, strict)
    corr, corr_high = concept_correlation(concepts)
    residual = residual_manifest(canonical)
    resid_summary = residual_summary(residual)
    commands = residual_commands(residual)
    gate = gate_update(canonical_summary, residual)

    strict.to_csv(OUT / "probe_encoding_strict_by_concept.csv", index=False)
    strict_summary.to_csv(OUT / "probe_encoding_strict_summary.csv", index=False)
    task_audit.to_csv(OUT / "task_overlap_audit.csv", index=False)
    canonical.to_csv(OUT / "continuation_canonical_strict_fdr.csv", index=False)
    canonical_summary.to_csv(OUT / "continuation_canonical_strict_summary.csv", index=False)
    corr.to_csv(OUT / "concept_spearman_correlation_matrix.csv")
    corr_high.to_csv(OUT / "concept_spearman_high_pairs.csv", index=False)
    residual.to_csv(OUT / "residual_probe_rerun_manifest.csv", index=False)
    resid_summary.to_csv(OUT / "residual_probe_summary.csv", index=False)
    (OUT / "residual_probe_commands.txt").write_text(
        "\n".join(commands) + ("\n" if commands else ""),
        encoding="utf-8",
    )
    gate.to_csv(OUT / "model_gate_cleanup_update.csv", index=False)
    (OUT / "cleanup_audit_report.md").write_text(
        report_text(strict_summary, canonical_summary, task_audit, residual, gate),
        encoding="utf-8",
    )
    print(f"wrote cleanup audit to {OUT}")


if __name__ == "__main__":
    main()
