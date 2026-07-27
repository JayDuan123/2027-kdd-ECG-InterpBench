#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "multicohort" / "track_f_closure"
TRACK_F_ROOT = ROOT / "results" / "multicohort" / "track_f_full"
MULTICOHORT_ROOT = ROOT / "results" / "multicohort"


TASK_SPECS = [
    {
        "cohort": "mimic",
        "task": "af_rhythm_icd",
        "label_file": "mimic_icd",
        "label_key": "study_id",
        "concepts": ["rr_mean"],
        "task_scope": "primary_full_gate",
    },
    {
        "cohort": "mimic",
        "task": "mi_ischemia_icd",
        "label_file": "mimic_icd",
        "label_key": "study_id",
        "concepts": ["st_amp_global", "t_amp_global"],
        "task_scope": "primary_full_gate_track_f",
    },
    {
        "cohort": "mimic",
        "task": "qt_interval_icd",
        "label_file": "mimic_icd",
        "label_key": "study_id",
        "concepts": ["qt_like"],
        "task_scope": "sensitivity_partial_concept_gate",
    },
    {
        "cohort": "mimic",
        "task": "hypertrophy_icd",
        "label_file": "mimic_icd",
        "label_key": "study_id",
        "concepts": ["r_amp_global"],
        "task_scope": "sensitivity_partial_concept_gate",
    },
    {
        "cohort": "chapman",
        "task": "af_rhythm_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["rr_mean"],
        "task_scope": "primary_full_gate",
    },
    {
        "cohort": "chapman",
        "task": "st_t_abnormal_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["st_amp_global", "t_amp_global"],
        "task_scope": "primary_full_gate",
    },
    {
        "cohort": "chapman",
        "task": "qt_interval_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["qt_like"],
        "task_scope": "sensitivity_partial_concept_gate",
    },
    {
        "cohort": "cpsc",
        "task": "af_rhythm_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["rr_mean"],
        "task_scope": "primary_full_gate",
    },
    {
        "cohort": "ningbo",
        "task": "af_rhythm_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["rr_mean"],
        "task_scope": "primary_full_gate",
    },
    {
        "cohort": "ningbo",
        "task": "st_t_abnormal_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["st_amp_global", "t_amp_global"],
        "task_scope": "primary_full_gate",
    },
    {
        "cohort": "ningbo",
        "task": "qt_interval_native",
        "label_file": "challenge_native",
        "label_key": "record_id",
        "concepts": ["qt_like"],
        "task_scope": "sensitivity_partial_concept_gate",
    },
]


CONCEPT_FIELDS = {
    "rr_mean": "rr_mean_ms",
    "qt_like": "qt_like_ms",
    "r_amp_global": "r_amp_global_mv",
    "st_amp_global": "st_amp_global_mv",
    "t_amp_global": "t_amp_global_mv",
}

LABEL_COHORT = {
    "chapman": "Chapman-F",
    "cpsc": "CPSC-F",
    "ningbo": "Ningbo-F",
}


OUT_FIELDS = [
    "cohort",
    "task",
    "task_scope",
    "status",
    "concepts",
    "concept_gate_statuses",
    "n_total_joined",
    "n_train",
    "n_val",
    "n_test",
    "positive_train",
    "positive_val",
    "positive_test",
    "btrackf_val_auroc",
    "btrackf_val_auprc",
    "btrackf_test_auroc",
    "btrackf_test_auprc",
    "brand_test_auroc",
    "brand_test_auprc",
    "closure_gain_vs_brand_auroc",
    "split_unit",
    "quality_note",
]


def import_runtime():
    try:
        import numpy as np
        import pandas as pd
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Track F closure requires numpy, pandas, and scikit-learn.") from exc
    return np, pd, SimpleImputer, LogisticRegression, average_precision_score, roc_auc_score, make_pipeline, StandardScaler


def stable_split(value: str) -> str:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def read_quality(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["cohort"], row["concept"])] = row["full_extraction_status"]
    return out


def read_feasibility(path: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[(row["cohort"], row["task"])] = row["eligible_for_auroc"]
    return out


def finite_metric(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.8g}"


def make_model(LogisticRegression, make_pipeline, StandardScaler):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced", random_state=4311),
    )


def score_binary(y_true, y_score, roc_auc_score, average_precision_score) -> tuple[float, float]:
    if len(set(int(x) for x in y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_score)), float(average_precision_score(y_true, y_score))


def run_cell(
    spec: dict[str, object],
    concepts,
    labels,
    quality: dict[tuple[str, str], str],
    feasibility: dict[tuple[str, str], str],
    np,
    SimpleImputer,
    LogisticRegression,
    average_precision_score,
    roc_auc_score,
    make_pipeline,
    StandardScaler,
) -> dict[str, str]:
    cohort = str(spec["cohort"])
    task = str(spec["task"])
    concept_ids = list(spec["concepts"])
    gate_statuses = [quality.get((cohort, concept), "not_assessed") for concept in concept_ids]
    feasibility_key = "MIMIC-F" if cohort == "mimic" else LABEL_COHORT.get(cohort, f"{cohort.capitalize()}-F")
    if feasibility.get((feasibility_key, task), "true") == "false":
        return skipped(spec, gate_statuses, "task_failed_g4_feasibility")
    if any(status == "excluded_smoke_fail" for status in gate_statuses):
        return skipped(spec, gate_statuses, "concept_failed_g3_smoke_gate")

    fields = [CONCEPT_FIELDS[concept] for concept in concept_ids]
    cell = concepts[concepts["cohort"] == cohort].copy()
    if spec["label_file"] == "mimic_icd":
        label_subset = labels["mimic_icd"][["study_id", "subject_id", task]].copy()
        cell["study_id_or_record_key"] = cell["study_id_or_record_key"].astype(str)
        label_subset["study_id"] = label_subset["study_id"].astype(str)
        label_subset["subject_id"] = label_subset["subject_id"].astype(str)
        merged = cell.merge(
            label_subset,
            left_on="study_id_or_record_key",
            right_on="study_id",
            how="inner",
            suffixes=("", "_label"),
        )
        split_source = merged["subject_id"].fillna(merged["study_id_or_record_key"]).astype(str)
        split_unit = "patient_level_subject_id"
    else:
        label_subset = labels["challenge_native"]
        label_subset = label_subset[label_subset["cohort"] == LABEL_COHORT[cohort]][["record_id", task]].copy()
        cell["record_id"] = cell["record_id"].astype(str)
        label_subset["record_id"] = label_subset["record_id"].astype(str)
        merged = cell.merge(label_subset, on="record_id", how="inner")
        split_source = merged["record_id"].astype(str)
        split_unit = "record_level_native_challenge_no_patient_id"

    for field in fields:
        merged[field] = np.asarray(merged[field], dtype=float)
    merged[task] = np.asarray(merged[task], dtype=int)
    merged["split"] = [stable_split(value) for value in split_source]
    merged = merged.dropna(subset=[task])
    n_total = len(merged)
    if n_total == 0:
        return skipped(spec, gate_statuses, "no_joined_rows")

    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    test = merged[merged["split"] == "test"]
    if min(len(train), len(val), len(test)) == 0:
        return skipped(spec, gate_statuses, "empty_split_after_join")
    if min(train[task].sum(), val[task].sum(), test[task].sum()) == 0:
        return skipped(spec, gate_statuses, "positive_class_missing_in_split")
    if min((1 - train[task]).sum(), (1 - val[task]).sum(), (1 - test[task]).sum()) == 0:
        return skipped(spec, gate_statuses, "negative_class_missing_in_split")

    x_train = train[fields].to_numpy(dtype=float)
    x_val = val[fields].to_numpy(dtype=float)
    x_test = test[fields].to_numpy(dtype=float)
    y_train = train[task].to_numpy(dtype=int)
    y_val = val[task].to_numpy(dtype=int)
    y_test = test[task].to_numpy(dtype=int)

    model = make_model(LogisticRegression, make_pipeline, StandardScaler)
    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_val_imp = imputer.transform(x_val)
    x_test_imp = imputer.transform(x_test)
    model.fit(x_train_imp, y_train)
    val_score = model.predict_proba(x_val_imp)[:, 1]
    test_score = model.predict_proba(x_test_imp)[:, 1]
    val_auc, val_auprc = score_binary(y_val, val_score, roc_auc_score, average_precision_score)
    test_auc, test_auprc = score_binary(y_test, test_score, roc_auc_score, average_precision_score)

    rng = np.random.default_rng(4311)
    brand_train = rng.normal(size=x_train_imp.shape)
    brand_test = rng.normal(size=x_test_imp.shape)
    brand = make_model(LogisticRegression, make_pipeline, StandardScaler)
    brand.fit(brand_train, y_train)
    brand_score = brand.predict_proba(brand_test)[:, 1]
    brand_auc, brand_auprc = score_binary(y_test, brand_score, roc_auc_score, average_precision_score)

    return {
        "cohort": cohort,
        "task": task,
        "task_scope": str(spec["task_scope"]),
        "status": "ok",
        "concepts": "|".join(concept_ids),
        "concept_gate_statuses": "|".join(gate_statuses),
        "n_total_joined": str(n_total),
        "n_train": str(len(train)),
        "n_val": str(len(val)),
        "n_test": str(len(test)),
        "positive_train": str(int(y_train.sum())),
        "positive_val": str(int(y_val.sum())),
        "positive_test": str(int(y_test.sum())),
        "btrackf_val_auroc": finite_metric(val_auc),
        "btrackf_val_auprc": finite_metric(val_auprc),
        "btrackf_test_auroc": finite_metric(test_auc),
        "btrackf_test_auprc": finite_metric(test_auprc),
        "brand_test_auroc": finite_metric(brand_auc),
        "brand_test_auprc": finite_metric(brand_auprc),
        "closure_gain_vs_brand_auroc": finite_metric(test_auc - brand_auc),
        "split_unit": split_unit,
        "quality_note": "partial concept gate" if any(status.startswith("partial") for status in gate_statuses) else "full concept gate",
    }


def skipped(spec: dict[str, object], gate_statuses: list[str], reason: str) -> dict[str, str]:
    return {
        "cohort": str(spec["cohort"]),
        "task": str(spec["task"]),
        "task_scope": str(spec["task_scope"]),
        "status": reason,
        "concepts": "|".join(spec["concepts"]),
        "concept_gate_statuses": "|".join(gate_statuses),
        "n_total_joined": "",
        "n_train": "",
        "n_val": "",
        "n_test": "",
        "positive_train": "",
        "positive_val": "",
        "positive_test": "",
        "btrackf_val_auroc": "",
        "btrackf_val_auprc": "",
        "btrackf_test_auroc": "",
        "btrackf_test_auprc": "",
        "brand_test_auroc": "",
        "brand_test_auprc": "",
        "closure_gain_vs_brand_auroc": "",
        "split_unit": "",
        "quality_note": reason,
    }


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def render_report(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Track F Closure Transfer",
        "",
        "This report uses waveform-derived Track F concepts after G3 quality gating. It is a transparent closure/sanity analysis, not an external leaderboard.",
        "",
        "| Cohort | Task | Scope | Concepts | Status | Test AUROC | Brand AUROC | Gain |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['cohort']} | {row['task']} | {row['task_scope']} | {row['concepts']} | "
            f"{row['status']} | {row['btrackf_test_auroc']} | {row['brand_test_auroc']} | "
            f"{row['closure_gain_vs_brand_auroc']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "- Primary Track F claims use only concepts with `full_candidate` G3 status.",
            "- Rows marked `sensitivity_partial_concept_gate` are descriptive sensitivity checks.",
            "- `qrs_duration` and `pr_interval` are excluded because their PTB-XL extractor-vendor smoke gate failed.",
            "- Challenge cohort labels are native SNOMED labels and remain secondary/semi-independent relative to MIMIC ICD-linked labels.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track F waveform-derived closure transfer.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows-per-cohort", type=int, default=None, help="Optional smoke cap after cohort filtering.")
    args = parser.parse_args()

    (
        np,
        pd,
        SimpleImputer,
        LogisticRegression,
        average_precision_score,
        roc_auc_score,
        make_pipeline,
        StandardScaler,
    ) = import_runtime()
    usecols = [
        "cohort",
        "record_id",
        "subject_id",
        "study_id_or_record_key",
        *CONCEPT_FIELDS.values(),
    ]
    concepts = pd.read_csv(
        TRACK_F_ROOT / "waveform_concepts_by_record.csv",
        usecols=usecols,
        dtype={
            "cohort": str,
            "record_id": str,
            "subject_id": str,
            "study_id_or_record_key": str,
        },
        low_memory=False,
    )
    if args.max_rows_per_cohort is not None:
        concepts = concepts.groupby("cohort", group_keys=False).head(args.max_rows_per_cohort).copy()
    labels = {
        "mimic_icd": pd.read_csv(MULTICOHORT_ROOT / "mimic_icd_label_matrix.csv", dtype={"study_id": str, "subject_id": str}),
        "challenge_native": pd.read_csv(MULTICOHORT_ROOT / "challenge_native_label_matrix.csv", dtype={"record_id": str, "cohort": str}),
    }
    quality = read_quality(TRACK_F_ROOT / "waveform_concept_quality.csv")
    feasibility = read_feasibility(MULTICOHORT_ROOT / "external_task_feasibility.csv")
    rows = [
        run_cell(
            spec,
            concepts,
            labels,
            quality,
            feasibility,
            np,
            SimpleImputer,
            LogisticRegression,
            average_precision_score,
            roc_auc_score,
            make_pipeline,
            StandardScaler,
        )
        for spec in TASK_SPECS
    ]
    out_csv = args.out_dir / "closure_transfer_track_f.csv"
    write_csv_atomic(out_csv, rows)
    report_path = args.out_dir / "closure_transfer_track_f_report.md"
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(render_report(rows))
    os.replace(tmp, report_path)
    print(f"wrote: {out_csv}")
    print(f"wrote: {report_path}")
    for row in rows:
        print(row["cohort"], row["task"], row["status"], row["btrackf_test_auroc"], row["brand_test_auroc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
