#!/usr/bin/env python
"""Selection-bias and inverse-probability sensitivity for waveform edits."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_common import bh  # noqa: E402
from scripts.benchmark_extension_v2_common import (  # noqa: E402
    V1,
    V2,
    interval_and_p,
    stable_seed,
    write_json,
)


OUT = V2 / "waveform_failure_bias"
WORKERS = V1 / "waveform_interventions" / "workers"
WAVEFORM_MANIFEST = ROOT / "results" / "multicohort" / "track_f_full" / "waveform_concepts_by_record.csv"
LABELS = ROOT / "results" / "multicohort" / "challenge_native_label_matrix.csv"
TARGETS = {
    "rr_irregularity": "af_rhythm_native",
    "qrs_duration": "bbb_conduction_native",
    "qt_interval": "qt_interval_native",
}
COHORT_NAMES = {"chapman_f": "Chapman-F", "ningbo_f": "Ningbo-F"}
COVARIATES = (
    "age",
    "male",
    "target_label",
    "rr_mean_ms",
    "qrs_duration_ms",
    "qt_like_ms",
    "r_amp_global_mv",
    "st_amp_global_mv",
    "t_amp_global_mv",
)
IPW_METRICS = ("measurement_delta", "target_head_delta", "sae_top5_delta_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def record_sets(model: str, cohort: str, phenotype: str) -> tuple[set[str], set[str]]:
    root = WORKERS / model / cohort / phenotype
    metrics = pd.read_csv(root / "per_variant_metrics.csv", usecols=["record_id", "variant"])
    success = set(metrics.loc[metrics.variant.eq("identity"), "record_id"].astype(str))
    failures = set(pd.read_csv(root / "failures.csv")["record_id"].astype(str))
    if success & failures:
        raise RuntimeError(f"Overlapping success/failure records for {model}/{cohort}/{phenotype}")
    return success, failures


def build_candidates() -> pd.DataFrame:
    labels = pd.read_csv(LABELS)
    labels["record_id"] = labels.record_id.astype(str)
    waveform = pd.read_csv(WAVEFORM_MANIFEST, low_memory=False)
    waveform["record_id"] = waveform.record_id.astype(str)
    candidate_rows = []
    for cohort in COHORT_NAMES:
        for phenotype in TARGETS:
            success, failures = record_sets("ecg_jepa", cohort, phenotype)
            fm_success, fm_failures = record_sets("ecg_fm", cohort, phenotype)
            if success != fm_success or failures != fm_failures:
                raise RuntimeError(f"Model-specific candidate mismatch for {cohort}/{phenotype}")
            for record_id in sorted(success | failures):
                candidate_rows.append(
                    {
                        "cohort": cohort,
                        "phenotype": phenotype,
                        "record_id": record_id,
                        "success": int(record_id in success),
                    }
                )
    candidates = pd.DataFrame(candidate_rows)
    label_keep = ["cohort", "record_id", "age", "sex", *sorted(set(TARGETS.values()))]
    label_frame = labels[label_keep].rename(columns={"cohort": "label_cohort"})
    candidates["label_cohort"] = candidates.cohort.map(COHORT_NAMES)
    candidates = candidates.merge(
        label_frame,
        on=["label_cohort", "record_id"],
        how="left",
        validate="many_to_one",
    )
    waveform_keep = [
        "cohort",
        "record_id",
        "rr_mean_ms",
        "qrs_duration_ms",
        "qt_like_ms",
        "r_amp_global_mv",
        "st_amp_global_mv",
        "t_amp_global_mv",
    ]
    waveform_frame = waveform[waveform_keep].copy()
    waveform_frame["cohort"] = waveform_frame.cohort.astype(str) + "_f"
    candidates = candidates.merge(
        waveform_frame,
        on=["cohort", "record_id"],
        how="left",
        validate="many_to_one",
    )
    candidates["age"] = pd.to_numeric(candidates.age, errors="coerce")
    candidates["male"] = candidates.sex.astype(str).str.lower().map({"male": 1.0, "female": 0.0})
    candidates["target"] = candidates.phenotype.map(TARGETS)
    candidates["target_label"] = [
        row[row.target] if row.target in candidates.columns else np.nan
        for _, row in candidates.iterrows()
    ]
    candidates["target_label"] = pd.to_numeric(candidates.target_label, errors="coerce")
    if candidates.age.notna().sum() == 0 or candidates.target_label.notna().sum() == 0:
        raise RuntimeError("Candidate metadata join failed")
    return candidates


def standardized_difference(failure: np.ndarray, success: np.ndarray) -> float:
    failure = np.asarray(failure, dtype=float)
    success = np.asarray(success, dtype=float)
    failure = failure[np.isfinite(failure)]
    success = success[np.isfinite(success)]
    if len(failure) < 2 or len(success) < 2:
        return np.nan
    pooled = np.sqrt((np.var(failure, ddof=1) + np.var(success, ddof=1)) / 2.0)
    if pooled <= 0:
        return 0.0
    return float((np.mean(failure) - np.mean(success)) / pooled)


def bias_table(candidates: pd.DataFrame, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for (cohort, phenotype), group in candidates.groupby(["cohort", "phenotype"]):
        failed = group[group.success.eq(0)]
        passed = group[group.success.eq(1)]
        for covariate in COVARIATES:
            x0 = pd.to_numeric(failed[covariate], errors="coerce").to_numpy(dtype=float)
            x1 = pd.to_numeric(passed[covariate], errors="coerce").to_numpy(dtype=float)
            x0 = x0[np.isfinite(x0)]
            x1 = x1[np.isfinite(x1)]
            observed = standardized_difference(x0, x1)
            rng = np.random.default_rng(stable_seed("failure-bias", cohort, phenotype, covariate))
            samples = np.empty(n_bootstrap, dtype=float)
            for index in range(n_bootstrap):
                samples[index] = standardized_difference(
                    rng.choice(x0, len(x0), replace=True),
                    rng.choice(x1, len(x1), replace=True),
                )
            finite = samples[np.isfinite(samples)]
            ci_low, ci_high = (
                np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
            )
            binary = covariate in {"male", "target_label"}
            if binary:
                table = np.asarray(
                    [
                        [(x0 == 1).sum(), (x0 == 0).sum()],
                        [(x1 == 1).sum(), (x1 == 0).sum()],
                    ]
                )
                p_value = float(fisher_exact(table).pvalue) if table.sum() else np.nan
            else:
                p_value = float(ttest_ind(x0, x1, equal_var=False).pvalue)
            rows.append(
                {
                    "cohort": cohort,
                    "phenotype": phenotype,
                    "covariate": covariate,
                    "failed_n": len(x0),
                    "success_n": len(x1),
                    "failed_mean": float(np.mean(x0)),
                    "success_mean": float(np.mean(x1)),
                    "standardized_mean_difference_failure_minus_success": observed,
                    "smd_ci_low": float(ci_low),
                    "smd_ci_high": float(ci_high),
                    "p_two_sided": p_value,
                    "test": "fisher_exact" if binary else "welch_t",
                }
            )
    result = pd.DataFrame(rows)
    result["q_two_sided"] = bh(result.p_two_sided.to_numpy())
    return result


def propensity_pipeline() -> Pipeline:
    numeric = list(COVARIATES)
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("prepare", transformer),
            ("model", LogisticRegression(C=1.0, max_iter=3000, solver="liblinear")),
        ]
    )


def fit_propensity(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weighted = []
    diagnostics = []
    for (cohort, phenotype), group in candidates.groupby(["cohort", "phenotype"], sort=True):
        group = group.copy()
        y = group.success.to_numpy(dtype=int)
        model = propensity_pipeline()
        folds = min(5, int(np.bincount(y).min()))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=stable_seed("propensity-cv", cohort, phenotype),
        )
        cross_fitted = cross_val_predict(
            model,
            group,
            y,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        model.fit(group, y)
        fitted = model.predict_proba(group)[:, 1]
        success_rate = float(y.mean())
        raw_weight = success_rate / np.clip(fitted, 0.02, 0.98)
        success_raw = raw_weight[y == 1]
        low, high = np.quantile(success_raw, [0.01, 0.99])
        clipped = np.clip(raw_weight, low, high)
        group["selection_probability"] = fitted
        group["stabilized_ipw"] = np.where(y == 1, clipped, np.nan)
        weighted.append(group)
        success_weights = clipped[y == 1]
        diagnostics.append(
            {
                "cohort": cohort,
                "phenotype": phenotype,
                "candidate_records": len(group),
                "failed_records": int((y == 0).sum()),
                "success_records": int((y == 1).sum()),
                "success_rate": success_rate,
                "propensity_cross_validated_auroc": float(roc_auc_score(y, cross_fitted)),
                "ipw_min": float(success_weights.min()),
                "ipw_max": float(success_weights.max()),
                "ipw_mean": float(success_weights.mean()),
                "ipw_effective_sample_size": float(
                    success_weights.sum() ** 2 / np.square(success_weights).sum()
                ),
                "weight_clip_low": float(low),
                "weight_clip_high": float(high),
            }
        )
    return pd.concat(weighted, ignore_index=True), pd.DataFrame(diagnostics)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.any():
        return np.nan
    return float(np.average(values[finite], weights=weights[finite]))


def ipw_sensitivity(
    candidates: pd.DataFrame, n_bootstrap: int
) -> pd.DataFrame:
    paired = pd.read_csv(V1 / "waveform_interventions" / "waveform_paired_records.csv")
    weights = candidates[candidates.success.eq(1)][
        ["cohort", "phenotype", "record_id", "stabilized_ipw"]
    ]
    paired["record_id"] = paired.record_id.astype(str)
    paired = paired.merge(
        weights,
        on=["cohort", "phenotype", "record_id"],
        how="left",
        validate="many_to_one",
    )
    if paired.stabilized_ipw.isna().any():
        raise RuntimeError("Missing IPW weights for successful waveform records")
    rows = []
    group_keys = [
        "model",
        "cohort",
        "phenotype",
        "target",
        "variant",
        "direction_sign",
        "strength",
    ]
    for keys, full_group in paired.groupby(group_keys, sort=True):
        for analysis_set, group in (
            ("unfiltered_complete_case", full_group),
            ("measurement_qc", full_group[full_group.measurement_qc_pass.astype(bool)]),
        ):
            if not len(group):
                continue
            weights_array = group.stabilized_ipw.to_numpy(dtype=float)
            row = dict(zip(group_keys, keys))
            row.update(
                {
                    "analysis_set": analysis_set,
                    "records": len(group),
                    "ipw_effective_sample_size": float(
                        weights_array.sum() ** 2 / np.square(weights_array).sum()
                    ),
                }
            )
            for metric in IPW_METRICS:
                values = group[metric].to_numpy(dtype=float)
                unweighted = float(np.nanmean(values))
                weighted = weighted_mean(values, weights_array)
                rng = np.random.default_rng(
                    stable_seed("ipw", *keys, analysis_set, metric)
                )
                samples = np.empty(n_bootstrap, dtype=float)
                for index in range(n_bootstrap):
                    draw = rng.integers(0, len(group), size=len(group))
                    sample_values = values[draw]
                    sample_weights = weights_array[draw]
                    samples[index] = weighted_mean(sample_values, sample_weights) - float(
                        np.nanmean(sample_values)
                    )
                stats = interval_and_p(samples)
                row[f"{metric}_unweighted_mean"] = unweighted
                row[f"{metric}_ipw_mean"] = weighted
                row[f"{metric}_ipw_minus_unweighted"] = weighted - unweighted
                row[f"{metric}_difference_ci_low"] = stats["ci_low"]
                row[f"{metric}_difference_ci_high"] = stats["ci_high"]
                row[f"{metric}_difference_p_two_sided"] = stats["p_two_sided"]
            rows.append(row)
    result = pd.DataFrame(rows)
    for metric in IPW_METRICS:
        result[f"{metric}_difference_q"] = bh(
            result[f"{metric}_difference_p_two_sided"].to_numpy()
        )
    return result


def main() -> None:
    args = parse_args()
    candidates = build_candidates()
    preflight = {
        "candidate_rows": len(candidates),
        "unique_candidate_records_within_strata": int(
            candidates[["cohort", "phenotype", "record_id"]].drop_duplicates().shape[0]
        ),
        "failed_rows": int(candidates.success.eq(0).sum()),
        "success_rows": int(candidates.success.eq(1).sum()),
        "strata": int(candidates.groupby(["cohort", "phenotype"]).ngroups),
    }
    if args.preflight_only:
        print(preflight)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    bias = bias_table(candidates, args.bootstrap)
    candidates_with_weights, diagnostics = fit_propensity(candidates)
    ipw = ipw_sensitivity(candidates_with_weights, args.bootstrap)
    candidates_with_weights.to_csv(args.out / "candidate_selection_table.csv", index=False)
    bias.to_csv(args.out / "candidate_bias_covariates.csv", index=False)
    diagnostics.to_csv(args.out / "propensity_diagnostics.csv", index=False)
    ipw.to_csv(args.out / "waveform_ipw_sensitivity.csv", index=False)
    metadata = {
        "schema_version": 1,
        **preflight,
        "bootstrap_samples": args.bootstrap,
        "bias_rows": len(bias),
        "ipw_rows": len(ipw),
        "candidate_unit": "unique cohort-phenotype-record attempt; duplicate model runs removed",
        "weighting": "stabilized logistic propensity weights clipped at successful-record 1st/99th percentiles",
        "ipw_bootstrap_note": "record bootstrap with fitted propensity weights held fixed",
        "terminology": "unfiltered complete-case, not clinical-trial ITT",
        "all_complete": True,
    }
    write_json(args.out / "metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
