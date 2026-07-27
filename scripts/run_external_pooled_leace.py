#!/usr/bin/env python
"""Pooled-representation LEACE audit for one external model/cohort pair.

This is a cohort-local causal readout audit at the pooled representation. It is
not an internal-layer continuation intervention and is reported separately.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results/external_benchmark_v1"
ACT_ROOT = ROOT / "results/activations_external_full_v1/pooled"
RIDGE_REL = 1e-3
PROBE_ALPHA = 10.0
N_RANDOM = 20
FULL_MANIFESTS = {
    "chapman_f": ROOT / "results/activations_external_full_v1/plan_chapman_cpsc/full_manifest.csv",
    "cpsc_f": ROOT / "results/activations_external_full_v1/plan_chapman_cpsc/full_manifest.csv",
    "ningbo_f": ROOT / "results/activations_external_full_v1/plan_ningbo/full_manifest.csv",
    "mimic_f": ROOT / "results/activations_external_full_v1/plan_mimic_100k/mimic_main_manifest.csv",
}

CELL_MAP = {
    "chapman_f": (
        ("rr_mean_ms", "af_rhythm_native", "definition_positive_control"),
        ("heart_rate_bpm", "af_rhythm_native", "definition_positive_control"),
        ("qrs_duration_ms", "bbb_conduction_native", "low_coupling_candidate"),
        ("qt_like_ms", "qt_interval_native", "definition_positive_control"),
        ("qtc_bazett_ms", "qt_interval_native", "definition_positive_control"),
        ("st_amp_global_mv", "st_t_abnormal_native", "definition_positive_control"),
        ("t_amp_global_mv", "st_t_abnormal_native", "definition_positive_control"),
    ),
    "cpsc_f": (
        ("rr_mean_ms", "af_rhythm_native", "definition_positive_control"),
        ("heart_rate_bpm", "af_rhythm_native", "definition_positive_control"),
        ("qrs_duration_ms", "bbb_conduction_native", "low_coupling_candidate"),
    ),
    "ningbo_f": (
        ("rr_mean_ms", "af_rhythm_native", "definition_positive_control"),
        ("heart_rate_bpm", "af_rhythm_native", "definition_positive_control"),
        ("qrs_duration_ms", "bbb_conduction_native", "low_coupling_candidate"),
        ("qt_like_ms", "qt_interval_native", "definition_positive_control"),
        ("qtc_bazett_ms", "qt_interval_native", "definition_positive_control"),
        ("st_amp_global_mv", "st_t_abnormal_native", "definition_positive_control"),
        ("t_amp_global_mv", "st_t_abnormal_native", "definition_positive_control"),
    ),
    "mimic_f": (
        ("rr_mean_ms", "af_rhythm_icd", "high_coupling_control"),
        ("heart_rate_bpm", "af_rhythm_icd", "high_coupling_control"),
        ("qrs_duration_ms", "bbb_conduction_icd", "low_coupling_candidate"),
        ("qt_like_ms", "qt_interval_icd", "high_coupling_control"),
        ("qtc_bazett_ms", "qt_interval_icd", "high_coupling_control"),
        ("st_amp_global_mv", "mi_ischemia_icd", "low_coupling_candidate"),
        ("t_amp_global_mv", "mi_ischemia_icd", "low_coupling_candidate"),
        ("r_amp_global_mv", "hypertrophy_icd", "low_coupling_candidate"),
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-suffix", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--seed", type=int, default=20260712)
    return p.parse_args()


def robust_scale(y: np.ndarray) -> tuple[float, float]:
    median = float(np.nanmedian(y)); q25, q75 = np.nanpercentile(y, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.nanstd(y))
    return median, max(scale, 1e-8)


def covariance_maps(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu = x.mean(0); centered = x - mu
    cov = centered.T @ centered / max(len(centered), 1)
    ridge_abs = RIDGE_REL * float(np.trace(cov) / cov.shape[0])
    ridge_abs = ridge_abs if np.isfinite(ridge_abs) and ridge_abs > 0 else RIDGE_REL
    eigvals, eigvecs = np.linalg.eigh(cov + ridge_abs * np.eye(cov.shape[0]))
    eigvals = np.maximum(eigvals, 1e-12)
    sqrt_cov = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
    inv_sqrt_cov = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T
    return mu, sqrt_cov, inv_sqrt_cov, ridge_abs


def eraser_from_direction(mu: np.ndarray, sqrt_cov: np.ndarray, inv_sqrt_cov: np.ndarray,
                          direction: np.ndarray) -> dict[str, np.ndarray]:
    u = direction / max(float(np.linalg.norm(direction)), 1e-12)
    remove = inv_sqrt_cov @ np.outer(u, u) @ sqrt_cov
    return {"mu": mu.astype(np.float32), "remove": remove.astype(np.float32)}


def erase(x: np.ndarray, eraser: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32) - (np.asarray(x, dtype=np.float32) - eraser["mu"]) @ eraser["remove"]


def probe_r2(xtr: np.ndarray, xte: np.ndarray, ytr: np.ndarray, yte: np.ndarray) -> float:
    scaler = StandardScaler().fit(xtr)
    model = Ridge(alpha=PROBE_ALPHA, solver="lsqr").fit(scaler.transform(xtr), ytr)
    pred = model.predict(scaler.transform(xte))
    denom = float(np.sum((yte - yte.mean()) ** 2))
    return float(1.0 - np.sum((yte - pred) ** 2) / max(denom, 1e-12))


def main() -> None:
    args = parse_args(); cohort = args.cohort.lower().replace("-", "_")
    if cohort not in CELL_MAP:
        raise ValueError(cohort)
    pair = BASE / args.model_suffix / cohort
    bundle = joblib.load(pair / "frozen_heads.joblib")
    from scripts.train_external_frozen_heads import load_activations
    from scripts.run_external_layer_probe import concept_frame
    x, record_ids = load_activations(ACT_ROOT / args.model_suffix / cohort)
    if not np.array_equal(record_ids.astype(str), np.asarray(bundle["record_ids"]).astype(str)):
        raise RuntimeError("Activation/head record order mismatch")
    split = np.asarray(bundle["split"]); tr = split == "train"; va = split == "val"; te = split == "test"
    groups = np.asarray(bundle.get("group_ids", record_ids)).astype("U64")
    concepts = concept_frame(FULL_MANIFESTS[cohort], cohort, record_ids.astype(str))
    rng = np.random.default_rng(args.seed)
    out_root = pair / "pooled_leace"; out_root.mkdir(parents=True, exist_ok=True)
    completed = 0; skipped = 0

    for concept, task, coupling in CELL_MAP[cohort]:
        out = out_root / f"{concept}__to__{task}"
        final = out / "result.json"; records_path = out / "records.npz"
        if final.exists() and records_path.exists():
            completed += 1; continue
        if task not in bundle["heads"]:
            out.mkdir(parents=True, exist_ok=True)
            payload = {"status": "skipped", "reason": "head_not_trained", "concept": concept, "task": task}
            tmp = out / f"skipped.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(payload, indent=2)+"\n"); tmp.replace(out/"skipped.json")
            skipped += 1; continue
        y_concept = pd.to_numeric(concepts[concept], errors="coerce").to_numpy(dtype=float)
        valid_tr = tr & np.isfinite(y_concept); valid_va = va & np.isfinite(y_concept); valid_te = te & np.isfinite(y_concept)
        if valid_tr.sum() < 100 or valid_va.sum() < 20 or valid_te.sum() < 20:
            out.mkdir(parents=True, exist_ok=True)
            payload = {"status": "skipped", "reason": "insufficient_concept_support", "concept": concept, "task": task,
                       "n_train": int(valid_tr.sum()), "n_val": int(valid_va.sum()), "n_test": int(valid_te.sum())}
            tmp = out/f"skipped.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(payload,indent=2)+"\n"); tmp.replace(out/"skipped.json")
            skipped += 1; continue

        x_fit = np.asarray(x[valid_tr], dtype=np.float64)
        median, target_scale = robust_scale(y_concept[valid_tr])
        yc = (y_concept[valid_tr] - median) / target_scale; yc -= yc.mean()
        y_probe_train = (y_concept[valid_tr] - median) / target_scale
        y_probe_val = (y_concept[valid_va] - median) / target_scale
        pooled_scaler = StandardScaler().fit(x[valid_tr])
        x_probe_train = pooled_scaler.transform(x[valid_tr]); x_probe_val = pooled_scaler.transform(x[valid_va])
        pooled_probe = Ridge(alpha=PROBE_ALPHA, solver="lsqr").fit(x_probe_train, y_probe_train)
        val_probe_r2 = float(1.0 - np.sum((y_probe_val - pooled_probe.predict(x_probe_val))**2)
                             / max(float(np.sum((y_probe_val-y_probe_val.mean())**2)), 1e-12))
        shuffled_target = y_probe_train.copy(); rng.shuffle(shuffled_target)
        gaussian_target = rng.normal(size=len(y_probe_train))
        shuffled_probe = Ridge(alpha=PROBE_ALPHA, solver="lsqr").fit(x_probe_train, shuffled_target)
        gaussian_probe = Ridge(alpha=PROBE_ALPHA, solver="lsqr").fit(x_probe_train, gaussian_target)
        val_shuffled_r2 = float(1.0 - np.sum((y_probe_val-shuffled_probe.predict(x_probe_val))**2)
                                / max(float(np.sum((y_probe_val-y_probe_val.mean())**2)), 1e-12))
        val_gaussian_r2 = float(1.0 - np.sum((y_probe_val-gaussian_probe.predict(x_probe_val))**2)
                                / max(float(np.sum((y_probe_val-y_probe_val.mean())**2)), 1e-12))
        strict_encoded = bool(
            val_probe_r2 >= 0.04
            and val_probe_r2-val_shuffled_r2 >= 0.01
            and val_probe_r2-val_gaussian_r2 >= 0.01
        )
        mu, sqrt_cov, inv_sqrt_cov, ridge_abs = covariance_maps(x_fit)
        whitened = (x_fit - mu) @ inv_sqrt_cov
        direction = whitened.T @ yc / len(yc)
        real = eraser_from_direction(mu, sqrt_cov, inv_sqrt_cov, direction)
        random_erasers = [eraser_from_direction(mu, sqrt_cov, inv_sqrt_cov, rng.normal(size=x.shape[1])) for _ in range(N_RANDOM)]

        ytr = (y_concept[valid_tr] - median) / target_scale
        yte = (y_concept[valid_te] - median) / target_scale
        original_r2 = probe_r2(x[valid_tr], x[valid_te], ytr, yte)
        erased_train = erase(x[valid_tr], real); erased_test_valid = erase(x[valid_te], real)
        residual_r2 = probe_r2(erased_train, erased_test_valid, ytr, yte)
        residual_threshold = max(0.02, 0.35 * max(original_r2, 0.04))

        clf = bundle["heads"][task]["clf"]; scaler = bundle["scaler"]
        y_task = np.asarray(bundle["heads"][task]["labels"], dtype=int)[te]
        x_test = np.asarray(x[te], dtype=np.float32)
        base_score = clf.decision_function(scaler.transform(x_test)).reshape(-1)
        erased_score = clf.decision_function(scaler.transform(erase(x_test, real))).reshape(-1)
        random_scores = np.column_stack([
            clf.decision_function(scaler.transform(erase(x_test, random))).reshape(-1)
            for random in random_erasers
        ])
        base_auroc = float(roc_auc_score(y_task, base_score))
        erased_auroc = float(roc_auc_score(y_task, erased_score))
        random_aurocs = np.asarray([roc_auc_score(y_task, random_scores[:, j]) for j in range(N_RANDOM)])
        payload = {
            "schema_version": 1, "status": "ok", "model_suffix": args.model_suffix, "cohort": cohort,
            "concept": concept, "task": task, "coupling_role": coupling, "representation": "pooled_head_input",
            "continuation_claim": False, "split_unit": bundle.get("split_unit", "record"),
            "n_train_concept": int(valid_tr.sum()), "n_test_concept": int(valid_te.sum()),
            "n_val_concept": int(valid_va.sum()),
            "pooled_val_probe_r2": val_probe_r2,
            "pooled_val_probe_r2_shuffled": val_shuffled_r2,
            "pooled_val_probe_r2_gaussian": val_gaussian_r2,
            "pooled_strict_encoded": strict_encoded,
            "n_test_task": int(te.sum()), "eraser_method": "closed_form_leace", "eraser_rank": 1,
            "leace_ridge_relative": RIDGE_REL, "leace_ridge_absolute": ridge_abs,
            "original_probe_r2": original_r2, "residual_probe_r2": residual_r2,
            "residual_probe_threshold": residual_threshold, "eraser_effective": bool(residual_r2 < residual_threshold),
            "base_auroc": base_auroc, "erased_auroc": erased_auroc,
            "delta_auroc": base_auroc - erased_auroc,
            "random_auroc_mean": float(random_aurocs.mean()),
            "delta_auroc_minus_random": float(random_aurocs.mean() - erased_auroc),
            "base_auprc": float(average_precision_score(y_task, base_score)),
            "erased_auprc": float(average_precision_score(y_task, erased_score)),
            "n_random": N_RANDOM, "seed": args.seed,
        }
        out.mkdir(parents=True, exist_ok=True)
        tmp_npz = out / f"records.npz.tmp.{os.getpid()}"
        with tmp_npz.open("wb") as handle:
            np.savez(handle, group_ids=groups[te], y=y_task.astype(np.int8), base_score=base_score.astype(np.float32),
                     erased_score=erased_score.astype(np.float32), random_scores=random_scores.astype(np.float32))
        tmp_npz.replace(records_path)
        tmp = out / f"result.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(payload, indent=2)+"\n"); tmp.replace(final)
        completed += 1
        print(json.dumps({"concept": concept, "task": task, "delta_minus_random": payload["delta_auroc_minus_random"],
                          "residual_r2": residual_r2}))

    summary = {"model_suffix": args.model_suffix, "cohort": cohort, "requested_cells": len(CELL_MAP[cohort]),
               "completed_cells": completed, "skipped_cells": skipped}
    tmp = out_root/f"pair_summary.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(summary,indent=2)+"\n"); tmp.replace(out_root/"pair_summary.json")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
