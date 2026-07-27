#!/usr/bin/env python
"""Paired bootstrap summary for controlled waveform interventions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "benchmark_extension_v1" / "waveform_interventions"
from scripts.benchmark_extension_common import bh  # noqa: E402

MEASUREMENT = {
    "rr_irregularity": "rr_cv",
    "qrs_duration": "qrs_duration_ms",
    "qt_interval": "qt_interval_ms",
}
MIN_CHANGE = {"rr_irregularity": 0.005, "qrs_duration": 5.0, "qt_interval": 10.0}


def bootstrap_mean(values: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    index = rng.integers(0, len(values), size=(n, len(values)))
    return values[index].mean(axis=1)


def p_one_sided(values: np.ndarray, sign: int) -> float:
    oriented = np.asarray(values) * sign
    return (1.0 + float((oriented <= 0).sum())) / (len(oriented) + 1.0)


def main() -> None:
    complete = sorted((BASE / "workers").glob("*/*/*/complete.json"))
    paths = sorted((BASE / "workers").glob("*/*/*/per_variant_metrics.csv"))
    if len(complete) != 12 or len(paths) != 12:
        raise RuntimeError(f"Waveform workers incomplete: complete={len(complete)} metrics={len(paths)}")
    worker_metadata = [json.loads(path.read_text()) for path in complete]
    incomplete = [
        item for item in worker_metadata
        if item.get("status") != "complete" or item.get("successful_records") != item.get("requested_records")
    ]
    if incomplete:
        raise RuntimeError(f"Waveform worker sample-count audit failed: {incomplete}")
    worker_audit = pd.DataFrame(worker_metadata)
    worker_audit["candidate_failure_fraction"] = (
        worker_audit.failed_records / worker_audit.candidate_records_examined
    )
    worker_audit.to_csv(BASE / "waveform_worker_sample_audit.csv", index=False)
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    identity = frame[frame.variant.eq("identity")].copy()
    edited = frame[~frame.variant.eq("identity")].copy()
    key = ["model", "model_suffix", "cohort", "phenotype", "target", "record_id"]
    keep = key + [column for column in identity.columns if column.startswith("head_") or column.startswith("sae_seed")]
    keep += list(MEASUREMENT.values())
    identity = identity[keep].drop_duplicates(key)
    merged = edited.merge(identity, on=key, suffixes=("", "_identity"), validate="many_to_one")
    measurement_columns = list(MEASUREMENT.values())
    for column in measurement_columns:
        merged[f"delta_{column}"] = merged[column] - merged[f"{column}_identity"]
    merged["measurement_name"] = merged.phenotype.map(MEASUREMENT)
    merged["measurement_delta"] = [row[f"delta_{MEASUREMENT[row.phenotype]}"] for _, row in merged.iterrows()]
    merged["measurement_qc_pass"] = (
        merged.direction_sign * merged.measurement_delta
        >= merged.phenotype.map(MIN_CHANGE)
    )
    merged["target_head_delta"] = [
        row[f"head_{row.target}"] - row[f"head_{row.target}_identity"] for _, row in merged.iterrows()
    ]
    for seed in (4311, 4312, 4313):
        merged[f"sae_seed{seed}_delta"] = (
            merged[f"sae_seed{seed}_top5_logit_contribution"]
            - merged[f"sae_seed{seed}_top5_logit_contribution_identity"]
        )
    merged["sae_top5_delta_mean"] = merged[[f"sae_seed{seed}_delta" for seed in (4311, 4312, 4313)]].mean(axis=1)
    off_target = []
    for _, row in merged.iterrows():
        heads = [column[5:] for column in frame.columns if column.startswith("head_")]
        deltas = [abs(row[f"head_{head}"] - row[f"head_{head}_identity"]) for head in heads if head != row.target]
        off_target.append(float(np.mean(deltas)))
    merged["off_target_abs_delta_mean"] = off_target
    merged.to_csv(BASE / "waveform_paired_records.csv", index=False)

    rows = []
    for keys, group in merged.groupby(
        ["model", "cohort", "phenotype", "target", "variant", "direction_sign", "strength"]
    ):
        valid = group[group.measurement_qc_pass].copy()
        if not len(valid):
            continue
        seed = 20260713 + sum(map(ord, "|".join(map(str, keys))))
        measurement_samples = bootstrap_mean(valid.measurement_delta.to_numpy(), 2000, seed)
        head_samples = bootstrap_mean(valid.target_head_delta.to_numpy(), 2000, seed + 1)
        atom_samples = bootstrap_mean(valid.sae_top5_delta_mean.to_numpy(), 2000, seed + 2)
        off_samples = bootstrap_mean(valid.off_target_abs_delta_mean.to_numpy(), 2000, seed + 3)
        itt_measurement_samples = bootstrap_mean(group.measurement_delta.to_numpy(), 2000, seed + 4)
        itt_head_samples = bootstrap_mean(group.target_head_delta.to_numpy(), 2000, seed + 5)
        itt_atom_samples = bootstrap_mean(group.sae_top5_delta_mean.to_numpy(), 2000, seed + 6)
        direction = int(keys[5])
        row = dict(
            zip(("model", "cohort", "phenotype", "target", "variant", "direction_sign", "strength"), keys),
            attempted_records=len(group), valid_records=len(valid), qc_pass_fraction=len(valid) / len(group),
            measurement_delta_mean=float(valid.measurement_delta.mean()),
            measurement_delta_ci_low=float(np.quantile(measurement_samples, .025)),
            measurement_delta_ci_high=float(np.quantile(measurement_samples, .975)),
            measurement_p_one_sided=p_one_sided(measurement_samples, direction),
            target_head_delta_mean=float(valid.target_head_delta.mean()),
            target_head_delta_ci_low=float(np.quantile(head_samples, .025)),
            target_head_delta_ci_high=float(np.quantile(head_samples, .975)),
            target_head_p_one_sided=p_one_sided(head_samples, direction),
            sae_top5_delta_mean=float(valid.sae_top5_delta_mean.mean()),
            sae_top5_delta_ci_low=float(np.quantile(atom_samples, .025)),
            sae_top5_delta_ci_high=float(np.quantile(atom_samples, .975)),
            sae_top5_p_one_sided=p_one_sided(atom_samples, direction),
            off_target_abs_delta_mean=float(valid.off_target_abs_delta_mean.mean()),
            off_target_abs_delta_ci_high=float(np.quantile(off_samples, .975)),
            itt_measurement_delta_mean=float(group.measurement_delta.mean()),
            itt_measurement_delta_ci_low=float(np.quantile(itt_measurement_samples, .025)),
            itt_measurement_delta_ci_high=float(np.quantile(itt_measurement_samples, .975)),
            itt_measurement_p_one_sided=p_one_sided(itt_measurement_samples, direction),
            itt_target_head_delta_mean=float(group.target_head_delta.mean()),
            itt_target_head_delta_ci_low=float(np.quantile(itt_head_samples, .025)),
            itt_target_head_delta_ci_high=float(np.quantile(itt_head_samples, .975)),
            itt_target_head_p_one_sided=p_one_sided(itt_head_samples, direction),
            itt_sae_top5_delta_mean=float(group.sae_top5_delta_mean.mean()),
            itt_sae_top5_delta_ci_low=float(np.quantile(itt_atom_samples, .025)),
            itt_sae_top5_delta_ci_high=float(np.quantile(itt_atom_samples, .975)),
            itt_sae_top5_p_one_sided=p_one_sided(itt_atom_samples, direction),
            bootstrap_samples=2000,
        )
        rows.append(row)
    profile = pd.DataFrame(rows)
    for metric in ("measurement", "target_head", "sae_top5"):
        profile[f"{metric}_q"] = bh(profile[f"{metric}_p_one_sided"].to_numpy())
        profile[f"itt_{metric}_q"] = bh(profile[f"itt_{metric}_p_one_sided"].to_numpy())
    profile["measurement_direction_pass"] = (
        profile.direction_sign * profile.measurement_delta_ci_low.where(profile.direction_sign.gt(0), profile.measurement_delta_ci_high) > 0
    )
    profile["head_direction_pass"] = np.where(
        profile.direction_sign.gt(0), profile.target_head_delta_ci_low > 0, profile.target_head_delta_ci_high < 0
    ) & profile.target_head_q.lt(.05)
    profile["sae_direction_pass"] = np.where(
        profile.direction_sign.gt(0), profile.sae_top5_delta_ci_low > 0, profile.sae_top5_delta_ci_high < 0
    ) & profile.sae_top5_q.lt(.05)
    profile["joint_waveform_grounding_pass"] = (
        profile.measurement_direction_pass & profile.head_direction_pass & profile.sae_direction_pass
    )
    profile["itt_measurement_direction_pass"] = np.where(
        profile.direction_sign.gt(0),
        profile.itt_measurement_delta_ci_low > 0,
        profile.itt_measurement_delta_ci_high < 0,
    )
    profile["itt_head_direction_pass"] = np.where(
        profile.direction_sign.gt(0),
        profile.itt_target_head_delta_ci_low > 0,
        profile.itt_target_head_delta_ci_high < 0,
    ) & profile.itt_target_head_q.lt(.05)
    profile["itt_sae_direction_pass"] = np.where(
        profile.direction_sign.gt(0),
        profile.itt_sae_top5_delta_ci_low > 0,
        profile.itt_sae_top5_delta_ci_high < 0,
    ) & profile.itt_sae_top5_q.lt(.05)
    profile["itt_joint_waveform_grounding_pass"] = (
        profile.itt_measurement_direction_pass
        & profile.itt_head_direction_pass
        & profile.itt_sae_direction_pass
    )
    profile.to_csv(BASE / "waveform_intervention_profile.csv", index=False)
    summary = profile.groupby(["model", "phenotype"], as_index=False).agg(
        intervention_cells=("variant", "size"), attempted_records=("attempted_records", "sum"),
        valid_records=("valid_records", "sum"), qc_pass_fraction=("qc_pass_fraction", "mean"),
        measurement_direction_pass=("measurement_direction_pass", "sum"),
        head_direction_pass=("head_direction_pass", "sum"),
        sae_direction_pass=("sae_direction_pass", "sum"),
        joint_grounding_pass=("joint_waveform_grounding_pass", "sum"),
        itt_measurement_direction_pass=("itt_measurement_direction_pass", "sum"),
        itt_head_direction_pass=("itt_head_direction_pass", "sum"),
        itt_sae_direction_pass=("itt_sae_direction_pass", "sum"),
        itt_joint_grounding_pass=("itt_joint_waveform_grounding_pass", "sum"),
    )
    summary.to_csv(BASE / "waveform_intervention_summary.csv", index=False)
    metadata = {
        "schema_version": 1, "workers": len(complete), "raw_variant_rows": len(frame),
        "paired_edited_rows": len(merged), "profile_cells": len(profile), "bootstrap_samples": 2000,
        "candidate_records_examined": int(worker_audit.candidate_records_examined.sum()),
        "delineation_failures": int(worker_audit.failed_records.sum()),
        "candidate_failure_fraction": float(
            worker_audit.failed_records.sum() / worker_audit.candidate_records_examined.sum()
        ),
        "fdr_family": "all model x cohort x phenotype x direction x dose cells, separately for measurement/head/SAE",
        "claim_boundary": "Controlled waveform sensitivity only; not clinical or generative waveform causality.",
        "sae_grounding_metric": "signed target-head logit contribution of the selected top-5 latents",
        "analysis_sets": "ITT uses all paired records; QC uses records passing the preregistered minimum waveform-measurement change",
        "all_complete": True,
    }
    (BASE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
