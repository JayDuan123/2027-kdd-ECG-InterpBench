#!/usr/bin/env python
"""Audit and summarize dense-coordinate and replicated random E=8 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "accessibility_calibration_e8_v2"
METHODS = (
    "dense_fm",
    "dense_single",
    "full_sae",
    "sae_top16",
    "sae_top4",
    "sae_single",
    "random_single",
)
V1_METHODS = ("dense_fm", "full_sae", "sae_top16", "sae_top4", "sae_single")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v2/workers",
    )
    parser.add_argument(
        "--v1-summary-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v1/summary",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v2/summary",
    )
    parser.add_argument("--expected-groups", type=int, default=30)
    parser.add_argument("--expected-concepts", type=int, default=49)
    parser.add_argument("--expected-random-replicates", type=int, default=20)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def method_summary(
    v1: pd.DataFrame, dense: pd.DataFrame, random: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for (model, method), values in v1[v1.method.isin(V1_METHODS)].groupby(
        ["model", "method"], sort=False
    ):
        rows.append(
            {
                "model": model,
                "method": method,
                "depths": values.relative_depth.nunique(),
                "sae_seeds": values.seed.nunique(),
                "random_replicates": 0,
                "concepts": values.concept.nunique(),
                "mean_test_abs_r": values.test_abs_r.mean(),
                "median_test_abs_r": values.test_abs_r.median(),
                "coverage_020": np.mean(values.test_abs_r >= 0.20),
            }
        )
    for model, values in dense.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "method": "dense_single",
                "depths": values.relative_depth.nunique(),
                "sae_seeds": 0,
                "random_replicates": 0,
                "concepts": values.concept.nunique(),
                "mean_test_abs_r": values.test_abs_r.mean(),
                "median_test_abs_r": values.test_abs_r.median(),
                "coverage_020": values.covered_020.mean(),
            }
        )
    for model, values in random.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "method": "random_single",
                "depths": values.relative_depth.nunique(),
                "sae_seeds": 0,
                "random_replicates": values.random_replicate.nunique(),
                "concepts": values.concept.nunique(),
                "mean_test_abs_r": values.test_abs_r.mean(),
                "median_test_abs_r": values.test_abs_r.median(),
                "coverage_020": values.covered_020.mean(),
            }
        )
    frame = pd.DataFrame(rows)
    order = {method: index for index, method in enumerate(METHODS)}
    frame["method_order"] = frame.method.map(order)
    return frame.sort_values(["model", "method_order"]).drop(columns="method_order")


def paired_summary(
    v1: pd.DataFrame, dense: pd.DataFrame, random: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["model", "relative_depth", "concept", "family"]
    sae_rows = v1[v1.method == "sae_single"].copy()
    sae_rows["covered_020"] = (sae_rows.test_abs_r >= 0.20).astype(int)
    sae = (
        sae_rows
        .groupby(keys, as_index=False)
        .agg(
            sae_single_abs_r=("test_abs_r", "mean"),
            sae_coverage_probability=("covered_020", "mean"),
        )
    )
    dense_one = dense[keys + ["test_abs_r", "covered_020"]].rename(
        columns={
            "test_abs_r": "dense_single_abs_r",
            "covered_020": "dense_covered_020",
        }
    )
    random_mean = (
        random.groupby(keys, as_index=False)
        .agg(
            random_single_abs_r=("test_abs_r", "mean"),
            random_coverage_probability=("covered_020", "mean"),
        )
    )
    paired = sae.merge(dense_one, on=keys, validate="one_to_one").merge(
        random_mean, on=keys, validate="one_to_one"
    )
    paired["sae_minus_dense_single"] = (
        paired.sae_single_abs_r - paired.dense_single_abs_r
    )
    paired["sae_minus_random_single"] = (
        paired.sae_single_abs_r - paired.random_single_abs_r
    )
    depth = (
        paired.groupby(["model", "relative_depth"], as_index=False)
        .agg(
            sae_minus_dense_single=("sae_minus_dense_single", "mean"),
            sae_minus_random_single=("sae_minus_random_single", "mean"),
        )
    )
    rows = []
    for model, values in paired.groupby("model", sort=False):
        model_depth = depth[depth.model == model]
        rows.append(
            {
                "model": model,
                "mean_sae_minus_dense_single": values.sae_minus_dense_single.mean(),
                "sae_over_dense_concept_depth_fraction": np.mean(
                    values.sae_minus_dense_single > 0
                ),
                "positive_sae_minus_dense_depths": int(
                    np.sum(model_depth.sae_minus_dense_single > 0)
                ),
                "mean_sae_minus_random_single": values.sae_minus_random_single.mean(),
                "sae_over_random_concept_depth_fraction": np.mean(
                    values.sae_minus_random_single > 0
                ),
                "positive_sae_minus_random_depths": int(
                    np.sum(model_depth.sae_minus_random_single > 0)
                ),
                "depths": len(model_depth),
                "sae_coverage_020": values.sae_coverage_probability.mean(),
                "dense_coverage_020": values.dense_covered_020.mean(),
                "random_coverage_020": values.random_coverage_probability.mean(),
            }
        )
    return paired, pd.DataFrame(rows)


def random_seed_intervals(random: pd.DataFrame) -> pd.DataFrame:
    replicate = (
        random.groupby(["model", "random_replicate"], as_index=False)
        .agg(
            mean_test_abs_r=("test_abs_r", "mean"),
            coverage_020=("covered_020", "mean"),
        )
    )
    rows = []
    for model, values in replicate.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "random_replicates": len(values),
                "mean_test_abs_r": values.mean_test_abs_r.mean(),
                "mean_test_abs_r_q025": values.mean_test_abs_r.quantile(0.025),
                "mean_test_abs_r_q975": values.mean_test_abs_r.quantile(0.975),
                "coverage_020": values.coverage_020.mean(),
                "coverage_020_q025": values.coverage_020.quantile(0.025),
                "coverage_020_q975": values.coverage_020.quantile(0.975),
            }
        )
    return pd.DataFrame(rows)


def write_paper_table(summary: pd.DataFrame, path: Path) -> None:
    lookup = summary.set_index(["model", "method"])
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Dense ridge & Dense 1D & Full SAE & Top-16 & Top-4 & SAE 1D & Random 1D \\",
        r"\midrule",
    ]
    for model in summary.model.drop_duplicates():
        value = {
            method: {
                "r": float(lookup.loc[(model, method), "mean_test_abs_r"]),
                "coverage": float(lookup.loc[(model, method), "coverage_020"]),
            }
            for method in METHODS
        }
        lines.append(
            f"{model} & {value['dense_fm']['r']:.3f} & "
            f"{value['dense_single']['r']:.3f}/{value['dense_single']['coverage']:.3f} & "
            f"{value['full_sae']['r']:.3f} & {value['sae_top16']['r']:.3f} & "
            f"{value['sae_top4']['r']:.3f} & "
            f"{value['sae_single']['r']:.3f}/{value['sae_single']['coverage']:.3f} & "
            f"{value['random_single']['r']:.3f}/{value['random_single']['coverage']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    errors = []
    v1_audit_path = args.v1_summary_root / "audit.json"
    v1_audit = json.loads(v1_audit_path.read_text()) if v1_audit_path.exists() else {}
    if not v1_audit.get("audit_pass") or int(v1_audit.get("complete_cells", 0)) != 90:
        errors.append("v1 accessibility calibration audit is missing or incomplete")
    summaries = sorted(args.workers_root.glob("baseline_*/summary.json"))
    if len(summaries) != args.expected_groups:
        errors.append(f"expected {args.expected_groups} worker summaries, found {len(summaries)}")
    dense_tables = []
    random_tables = []
    seen_indices = set()
    seen_model_layers = set()
    test_record_hashes = set()
    observed_random_seed_sets = set()
    for path in summaries:
        payload = json.loads(path.read_text())
        if payload.get("status") != "complete" or payload.get("protocol") != PROTOCOL:
            errors.append(f"incomplete or wrong-protocol worker: {path}")
            continue
        index = int(payload["baseline_index"])
        identity = (payload["model_safe"], int(payload["layer"]))
        if index in seen_indices or identity in seen_model_layers:
            errors.append(f"duplicate baseline worker identity: {path}")
        seen_indices.add(index)
        seen_model_layers.add(identity)
        if int(payload.get("random_replicates", 0)) != args.expected_random_replicates:
            errors.append(f"wrong random replicate count: {path}")
        random_seed_tuple = tuple(int(value) for value in payload.get("random_seeds", []))
        if len(random_seed_tuple) != args.expected_random_replicates:
            errors.append(f"wrong random seed count: {path}")
        observed_random_seed_sets.add(random_seed_tuple)
        normalization = payload.get("normalization_audit", {})
        if float(normalization.get("max_abs_mean_difference", 1.0)) > 1e-7:
            errors.append(f"normalization mean mismatch: {path}")
        if float(normalization.get("max_abs_scale_difference", 1.0)) > 1e-7:
            errors.append(f"normalization scale mismatch: {path}")
        dense_path = Path(payload["dense_single_table"])
        random_path = Path(payload["random_replicates_table"])
        predictions_path = Path(payload["test_predictions"])
        if not dense_path.exists() or not random_path.exists() or not predictions_path.exists():
            errors.append(f"missing worker artifact: {path}")
            continue
        dense = pd.read_csv(dense_path)
        random = pd.read_csv(random_path)
        if len(dense) != args.expected_concepts:
            errors.append(f"wrong dense row count: {dense_path}")
        if len(random) != args.expected_concepts * args.expected_random_replicates:
            errors.append(f"wrong random row count: {random_path}")
        if set(random.random_replicate) != set(range(args.expected_random_replicates)):
            errors.append(f"random replicate support mismatch: {random_path}")
        if dense.duplicated(["model", "relative_depth", "concept"]).any():
            errors.append(f"duplicate dense concept row: {dense_path}")
        if random.duplicated(
            ["model", "relative_depth", "random_replicate", "concept"]
        ).any():
            errors.append(f"duplicate random concept row: {random_path}")
        for table_path, table in ((dense_path, dense), (random_path, random)):
            values = table[["validation_r", "test_r", "test_abs_r"]].to_numpy()
            if not np.isfinite(values).all() or np.max(np.abs(values)) > 1.000001:
                errors.append(f"invalid correlation value: {table_path}")
        with np.load(predictions_path, allow_pickle=False) as archive:
            required = {
                "concept_names",
                "test_ecg_ids",
                "test_patient_ids",
                "y_test",
                "prediction_dense_single",
                "prediction_random_single",
                "random_seeds",
            }
            if set(archive.files) != required:
                errors.append(f"prediction archive key mismatch: {predictions_path}")
            else:
                n_test = int(payload["n_test"])
                expected_shapes = {
                    "concept_names": (args.expected_concepts,),
                    "test_ecg_ids": (n_test,),
                    "test_patient_ids": (n_test,),
                    "y_test": (n_test, args.expected_concepts),
                    "prediction_dense_single": (n_test, args.expected_concepts),
                    "prediction_random_single": (
                        args.expected_random_replicates,
                        n_test,
                        args.expected_concepts,
                    ),
                    "random_seeds": (args.expected_random_replicates,),
                }
                for key, expected_shape in expected_shapes.items():
                    if archive[key].shape != expected_shape:
                        errors.append(
                            f"prediction shape mismatch {key}: "
                            f"{archive[key].shape} != {expected_shape}"
                        )
                for key in ("y_test", "prediction_dense_single", "prediction_random_single"):
                    if not np.isfinite(archive[key]).all():
                        errors.append(f"non-finite prediction archive value {key}: {predictions_path}")
                if tuple(archive["random_seeds"].astype(int)) != random_seed_tuple:
                    errors.append(f"prediction random seeds mismatch: {predictions_path}")
                record_digest = hashlib.sha256(
                    "\n".join(archive["test_ecg_ids"].astype(str)).encode()
                ).hexdigest()
                test_record_hashes.add(record_digest)
        dense_tables.append(dense)
        random_tables.append(random)
    if len(observed_random_seed_sets) != 1:
        errors.append("random seed sets differ across model-depth workers")
    if len(test_record_hashes) != 1:
        errors.append("test record identity differs across model-depth workers")
    if errors:
        audit = {"status": "failed", "audit_pass": False, "errors": errors}
        atomic_json(args.output_root / "audit.json", audit)
        raise RuntimeError("; ".join(errors))

    v1 = pd.read_csv(args.v1_summary_root / "all_cell_concepts.csv")
    dense = pd.concat(dense_tables, ignore_index=True)
    random = pd.concat(random_tables, ignore_index=True)
    if len(dense) != args.expected_groups * args.expected_concepts:
        raise RuntimeError("global dense row count mismatch")
    if len(random) != args.expected_groups * args.expected_concepts * args.expected_random_replicates:
        raise RuntimeError("global random row count mismatch")
    dense.to_csv(args.output_root / "all_dense_single.csv", index=False)
    random.to_csv(args.output_root / "all_random_replicates.csv", index=False)

    summary = method_summary(v1, dense, random)
    if set(summary.method) != set(METHODS) or len(summary) != 6 * len(METHODS):
        raise RuntimeError("model-method summary support mismatch")
    summary.to_csv(args.output_root / "model_method_summary.csv", index=False)
    paired, paired_model = paired_summary(v1, dense, random)
    paired.to_csv(args.output_root / "paired_concept_depth.csv", index=False)
    paired_model.to_csv(args.output_root / "paired_model_summary.csv", index=False)
    random_intervals = random_seed_intervals(random)
    random_intervals.to_csv(args.output_root / "random_seed_intervals.csv", index=False)
    paper_table = args.output_root / "paper_table_accessibility_calibration_v2.tex"
    write_paper_table(summary, paper_table)

    report_columns = [
        "model",
        "method",
        "mean_test_abs_r",
        "median_test_abs_r",
        "coverage_020",
    ]
    report = [
        "# E=8 Dense-Coordinate and Replicated-Random Calibration",
        "",
        f"- complete model-depth groups: {args.expected_groups}/{args.expected_groups}",
        f"- concepts: {args.expected_concepts}",
        f"- random dictionaries per model-depth: {args.expected_random_replicates}",
        "- all coordinate selection uses the frozen 4,096-record training subset",
        "",
        markdown_table(summary[report_columns]),
        "",
        "## Paired localization differences",
        "",
        markdown_table(paired_model),
        "",
        "## Random-seed intervals",
        "",
        markdown_table(random_intervals),
        "",
        "Dense 1D tests localization in native FM axes. Random 1D controls width, ReLU, BatchTopK sparsity, batching, and train-selection multiplicity. Neither is a monosemanticity or clinical-validity certificate.",
    ]
    report_path = args.output_root / "report.md"
    report_path.write_text("\n".join(report) + "\n")
    audit = {
        "status": "complete",
        "audit_pass": True,
        "errors": [],
        "protocol": PROTOCOL,
        "complete_groups": args.expected_groups,
        "expected_groups": args.expected_groups,
        "concepts": args.expected_concepts,
        "random_replicates_per_group": args.expected_random_replicates,
        "dense_rows": int(len(dense)),
        "random_rows": int(len(random)),
        "paired_rows": int(len(paired)),
        "model_method_rows": int(len(summary)),
        "v1_audit_pass": True,
        "random_seeds": list(next(iter(observed_random_seed_sets))),
        "test_record_sha256": next(iter(test_record_hashes)),
        "model_method_summary": str(args.output_root / "model_method_summary.csv"),
        "paired_model_summary": str(args.output_root / "paired_model_summary.csv"),
        "random_seed_intervals": str(args.output_root / "random_seed_intervals.csv"),
        "paper_table": str(paper_table),
        "report": str(report_path),
    }
    atomic_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
