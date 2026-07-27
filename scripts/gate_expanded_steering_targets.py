#!/usr/bin/env python
"""Freeze model-target eligibility from prevalence and frozen-readout quality."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
MODELS = ["CSFM", "CARDIAC-FM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--new-source",
        default=None,
        help="Also write eligible_new_steering_cells.csv for this registry source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args(); base = args.base
    registry = pd.read_csv(base / "candidate_target_registry.csv"); coverage = pd.read_csv(base / "candidate_target_coverage.csv").set_index("target")
    rows = []
    for model in MODELS:
        safe = model.lower().replace("-", "_"); path = base / "models" / safe / "frozen_heads.metrics.json"
        if not path.exists(): raise FileNotFoundError(path)
        metrics = json.loads(path.read_text())
        for spec in registry.itertuples(index=False):
            cov = coverage.loc[spec.target]; metric = metrics[spec.target]
            if spec.target_type == "binary":
                prevalence_pass = int(cov.train_positive) >= 150 and int(cov.test_positive_patients) >= 30
                readout_metric = float(metric["test_auroc"]); readout_pass = readout_metric >= .75
                metric_name = "test_auroc"
            else:
                prevalence_pass = int(cov.train_valid) >= 5000 and int(cov.test_valid) >= 1000
                readout_metric = float(metric["test_r2"]); readout_pass = readout_metric >= .20
                metric_name = "test_r2"
            rows.append({"model": model, "target": spec.target, "target_type": spec.target_type, "family": spec.family,
                         "analysis_role": spec.analysis_role, "prevalence_pass": bool(prevalence_pass),
                         "readout_metric_name": metric_name, "readout_test_metric": readout_metric,
                         "readout_pass": bool(readout_pass), "cell_eligible": bool(prevalence_pass and readout_pass)})
    gate = pd.DataFrame(rows)
    counts = gate.groupby("target").cell_eligible.sum().rename("eligible_models")
    gate = gate.merge(counts, on="target", how="left")
    gate["headline_eligible"] = gate.cell_eligible & (gate.eligible_models >= 3)
    gate["analysis_tier"] = "excluded"
    gate.loc[gate.cell_eligible & (gate.eligible_models < 3), "analysis_tier"] = "extended_model_specific"
    gate.loc[gate.headline_eligible, "analysis_tier"] = "headline_cross_model"
    gate.to_csv(base / "model_target_gate.csv", index=False)
    eligible = gate[gate.cell_eligible].copy(); eligible.insert(0, "task_index", range(len(eligible)))
    eligible.to_csv(base / "eligible_steering_cells.csv", index=False)
    if args.new_source is not None:
        source_targets = set(registry.loc[registry.source.eq(args.new_source), "target"])
        eligible_new = eligible[eligible.target.isin(source_targets)].copy()
        eligible_new["task_index"] = range(len(eligible_new))
        eligible_new.to_csv(base / "eligible_new_steering_cells.csv", index=False)
    target_summary = gate.groupby(["target", "target_type", "family", "analysis_role"], as_index=False).agg(
        eligible_models=("cell_eligible", "sum"), min_readout=("readout_test_metric", "min"), median_readout=("readout_test_metric", "median"))
    target_summary["headline_cross_model"] = target_summary.eligible_models >= 3
    target_summary.to_csv(base / "target_gate_summary.csv", index=False)
    print(f"candidate model-target cells={len(gate)} eligible={len(eligible)} headline targets={(target_summary.headline_cross_model).sum()}")
    print(target_summary.sort_values(["headline_cross_model", "eligible_models", "target"], ascending=[False, False, True]).to_string(index=False))


if __name__ == "__main__": main()
