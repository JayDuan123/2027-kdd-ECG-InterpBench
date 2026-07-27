#!/usr/bin/env python
"""Audit and summarize the complete E=8 accessibility calibration ladder."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "dense_fm",
    "full_sae",
    "sae_top16",
    "sae_top4",
    "sae_single",
    "random_single",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v1/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/accessibility_calibration_e8_v1/summary",
    )
    parser.add_argument("--expected-cells", type=int, default=90)
    parser.add_argument("--expected-concepts", type=int, default=49)
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


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = sorted(args.workers_root.glob("task_*/summary.json"))
    errors: list[str] = []
    if len(summaries) != args.expected_cells:
        errors.append(f"expected {args.expected_cells} worker summaries, found {len(summaries)}")
    tables = []
    seen_cells: set[int] = set()
    for summary_path in summaries:
        payload = json.loads(summary_path.read_text())
        if payload.get("status") != "complete":
            errors.append(f"incomplete worker: {summary_path}")
            continue
        index = int(payload["calibration_index"])
        if index in seen_cells:
            errors.append(f"duplicate calibration index: {index}")
        seen_cells.add(index)
        table_path = Path(payload["calibration_table"])
        if not table_path.exists():
            errors.append(f"missing worker table: {table_path}")
            continue
        table = pd.read_csv(table_path)
        expected_rows = args.expected_concepts * len(METHODS)
        if len(table) != expected_rows:
            errors.append(f"{table_path}: expected {expected_rows} rows, found {len(table)}")
        if set(table.method) != set(METHODS):
            errors.append(f"{table_path}: method support mismatch")
        tables.append(table)
    if errors:
        payload = {"status": "failed", "audit_pass": False, "errors": errors}
        atomic_json(args.output_root / "audit.json", payload)
        raise RuntimeError("; ".join(errors))

    frame = pd.concat(tables, ignore_index=True)
    frame.to_csv(args.output_root / "all_cell_concepts.csv", index=False)
    grouped = (
        frame.groupby(["model", "method"], as_index=False)
        .agg(
            cells=("calibration_index", "nunique"),
            concepts=("concept", "nunique"),
            mean_test_abs_r=("test_abs_r", "mean"),
            median_test_abs_r=("test_abs_r", "median"),
            coverage_020=("test_abs_r", lambda values: float(np.mean(values >= 0.20))),
            mean_ratio_to_dense=("ratio_to_dense_fm", "mean"),
            median_ratio_to_dense=("ratio_to_dense_fm", "median"),
        )
        .sort_values(["model", "mean_test_abs_r"], ascending=[True, False])
    )
    grouped.to_csv(args.output_root / "model_method_summary.csv", index=False)
    concept_summary = (
        frame.groupby(["model", "method", "concept", "family"], as_index=False)
        .agg(
            mean_test_abs_r=("test_abs_r", "mean"),
            median_test_abs_r=("test_abs_r", "median"),
            mean_ratio_to_dense=("ratio_to_dense_fm", "mean"),
        )
        .sort_values(["model", "method", "mean_test_abs_r"], ascending=[True, True, False])
    )
    concept_summary.to_csv(args.output_root / "model_concept_summary.csv", index=False)
    family_summary = (
        frame.groupby(["model", "method", "family"], as_index=False)
        .agg(
            concepts=("concept", "nunique"),
            mean_test_abs_r=("test_abs_r", "mean"),
            coverage_020=("test_abs_r", lambda values: float(np.mean(values >= 0.20))),
            mean_ratio_to_dense=("ratio_to_dense_fm", "mean"),
        )
        .sort_values(["model", "method", "family"])
    )
    family_summary.to_csv(args.output_root / "model_family_summary.csv", index=False)

    paired = frame.pivot(
        index=[
            "calibration_index",
            "model",
            "layer",
            "relative_depth",
            "seed",
            "concept",
        ],
        columns="method",
        values="test_abs_r",
    ).reset_index()
    paired["single_minus_random"] = paired["sae_single"] - paired["random_single"]
    paired["full_sae_retention"] = paired["full_sae"] / paired["dense_fm"]
    paired["top16_retention"] = paired["sae_top16"] / paired["dense_fm"]
    cell_advantage = (
        paired.groupby(["model", "relative_depth", "seed"], as_index=False)
        .single_minus_random.mean()
    )
    advantage_rows = []
    for model, model_rows in paired.groupby("model", sort=False):
        model_cells = cell_advantage[cell_advantage.model == model]
        advantage_rows.append(
            {
                "model": model,
                "mean_single_minus_random": float(
                    model_rows.single_minus_random.mean()
                ),
                "concept_cell_win_fraction": float(
                    (model_rows.single_minus_random > 0).mean()
                ),
                "positive_depth_seed_cells": int(
                    (model_cells.single_minus_random > 0).sum()
                ),
                "depth_seed_cells": int(len(model_cells)),
                "min_cell_mean_single_minus_random": float(
                    model_cells.single_minus_random.min()
                ),
                "max_cell_mean_single_minus_random": float(
                    model_cells.single_minus_random.max()
                ),
                "mean_full_sae_retention": float(
                    model_rows.full_sae_retention.mean()
                ),
                "mean_top16_retention": float(model_rows.top16_retention.mean()),
            }
        )
    advantage = pd.DataFrame(advantage_rows)
    advantage.to_csv(args.output_root / "paired_advantage_summary.csv", index=False)

    by_model_method = grouped.set_index(["model", "method"])
    latex_lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & Dense & Full SAE & Top-16 & Top-4 & Single & Random & $\Delta$ \\",
        r"\midrule",
    ]
    for model in grouped.model.drop_duplicates():
        values = {
            method: float(by_model_method.loc[(model, method), "mean_test_abs_r"])
            for method in METHODS
        }
        delta = values["sae_single"] - values["random_single"]
        latex_lines.append(
            f"{model} & {values['dense_fm']:.3f} & {values['full_sae']:.3f} & "
            f"{values['sae_top16']:.3f} & {values['sae_top4']:.3f} & "
            f"{values['sae_single']:.3f} & {values['random_single']:.3f} & "
            f"{delta:+.3f} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    paper_table = args.output_root / "paper_table_accessibility_calibration.tex"
    paper_table.write_text("\n".join(latex_lines) + "\n")

    report_columns = [
        "model",
        "method",
        "mean_test_abs_r",
        "median_test_abs_r",
        "coverage_020",
        "mean_ratio_to_dense",
    ]
    report = [
        "# E=8 Accessibility Calibration Ladder",
        "",
        f"- complete cells: {frame.calibration_index.nunique()}/{args.expected_cells}",
        f"- concepts per cell: {frame.concept.nunique()}",
        f"- methods: {', '.join(METHODS)}",
        "- ridge alpha: fixed at 10 before test evaluation",
        "- localization ratio: held-out method |r| divided by dense-FM held-out |r|",
        "",
        markdown_table(grouped[report_columns]),
        "",
        "Across all models, the learned SAE single coordinate exceeds the matched-random single coordinate in "
        f"{int((cell_advantage.single_minus_random > 0).sum())}/{len(cell_advantage)} depth-by-seed cells.",
        "",
        markdown_table(advantage),
        "",
        "The random baseline uses Gaussian unit directions with the same N=6144, BatchTopK k=96, evaluation batching, and train-only coordinate selection as the learned SAE.",
        "",
        "These are readout/localization comparisons, not certificates of monosemanticity or clinical validity.",
    ]
    (args.output_root / "report.md").write_text("\n".join(report) + "\n")
    audit = {
        "status": "complete",
        "audit_pass": True,
        "errors": [],
        "expected_cells": args.expected_cells,
        "complete_cells": int(frame.calibration_index.nunique()),
        "expected_concepts": args.expected_concepts,
        "observed_concepts": int(frame.concept.nunique()),
        "methods": list(METHODS),
        "rows": int(len(frame)),
        "model_method_summary": str(args.output_root / "model_method_summary.csv"),
        "model_concept_summary": str(args.output_root / "model_concept_summary.csv"),
        "model_family_summary": str(args.output_root / "model_family_summary.csv"),
        "paired_advantage_summary": str(
            args.output_root / "paired_advantage_summary.csv"
        ),
        "paper_table": str(paper_table),
        "report": str(args.output_root / "report.md"),
    }
    atomic_json(args.output_root / "audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
