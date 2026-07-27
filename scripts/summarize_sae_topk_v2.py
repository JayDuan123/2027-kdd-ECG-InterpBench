#!/usr/bin/env python
"""Summarize the recon-matched top-k SAE group steering audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    display = df.copy()
    for col in display.select_dtypes(include=["float"]).columns:
        display[col] = display[col].map(
            lambda value: f"{value:.4g}" if pd.notna(value) else ""
        )
    lines = [
        "| " + " | ".join(display.columns) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in display.columns) + " |")
    return "\n".join(lines)


def load_results(run_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_root.glob("task_*/sae_layer_per_cell.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["result_csv"] = str(path)
        task_token = path.parent.name.split("_", 2)[1]
        frame["task_index"] = int(task_token)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def classify(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "target_effect",
        "random_target_effect_mean",
        "excess_selectivity_ci_low",
        "wbi_improvement_ci_low",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["difference_significant"] = out["excess_selectivity_ci_low"] > 0.0
    out["wbi_ratio_stable"] = (
        (out["target_effect"] > 0.0) & (out["random_target_effect_mean"] > 0.0)
    )
    out["ratio_corrob_positive"] = (
        out["wbi_ratio_stable"] & (out["wbi_improvement_ci_low"] > 0.0)
    )
    out["full_selective_pass"] = (
        out["difference_significant"] & out["ratio_corrob_positive"]
    )
    status = pd.Series("not_selective_by_difference", index=out.index, dtype=object)
    status.loc[out["full_selective_pass"]] = "full_selective"
    status.loc[
        out["difference_significant"] & ~out["wbi_ratio_stable"]
    ] = "difference_positive_ratio_unstable"
    status.loc[
        out["difference_significant"]
        & out["wbi_ratio_stable"]
        & ~out["ratio_corrob_positive"]
    ] = "difference_positive_ratio_not_corrob"
    out["steering_reclass"] = status
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/sae_extension/six_model_sae_audit/topk_group_steering_v2"
        ),
    )
    args = parser.parse_args()
    manifest = pd.read_csv(args.root / "topk_v2_manifest.csv")
    coverage = pd.read_csv(args.root / "topk_v2_coverage.csv")
    raw = load_results(args.root / "runs")
    if raw.empty:
        completed = pd.DataFrame(columns=list(manifest.columns))
    else:
        completed = classify(manifest.merge(raw, on="task_index", suffixes=("_manifest", "")))

    done = set(completed.get("task_index", pd.Series(dtype=int)).astype(int).tolist())
    missing = manifest[~manifest["task_index"].isin(done)].copy()
    completed.to_csv(args.root / "topk_v2_combined_results.csv", index=False)
    missing.to_csv(args.root / "topk_v2_missing_tasks.csv", index=False)

    if completed.empty:
        by_k = pd.DataFrame()
        by_model_k = pd.DataFrame()
    else:
        by_k = (
            completed.groupby("group_k")
            .agg(
                cells=("task_index", "size"),
                models=("model", "nunique"),
                full_selective=("full_selective_pass", "sum"),
                difference_significant=("difference_significant", "sum"),
                ratio_stable=("wbi_ratio_stable", "sum"),
                median_excess=("excess_selectivity", "median"),
                median_target_effect=("target_effect", "median"),
            )
            .reset_index()
        )
        by_model_k = (
            completed.groupby(["model", "group_k"])
            .agg(
                cells=("task_index", "size"),
                full_selective=("full_selective_pass", "sum"),
                difference_significant=("difference_significant", "sum"),
                median_excess=("excess_selectivity", "median"),
            )
            .reset_index()
        )
    by_k.to_csv(args.root / "topk_v2_by_k_summary.csv", index=False)
    by_model_k.to_csv(args.root / "topk_v2_by_model_k_summary.csv", index=False)

    unavailable = coverage[coverage["group_status"] == "k_unavailable"]
    report = [
        "# SAE Top-k Group Steering Audit v2",
        "",
        "Recon-band-matched, activation-ranked SAE feature groups are clamped to the train population centroid. Every endpoint is compared with same-size random feature groups and evaluated using a shared patient-level paired bootstrap.",
        "",
        f"- expected runnable tasks: {len(manifest)}",
        f"- completed tasks: {len(completed)}",
        f"- missing tasks: {len(missing)}",
        f"- explicitly unavailable k/cell combinations: {len(unavailable)}",
        "- `top10` is never silently truncated when the selected dictionary has fewer than 10 atoms.",
        "",
        "## By Group Size",
        "",
        md_table(by_k),
        "",
        "## By Model and Group Size",
        "",
        md_table(by_model_k),
        "",
        "## Interpretation Rule",
        "",
        "A full selective-steering pass requires the patient-bootstrap lower confidence bounds for both ExcessSelectivity and stable WBIImprovement to exceed zero. Difference-positive but ratio-unstable or ratio-uncorroborated cells are reported separately.",
    ]
    (args.root / "topk_v2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    status = {
        "expected": int(len(manifest)),
        "completed": int(len(completed)),
        "missing": int(len(missing)),
        "k_unavailable": int(len(unavailable)),
    }
    (args.root / "topk_v2_status.json").write_text(
        json.dumps(status, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(status))


if __name__ == "__main__":
    main()
