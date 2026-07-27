#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = [
    ("CSFM", "csfm", "csfm_cu118_commons"),
    ("ECG-FM", "ecg_fm", "ecg_fm_cu118_commons"),
    ("ECG-JEPA", "ecg_jepa", "ecg_jepa_cu118_commons"),
    ("ST-MEM", "st_mem", "st_mem_cu118_commons"),
    ("HuBERT-ECG", "hubert_ecg", "hubert_ecg_cu118_commons"),
    ("CARDIAC-FM", "cardiac_fm", "cardiac_fm_cu118_commons"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified continuation-erasure candidate panel.")
    parser.add_argument("--analysis-root", default=Path("results/analysis"), type=Path)
    parser.add_argument("--out-dir", default=Path("results/analysis/continuation_panel"), type=Path)
    parser.add_argument("--top-per-task", default=3, type=int)
    parser.add_argument("--top-per-family", default=2, type=int)
    parser.add_argument("--min-adj-drop", default=0.0, type=float)
    parser.add_argument("--max-per-model", default=40, type=int)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def layer_from_feature(feature: str) -> int:
    if not feature.startswith("layer_") or "_mean" not in feature:
        raise ValueError(f"cannot parse layer from feature {feature!r}")
    return int(feature.split("_")[1])


def score(row: dict[str, str]) -> float:
    return float(row["delta_auroc_minus_random"])


def add_rows(selected: dict[tuple[str, str, str, str], dict[str, str]], rows: list[dict[str, str]]) -> None:
    for row in rows:
        key = (row["model_key"], row["concept_id"], row["task_id"], row["feature"])
        previous = selected.get(key)
        if previous is None or score(row) > score(previous):
            selected[key] = row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    panel_rows: list[dict[str, str]] = []
    commands: list[str] = []
    report: dict[str, object] = {
        "top_per_task": args.top_per_task,
        "top_per_family": args.top_per_family,
        "min_adj_drop": args.min_adj_drop,
        "max_per_model": args.max_per_model,
        "models": {},
    }

    for display, model_key, suffix in MODELS:
        screen_path = args.analysis_root / suffix / "linear_erasure_screen.csv"
        rows = read_csv(screen_path)
        rows = [
            {**row, "model": display, "model_key": model_key, "suffix": suffix}
            for row in rows
            if score(row) > args.min_adj_drop and row["feature"].startswith("layer_")
        ]
        by_task: dict[str, list[dict[str, str]]] = {}
        by_family: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_task.setdefault(row["task_id"], []).append(row)
            by_family.setdefault(row["family"], []).append(row)

        selected: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for subset in by_task.values():
            add_rows(selected, sorted(subset, key=score, reverse=True)[: args.top_per_task])
        for subset in by_family.values():
            add_rows(selected, sorted(subset, key=score, reverse=True)[: args.top_per_family])

        model_rows = sorted(selected.values(), key=score, reverse=True)[: args.max_per_model]
        report["models"][display] = {
            "screen_rows_positive": len(rows),
            "selected_rows": len(model_rows),
        }
        for row in model_rows:
            layer = layer_from_feature(row["feature"])
            out_json = (
                args.analysis_root
                / suffix
                / f"continuation_erase_{row['concept_id']}_to_{row['task_id']}_layer{layer:02d}.json"
            )
            panel_row = {
                "model": display,
                "model_key": model_key,
                "suffix": suffix,
                "concept_id": row["concept_id"],
                "family": row["family"],
                "task_id": row["task_id"],
                "feature": row["feature"],
                "layer": str(layer),
                "linear_base_auroc": row["base_auroc"],
                "linear_delta_auroc": row["delta_auroc"],
                "linear_delta_auroc_minus_random": row["delta_auroc_minus_random"],
                "out_json": str(out_json),
            }
            panel_rows.append(panel_row)
            commands.append(
                " ".join(
                    [
                        "scripts/run_one_continuation_candidate.sh",
                        panel_row["model_key"],
                        panel_row["suffix"],
                        panel_row["concept_id"],
                        panel_row["task_id"],
                        panel_row["layer"],
                    ]
                )
            )

    fields = [
        "model",
        "model_key",
        "suffix",
        "concept_id",
        "family",
        "task_id",
        "feature",
        "layer",
        "linear_base_auroc",
        "linear_delta_auroc",
        "linear_delta_auroc_minus_random",
        "out_json",
    ]
    with (args.out_dir / "candidate_panel.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(panel_rows)
    (args.out_dir / "all_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    report["total_selected"] = len(panel_rows)
    (args.out_dir / "candidate_panel_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
