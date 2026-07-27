#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.config import CONCEPTS_CSV, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize probe scores into concept/layer/family atlas tables.")
    parser.add_argument("--probe-scores", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--encoded-threshold", type=float, default=0.1)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    concepts = read_csv(CONCEPTS_CSV)
    family_by_concept = {row["concept_id"]: row["family"] for row in concepts}
    display_by_concept = {row["concept_id"]: row["display_name"] for row in concepts}

    rows = read_csv(args.probe_scores)
    for row in rows:
        for key in ["val_r2", "test_r2", "val_r2_shuffled", "val_r2_gaussian"]:
            row[key] = float(row[key])
        row["family"] = family_by_concept.get(row["concept_id"], "")
        row["display_name"] = display_by_concept.get(row["concept_id"], row["concept_id"])

    best: dict[str, dict[str, object]] = {}
    for row in rows:
        concept_id = str(row["concept_id"])
        if concept_id not in best or float(row["val_r2"]) > float(best[concept_id]["val_r2"]):
            best[concept_id] = row

    peak_rows: list[dict[str, object]] = []
    for concept_id, row in sorted(best.items(), key=lambda item: float(item[1]["val_r2"]), reverse=True):
        encoded = (
            float(row["val_r2"]) >= args.encoded_threshold
            and float(row["val_r2"]) > float(row["val_r2_shuffled"])
            and float(row["val_r2"]) > float(row["val_r2_gaussian"])
        )
        peak_rows.append(
            {
                "concept_id": concept_id,
                "display_name": row["display_name"],
                "family": row["family"],
                "peak_feature": row["feature"],
                "peak_val_r2": f"{float(row['val_r2']):.8g}",
                "test_r2_at_peak": f"{float(row['test_r2']):.8g}",
                "val_r2_shuffled_at_peak": f"{float(row['val_r2_shuffled']):.8g}",
                "val_r2_gaussian_at_peak": f"{float(row['val_r2_gaussian']):.8g}",
                "encoded": "yes" if encoded else "no",
            }
        )

    layer_rows: list[dict[str, object]] = []
    for feature in sorted({str(row["feature"]) for row in rows}, key=lambda x: (-1 if x == "pooled" else int(x.split("_")[1]))):
        subset = [row for row in rows if row["feature"] == feature]
        vals = [float(row["val_r2"]) for row in subset]
        tests = [float(row["test_r2"]) for row in subset]
        layer_rows.append(
            {
                "feature": feature,
                "n_scores": len(subset),
                "mean_val_r2": f"{statistics.mean(vals):.8g}",
                "median_val_r2": f"{statistics.median(vals):.8g}",
                "mean_test_r2": f"{statistics.mean(tests):.8g}",
                "max_val_r2": f"{max(vals):.8g}",
            }
        )

    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in peak_rows:
        by_family[str(row["family"])].append(row)
    family_rows: list[dict[str, object]] = []
    for family, subset in sorted(by_family.items()):
        vals = [float(row["peak_val_r2"]) for row in subset]
        tests = [float(row["test_r2_at_peak"]) for row in subset]
        family_rows.append(
            {
                "family": family,
                "n_concepts": len(subset),
                "encoded_count": sum(row["encoded"] == "yes" for row in subset),
                "strong_count_val_r2_ge_0_5": sum(float(row["peak_val_r2"]) >= 0.5 for row in subset),
                "mean_peak_val_r2": f"{statistics.mean(vals):.8g}",
                "median_peak_val_r2": f"{statistics.median(vals):.8g}",
                "mean_test_r2_at_peak": f"{statistics.mean(tests):.8g}",
            }
        )

    fields_peak = [
        "concept_id",
        "display_name",
        "family",
        "peak_feature",
        "peak_val_r2",
        "test_r2_at_peak",
        "val_r2_shuffled_at_peak",
        "val_r2_gaussian_at_peak",
        "encoded",
    ]
    fields_layer = ["feature", "n_scores", "mean_val_r2", "median_val_r2", "mean_test_r2", "max_val_r2"]
    fields_family = [
        "family",
        "n_concepts",
        "encoded_count",
        "strong_count_val_r2_ge_0_5",
        "mean_peak_val_r2",
        "median_peak_val_r2",
        "mean_test_r2_at_peak",
    ]
    write_csv(args.out_dir / "probe_peak_by_concept.csv", peak_rows, fields_peak)
    write_csv(args.out_dir / "probe_layer_summary.csv", layer_rows, fields_layer)
    write_csv(args.out_dir / "probe_family_summary.csv", family_rows, fields_family)

    peak_layer_counts = Counter(row["peak_feature"] for row in peak_rows)
    report = {
        "probe_scores": str(args.probe_scores),
        "out_dir": str(args.out_dir),
        "n_scores": len(rows),
        "n_concepts": len(peak_rows),
        "encoded_threshold": args.encoded_threshold,
        "encoded_count": sum(row["encoded"] == "yes" for row in peak_rows),
        "peak_layer_counts": dict(peak_layer_counts),
        "top_concepts": peak_rows[:15],
        "weakest_concepts": sorted(peak_rows, key=lambda row: float(row["peak_val_r2"]))[:10],
    }
    (args.out_dir / "probe_atlas_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# CSFM Probe Atlas",
        "",
        f"- concepts: {len(peak_rows)}",
        f"- encoded concepts: {report['encoded_count']}/{len(peak_rows)} at val R2 >= {args.encoded_threshold}",
        f"- scores: {len(rows)}",
        "",
        "## Peak Layer Counts",
        "",
    ]
    for feature, count in peak_layer_counts.most_common():
        lines.append(f"- {feature}: {count}")
    lines.extend(["", "## Top Concepts", ""])
    for row in peak_rows[:15]:
        lines.append(
            f"- {row['concept_id']} ({row['family']}, {row['peak_feature']}): "
            f"val R2 {row['peak_val_r2']}, test R2 {row['test_r2_at_peak']}"
        )
    (args.out_dir / "probe_atlas_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
