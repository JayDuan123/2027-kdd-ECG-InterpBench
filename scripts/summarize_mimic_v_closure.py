#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results" / "multicohort" / "mimic_v_closure"
PRIMARY_TASKS = {"af_rhythm_icd", "bbb_conduction_icd"}
SENSITIVITY_TASKS = {"qt_interval_icd"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def summarize(scores: list[dict[str, str]]) -> list[dict[str, str]]:
    by_task_block = {(row["task"], row["block"]): row for row in scores}
    rows = []
    tasks = sorted({row["task"] for row in scores})
    for task in tasks:
        common = by_task_block.get((task, "Bcommon_mimic_v_all15"))
        minimal = by_task_block.get((task, "Bminimal_mimic_v_9"))
        brand = by_task_block.get((task, "Brand_gaussian_dim15"))
        if common is None or minimal is None or brand is None:
            continue
        common_auc = parse_float(common["test_auroc"])
        minimal_auc = parse_float(minimal["test_auroc"])
        brand_auc = parse_float(brand["test_auroc"])
        delta_common = common_auc - brand_auc if common_auc is not None and brand_auc is not None else None
        delta_minimal = minimal_auc - brand_auc if minimal_auc is not None and brand_auc is not None else None
        if task in PRIMARY_TASKS:
            task_scope = "primary"
            interpretation = "MIMIC-V primary interval/rate/axis task"
        elif task in SENSITIVITY_TASKS:
            task_scope = "sensitivity_label_measurement_proximal"
            interpretation = "sensitivity only; QT label is measurement-proximal to QT/QTc concepts"
        else:
            task_scope = "out_of_scope_missing_measurement_family"
            interpretation = "audit only; MIMIC-V lacks required ST/amplitude morphology"
        rows.append(
            {
                "task": task,
                "task_scope": task_scope,
                "n_test": common["n_test"],
                "brand_test_auroc": brand["test_auroc"],
                "bminimal_test_auroc": minimal["test_auroc"],
                "bcommon_test_auroc": common["test_auroc"],
                "bcommon_minus_brand_auroc": "" if delta_common is None else f"{delta_common:.8g}",
                "bminimal_minus_brand_auroc": "" if delta_minimal is None else f"{delta_minimal:.8g}",
                "bcommon_test_auprc": common["test_auprc"],
                "brand_test_auprc": brand["test_auprc"],
                "interpretation": interpretation,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(rows: list[dict[str, str]]) -> str:
    lines = [
        "# MIMIC-V Closure Summary",
        "",
        "MIMIC-V uses vendor interval/rate/axis concepts and ICD-linked labels. It does not test ST/amplitude morphology transfer.",
        "",
        "Primary tasks are AF/rhythm and BBB/conduction. QT-related ICD labels are reported as sensitivity only because the label is measurement-proximal to QT/QTc concepts. MI/ischemia and hypertrophy are audit-only because their key measurement families are absent from MIMIC-V.",
        "",
        "| Task | Scope | Brand AUROC | Bminimal AUROC | Bcommon AUROC | Bcommon-Brand |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['task_scope']} | {row['brand_test_auroc']} | "
            f"{row['bminimal_test_auroc']} | {row['bcommon_test_auroc']} | "
            f"{row['bcommon_minus_brand_auroc']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation discipline:",
            "",
            "- `Bcommon-Brand` is a measurement-vs-random improvement, not a ClosureRatio against an FM head.",
            "- ClosureRatio is not reported because no external frozen-FM head score is included in this MIMIC-V run.",
            "- QT-related ICD rows are sensitivity-only because the label is measurement-proximal to QT/QTc concepts.",
            "- Out-of-scope rows must not be used to claim failure or success of MI/HYP measurement saturation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MIMIC-V closure outputs.")
    parser.add_argument("--closure-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    scores = read_csv(args.closure_dir / "mimic_v_closure_scores.csv")
    rows = summarize(scores)
    write_csv(args.closure_dir / "mimic_v_closure_summary.csv", rows)
    (args.closure_dir / "mimic_v_closure_summary.md").write_text(render_report(rows))
    print(f"wrote: {args.closure_dir / 'mimic_v_closure_summary.csv'}")
    print(f"wrote: {args.closure_dir / 'mimic_v_closure_summary.md'}")
    for row in rows:
        print(row["task"], row["task_scope"], row["bcommon_test_auroc"], row["bcommon_minus_brand_auroc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
