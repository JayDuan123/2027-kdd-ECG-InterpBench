#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multicohort import (  # noqa: E402
    DEFAULT_OUT_DIR,
    ICD_TASK_RULES,
    MIMIC_RECORD_LIST_CSV,
    find_hadm_for_ecg,
    parse_time,
    read_admissions_by_subject,
    read_task_positive_hadm_ids,
    write_csv,
)


TASKS = [
    "af_rhythm_icd",
    "bbb_conduction_icd",
    "qt_interval_icd",
    "mi_ischemia_icd",
    "hypertrophy_icd",
]


def build_label_matrix(max_records: int | None = None) -> tuple[list[dict[str, str]], dict[str, int]]:
    admissions_by_subject = read_admissions_by_subject()
    task_positive_hadm_ids = read_task_positive_hadm_ids()
    rows: list[dict[str, str]] = []
    records_seen = 0
    records_linked = 0
    linked_hadm_ids: set[str] = set()
    linked_subject_ids: set[str] = set()

    with MIMIC_RECORD_LIST_CSV.open(newline="") as f:
        for record in csv.DictReader(f):
            if max_records is not None and records_seen >= max_records:
                break
            records_seen += 1
            subject_id = record.get("subject_id", "")
            ecg_time_raw = record.get("ecg_time", "")
            ecg_time = parse_time(ecg_time_raw)
            if not subject_id or ecg_time is None:
                continue
            hadm_id = find_hadm_for_ecg(subject_id, ecg_time, admissions_by_subject)
            if hadm_id is None:
                continue
            records_linked += 1
            linked_hadm_ids.add(hadm_id)
            linked_subject_ids.add(subject_id)
            out = {
                "subject_id": subject_id,
                "study_id": record.get("study_id", ""),
                "ecg_time": ecg_time_raw,
                "hadm_id": hadm_id,
            }
            for task in TASKS:
                out[task] = "1" if hadm_id in task_positive_hadm_ids[task] else "0"
            rows.append(out)

    stats = {
        "records_seen": records_seen,
        "records_linked_to_admission": records_linked,
        "linked_hadm_count": len(linked_hadm_ids),
        "linked_subject_count": len(linked_subject_ids),
    }
    for task in TASKS:
        stats[f"{task}_positive"] = sum(int(row[task]) for row in rows)
        stats[f"{task}_negative"] = len(rows) - stats[f"{task}_positive"]
    return rows, stats


def write_report(path: Path, stats: dict[str, int], max_records: int | None) -> None:
    mode = "smoke" if max_records is not None else "full"
    lines = [
        "# MIMIC ICD Label Matrix Report",
        "",
        f"- Mode: {mode}",
        f"- Records seen: {stats['records_seen']}",
        f"- Records linked to admission: {stats['records_linked_to_admission']}",
        f"- Linked admissions: {stats['linked_hadm_count']}",
        f"- Linked subjects: {stats['linked_subject_count']}",
        "",
        "## Task Counts",
        "",
        "| Task | Positive | Negative | Track V primary? |",
        "|---|---:|---:|---|",
    ]
    for task in TASKS:
        primary = "yes" if ICD_TASK_RULES[task]["primary_track_v"] else "no"
        lines.append(
            f"| {task} | {stats[f'{task}_positive']} | "
            f"{stats[f'{task}_negative']} | {primary} |"
        )
    lines.extend(
        [
            "",
            "QT-related ICD labels are retained as sensitivity-only because the long-QT/QT-prolongation label is measurement-proximal to QT/QTc concepts.",
            "",
            "MI/ischemia and hypertrophy labels are retained for audit/sensitivity but are not MIMIC-V primary closure tasks because MIMIC-V lacks ST/T and amplitude morphology measurements.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIMIC ICD-linked per-study label matrix.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "mimic_icd_label_matrix.csv")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT_DIR / "mimic_icd_label_matrix_report.md")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    rows, stats = build_label_matrix(max_records=args.max_records)
    write_csv(args.out, rows)
    write_report(args.report_out, stats, args.max_records)
    print(f"wrote: {args.out}")
    print(f"wrote: {args.report_out}")
    print(f"rows: {len(rows)}")
    print(f"records_seen: {stats['records_seen']}")
    print(f"records_linked_to_admission: {stats['records_linked_to_admission']}")
    for task in TASKS:
        print(task, stats[f"{task}_positive"], stats[f"{task}_negative"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
