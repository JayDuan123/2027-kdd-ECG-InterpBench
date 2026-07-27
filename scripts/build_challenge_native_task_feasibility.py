#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ROOT = Path("/rhf/allocations/wq8/yd68/challenge-2021")
DEFAULT_OUT_DIR = ROOT / "results" / "multicohort"


COHORT_META = {
    "Chapman-F": CHALLENGE_ROOT / "chapman_shaoxing_meta.csv",
    "CPSC-F": CHALLENGE_ROOT / "cpsc_2018_meta.csv",
    "Ningbo-F": CHALLENGE_ROOT / "ningbo_meta.csv",
}


# Code groups use the scored SNOMED CT mapping from the PhysioNet/CinC 2021
# evaluation repository. They are task labels only, never measurement concepts.
TASKS = {
    "af_rhythm_native": {
        "label_source": "native Challenge 2021 SNOMED labels",
        "label_independence_tier": "secondary_semi_independent",
        "task_measurement_family_covered": "rate/rhythm covered by RR/rate waveform concepts",
        "codes": {"164889003", "164890007"},
    },
    "bbb_conduction_native": {
        "label_source": "native Challenge 2021 SNOMED labels",
        "label_independence_tier": "secondary_semi_independent",
        "task_measurement_family_covered": "conduction covered by QRS/PR/QT-like waveform concepts if G3 permits",
        "codes": {
            "6374002",
            "733534002",
            "713427006",
            "713426002",
            "164909002",
            "59118001",
            "698252002",
            "270492004",
            "164947007",
        },
    },
    "qt_interval_native": {
        "label_source": "native Challenge 2021 SNOMED labels",
        "label_independence_tier": "secondary_semi_independent",
        "task_measurement_family_covered": "QT/QTc-like interval covered if waveform QT-like passes G3",
        "codes": {"111975006"},
    },
    "st_t_abnormal_native": {
        "label_source": "native Challenge 2021 SNOMED labels",
        "label_independence_tier": "secondary_semi_independent",
        "task_measurement_family_covered": "ST/T covered by waveform-derived ST/T concepts if G3 permits",
        "codes": {"164934002", "59931005"},
    },
}


FEASIBILITY_FIELDS = [
    "cohort",
    "task",
    "label_source",
    "label_independence_tier",
    "positive_count",
    "negative_count",
    "split_unit",
    "eligible_for_auroc",
    "task_measurement_family_covered",
    "exclusion_reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_codes(value: str) -> set[str]:
    if not value:
        return set()
    return {code.strip() for code in value.replace(",", ";").split(";") if code.strip()}


def build_label_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cohort, path in COHORT_META.items():
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                codes = parse_codes(row.get("snomed_codes", ""))
                out = {
                    "cohort": cohort,
                    "record_id": row.get("record_id", ""),
                    "age": row.get("age", ""),
                    "sex": row.get("sex", ""),
                    "snomed_codes": ";".join(sorted(codes)),
                }
                for task, spec in TASKS.items():
                    out[task] = "1" if codes.intersection(spec["codes"]) else "0"
                rows.append(out)
    return rows


def build_feasibility(label_rows: list[dict[str, str]], min_pos: int, min_neg: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for cohort in COHORT_META:
        subset = [row for row in label_rows if row["cohort"] == cohort]
        n = len(subset)
        for task, spec in TASKS.items():
            pos = sum(row[task] == "1" for row in subset)
            neg = n - pos
            eligible = pos >= min_pos and neg >= min_neg
            reason = "" if eligible else f"insufficient positives/negatives for AUROC gate: pos={pos}, neg={neg}"
            out.append(
                {
                    "cohort": cohort,
                    "task": task,
                    "label_source": spec["label_source"],
                    "label_independence_tier": spec["label_independence_tier"],
                    "positive_count": str(pos),
                    "negative_count": str(neg),
                    "split_unit": "record_level_native_challenge_no_patient_id",
                    "eligible_for_auroc": "true" if eligible else "false",
                    "task_measurement_family_covered": spec["task_measurement_family_covered"],
                    "exclusion_reason": reason,
                }
            )
    return out


def merge_external_task_feasibility(out_dir: Path, challenge_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    path = out_dir / "external_task_feasibility.csv"
    existing = read_csv(path)
    challenge_cohorts = set(COHORT_META)
    kept = [row for row in existing if row.get("cohort") not in challenge_cohorts]
    merged = kept + challenge_rows
    extra_fields = []
    for row in merged:
        for field in row:
            if field not in FEASIBILITY_FIELDS and field not in extra_fields:
                extra_fields.append(field)
    write_csv_atomic(path, merged, FEASIBILITY_FIELDS + extra_fields)
    return merged


def render_report(feasibility: list[dict[str, str]]) -> str:
    lines = [
        "# Challenge Native Task Feasibility",
        "",
        "This G4 artifact summarizes native Challenge-style SNOMED label feasibility for Track F cohorts. These labels are tasks only; they are never used as measurement concepts.",
        "",
        "| Cohort | Task | Positives | Negatives | Eligible |",
        "|---|---|---:|---:|---|",
    ]
    for row in feasibility:
        lines.append(
            f"| {row['cohort']} | {row['task']} | {row['positive_count']} | "
            f"{row['negative_count']} | {row['eligible_for_auroc']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "- These labels are native Challenge SNOMED labels and are less independent than MIMIC ICD-linked labels.",
            "- They support Track F robustness checks, not primary Track V conclusions.",
            "- Record-level split is marked because patient identifiers are not available in these metadata tables.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Chapman/CPSC/Ningbo native-label G4 feasibility artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-pos", type=int, default=50)
    parser.add_argument("--min-neg", type=int, default=50)
    args = parser.parse_args()

    label_rows = build_label_rows()
    label_fields = ["cohort", "record_id", "age", "sex", "snomed_codes", *TASKS.keys()]
    feasibility = build_feasibility(label_rows, args.min_pos, args.min_neg)
    write_csv_atomic(args.out_dir / "challenge_native_label_matrix.csv", label_rows, label_fields)
    write_csv_atomic(args.out_dir / "challenge_native_task_feasibility.csv", feasibility, FEASIBILITY_FIELDS)
    merge_external_task_feasibility(args.out_dir, feasibility)
    report_path = args.out_dir / "challenge_native_task_feasibility_report.md"
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(render_report(feasibility))
    os.replace(tmp, report_path)
    print(f"wrote: {args.out_dir / 'challenge_native_label_matrix.csv'}")
    print(f"wrote: {args.out_dir / 'challenge_native_task_feasibility.csv'}")
    print(f"updated: {args.out_dir / 'external_task_feasibility.csv'}")
    print(f"wrote: {report_path}")
    for row in feasibility:
        print(row["cohort"], row["task"], row["positive_count"], row["negative_count"], row["eligible_for_auroc"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
