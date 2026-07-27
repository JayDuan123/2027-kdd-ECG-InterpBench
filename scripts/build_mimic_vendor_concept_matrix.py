#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIMIC_ECG_ROOT = Path("/rhf/allocations/wq8/mimic_data/mimic-iv-ecg")
MIMIC_MEASUREMENTS = MIMIC_ECG_ROOT / "machine_measurements.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "multicohort"


CONCEPT_COLUMNS = [
    "hr_ventricular",
    "hr_atrial",
    "rr_mean",
    "pr_interval",
    "pq_interval",
    "p_duration_global",
    "qrs_duration",
    "qt_interval",
    "qtc_bazett",
    "qtc_fridericia",
    "qtc_framingham",
    "p_axis_front",
    "qrs_axis_front",
    "t_axis_front",
    "qrst_angle",
]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def fmt(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.8g}"


def diff_ms(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    value = a - b
    return value if math.isfinite(value) else None


def wrapped_angle(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


def derive_concepts(row: dict[str, str]) -> dict[str, str]:
    rr = parse_float(row.get("rr_interval"))
    p_on = parse_float(row.get("p_onset"))
    p_end = parse_float(row.get("p_end"))
    qrs_on = parse_float(row.get("qrs_onset"))
    qrs_end = parse_float(row.get("qrs_end"))
    t_end = parse_float(row.get("t_end"))
    p_axis = parse_float(row.get("p_axis"))
    qrs_axis = parse_float(row.get("qrs_axis"))
    t_axis = parse_float(row.get("t_axis"))

    hr = 60000.0 / rr if rr and rr > 0 else None
    pr = diff_ms(qrs_on, p_on)
    p_dur = diff_ms(p_end, p_on)
    qrs_dur = diff_ms(qrs_end, qrs_on)
    qt = diff_ms(t_end, qrs_on)
    rr_sec = rr / 1000.0 if rr and rr > 0 else None

    qtc_b = qt / math.sqrt(rr_sec) if qt is not None and rr_sec and rr_sec > 0 else None
    qtc_f = qt / (rr_sec ** (1.0 / 3.0)) if qt is not None and rr_sec and rr_sec > 0 else None
    qtc_fh = qt + 154.0 * (1.0 - rr_sec) if qt is not None and rr_sec and rr_sec > 0 else None

    values = {
        "hr_ventricular": hr,
        "hr_atrial": hr,
        "rr_mean": rr,
        "pr_interval": pr,
        "pq_interval": pr,
        "p_duration_global": p_dur,
        "qrs_duration": qrs_dur,
        "qt_interval": qt,
        "qtc_bazett": qtc_b,
        "qtc_fridericia": qtc_f,
        "qtc_framingham": qtc_fh,
        "p_axis_front": p_axis,
        "qrs_axis_front": qrs_axis,
        "t_axis_front": t_axis,
        "qrst_angle": wrapped_angle(qrs_axis, t_axis),
    }
    return {key: fmt(values[key]) for key in CONCEPT_COLUMNS}


def plausible_value(concept: str, value: float) -> bool:
    if concept in {"hr_ventricular", "hr_atrial"}:
        return 20 <= value <= 300
    if concept == "rr_mean":
        return 200 <= value <= 3000
    if concept in {"pr_interval", "pq_interval"}:
        return 50 <= value <= 400
    if concept == "p_duration_global":
        return 20 <= value <= 250
    if concept == "qrs_duration":
        return 30 <= value <= 250
    if concept in {"qt_interval", "qtc_bazett", "qtc_fridericia", "qtc_framingham"}:
        return 150 <= value <= 800
    if concept in {"p_axis_front", "qrs_axis_front", "t_axis_front"}:
        return -180 <= value <= 180
    if concept == "qrst_angle":
        return 0 <= value <= 180
    return True


def update_summary(summary: dict[str, dict[str, int]], concepts: dict[str, str]) -> None:
    for concept in CONCEPT_COLUMNS:
        value = parse_float(concepts.get(concept))
        if value is None:
            summary[concept]["missing"] += 1
        else:
            summary[concept]["nonmissing"] += 1
            if plausible_value(concept, value):
                summary[concept]["plausible"] += 1
            else:
                summary[concept]["implausible"] += 1


def write_summary(path: Path, summary: dict[str, dict[str, int]], total_rows: int) -> None:
    rows = []
    for concept in CONCEPT_COLUMNS:
        s = summary[concept]
        nonmissing = s["nonmissing"]
        rows.append(
            {
                "concept": concept,
                "rows": str(total_rows),
                "nonmissing": str(nonmissing),
                "missing": str(s["missing"]),
                "nonmissing_frac": f"{nonmissing / total_rows:.8g}" if total_rows else "",
                "plausible": str(s["plausible"]),
                "implausible": str(s["implausible"]),
                "plausible_frac_among_nonmissing": f"{s['plausible'] / nonmissing:.8g}" if nonmissing else "",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary_path: Path, total_rows: int, max_records: int | None) -> str:
    mode = "smoke" if max_records is not None else "full"
    lines = [
        "# MIMIC-V Vendor Concept Matrix Report",
        "",
        f"- Mode: {mode}",
        f"- Rows processed: {total_rows}",
        f"- Source: `{MIMIC_MEASUREMENTS}`",
        "- Measurement source discipline: report text columns are not used.",
        "- Scope: interval/rate/axis concepts only; ST/amplitude morphology is excluded from MIMIC-V.",
        "",
        f"Summary CSV: `{summary_path}`",
        "",
    ]
    return "\n".join(lines)


def build_matrix(out: Path, summary_out: Path, report_out: Path, max_records: int | None) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        concept: {"nonmissing": 0, "missing": 0, "plausible": 0, "implausible": 0}
        for concept in CONCEPT_COLUMNS
    }
    total_rows = 0
    with MIMIC_MEASUREMENTS.open(newline="") as f_in, out.open("w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fields = ["subject_id", "study_id", "ecg_time", *CONCEPT_COLUMNS]
        writer = csv.DictWriter(f_out, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            if max_records is not None and total_rows >= max_records:
                break
            concepts = derive_concepts(row)
            update_summary(summary, concepts)
            writer.writerow(
                {
                    "subject_id": row.get("subject_id", ""),
                    "study_id": row.get("study_id", ""),
                    "ecg_time": row.get("ecg_time", ""),
                    **concepts,
                }
            )
            total_rows += 1
    write_summary(summary_out, summary, total_rows)
    report_out.write_text(render_report(summary_out, total_rows, max_records))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MIMIC-V interval/rate/axis vendor concept matrix.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR / "mimic_vendor_concepts.csv")
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_OUT_DIR / "mimic_vendor_concept_summary.csv")
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT_DIR / "mimic_vendor_concept_report.md")
    parser.add_argument("--max-records", type=int, default=None, help="Optional smoke-test row cap.")
    args = parser.parse_args()
    build_matrix(args.out, args.summary_out, args.report_out, args.max_records)
    print(f"wrote: {args.out}")
    print(f"wrote: {args.summary_out}")
    print(f"wrote: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
