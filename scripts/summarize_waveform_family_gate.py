#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "results" / "multicohort"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def concept_status(ratio: float | None, n_pairs: int, minimum_ratio: float) -> tuple[str, str]:
    if ratio is None or n_pairs < 10:
        return "not_assessable", "too few pairs or missing correlation"
    if ratio >= minimum_ratio:
        return "smoke_pass", f"extractor-vendor ratio {ratio:.3g} >= {minimum_ratio:.3g}"
    if ratio >= 0.5 * minimum_ratio:
        return "partial", f"extractor-vendor ratio {ratio:.3g} below {minimum_ratio:.3g}"
    return "smoke_fail", f"extractor-vendor ratio {ratio:.3g} far below {minimum_ratio:.3g}"


def summarize_concepts(comparison_rows: list[dict[str, str]], minimum_ratio: float) -> list[dict[str, str]]:
    rows = []
    for row in comparison_rows:
        ratio = parse_float(row.get("corr_to_ceiling_ratio"))
        n_pairs = int(row.get("n_pairs") or "0")
        status, reason = concept_status(ratio, n_pairs, minimum_ratio)
        rows.append(
            {
                "concept": row["concept"],
                "family": row["family"],
                "n_pairs": str(n_pairs),
                "corr_extractor_12sl": row["corr_extractor_12sl"],
                "vendor_ceiling_corr_12sl_unig": row["vendor_ceiling_corr_12sl_unig"],
                "corr_to_ceiling_ratio": row["corr_to_ceiling_ratio"],
                "concept_gate_status": status,
                "gate_reason": reason,
                "scope_note": "20-record PTB-XL smoke; not full Track F validation",
            }
        )
    return rows


def family_status(concept_rows: list[dict[str, str]], smoke_summary: list[dict[str, str]]) -> list[dict[str, str]]:
    family_to_concepts: dict[str, list[dict[str, str]]] = {}
    for row in concept_rows:
        family_to_concepts.setdefault(row["family"], []).append(row)

    cohort_summary = {
        row["cohort"]: row
        for row in smoke_summary
    }
    out = []
    for family, rows in sorted(family_to_concepts.items()):
        statuses = [row["concept_gate_status"] for row in rows]
        pass_count = statuses.count("smoke_pass")
        partial_count = statuses.count("partial")
        fail_count = statuses.count("smoke_fail")
        if fail_count == 0 and partial_count == 0 and pass_count == len(rows):
            gate = "smoke_pass"
            reason = "all assessed concepts pass small-sample ratio gate"
        elif pass_count > 0 or partial_count > 0:
            gate = "partial"
            reason = "some concepts pass/partial but at least one concept fails or remains weak"
        else:
            gate = "smoke_fail"
            reason = "no assessed concepts pass small-sample ratio gate"

        for cohort, summary in sorted(cohort_summary.items()):
            if family == "INTERVAL":
                record_family_count = summary["interval_pass_count"]
            elif family == "AMPLITUDE":
                record_family_count = summary["amplitude_pass_count"]
            elif family == "ST_T":
                record_family_count = summary["st_t_pass_count"]
            else:
                record_family_count = "0"
            out.append(
                {
                    "family": family,
                    "cohort": cohort,
                    "records": summary["records"],
                    "extractor_success": summary["extractor_success"],
                    "record_family_pass_count": record_family_count,
                    "concepts_assessed": "|".join(row["concept"] for row in rows),
                    "concept_statuses": "|".join(f"{row['concept']}:{row['concept_gate_status']}" for row in rows),
                    "family_gate_status": gate,
                    "gate_reason": reason,
                    "vendor_comparison_scope": "PTB-XL small-sample only; external cohorts assessed for extractor run/missingness only",
                }
            )

    out.append(
        {
            "family": "AXIS",
            "cohort": "all",
            "records": "",
            "extractor_success": "",
            "record_family_pass_count": "0",
            "concepts_assessed": "",
            "concept_statuses": "",
            "family_gate_status": "not_assessed",
            "gate_reason": "NeuroKit2 smoke does not estimate P/QRS/T axes; Track V axis is available only from MIMIC vendor measurements.",
            "vendor_comparison_scope": "not_applicable",
        }
    )
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(concepts: list[dict[str, str]], families: list[dict[str, str]]) -> str:
    lines = [
        "# Waveform Family Gate Summary",
        "",
        "This summarizes G3 Track F smoke results at concept and family level. It is intentionally conservative: a family may be partial even when waveform loading succeeds, if PTB-XL extractor-vendor agreement is weak for some concepts.",
        "",
        "## Concept Gates",
        "",
        "| Concept | Family | N | extractor-vendor r | ceiling r | ratio | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in concepts:
        lines.append(
            f"| {row['concept']} | {row['family']} | {row['n_pairs']} | "
            f"{row['corr_extractor_12sl']} | {row['vendor_ceiling_corr_12sl_unig']} | "
            f"{row['corr_to_ceiling_ratio']} | {row['concept_gate_status']} |"
        )
    lines.extend(["", "## Family Gates", "", "| Family | Status | Reason |", "|---|---|---|"])
    seen = set()
    for row in families:
        key = (row["family"], row["family_gate_status"], row["gate_reason"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"| {row['family']} | {row['family_gate_status']} | {row['gate_reason']} |")
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "- `smoke_pass` is not full external validation; it only allows scaling to a larger Track F extraction job.",
            "- `partial` families must remain concept-specific in downstream claims.",
            "- Axis remains unassessed by waveform smoke and cannot be claimed from NeuroKit2 extraction here.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize G3 waveform smoke into concept/family gate artifacts.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--minimum-ratio", type=float, default=0.75)
    args = parser.parse_args()

    comparison = read_csv(args.out_dir / "ptbxl_extractor_vendor_smoke.csv")
    smoke_summary = read_csv(args.out_dir / "waveform_extraction_smoke_summary.csv")
    concept_rows = summarize_concepts(comparison, args.minimum_ratio)
    family_rows = family_status(concept_rows, smoke_summary)
    write_csv(args.out_dir / "waveform_concept_gate.csv", concept_rows)
    write_csv(args.out_dir / "waveform_family_gate.csv", family_rows)
    (args.out_dir / "waveform_family_gate_report.md").write_text(render_report(concept_rows, family_rows))
    print(f"wrote: {args.out_dir / 'waveform_concept_gate.csv'}")
    print(f"wrote: {args.out_dir / 'waveform_family_gate.csv'}")
    print(f"wrote: {args.out_dir / 'waveform_family_gate_report.md'}")
    for row in concept_rows:
        print(row["concept"], row["concept_gate_status"], row["gate_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
