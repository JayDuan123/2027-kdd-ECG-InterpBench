#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "multicohort" / "paper_ready_tables.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str, digits: int = 3) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    def escape_cell(value: str) -> str:
        return str(value).replace("\n", " ").replace("|", r"\|")

    lines = [
        "| " + " | ".join(escape_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(escape_cell(str(cell)) for cell in row) + " |")
    return lines


def section(title: str) -> list[str]:
    return ["", f"## {title}", ""]


def render_cohort_gate(root: Path) -> list[str]:
    rows = read_csv(root / "results" / "multicohort" / "cohort_gate_summary.csv")
    out = section("Table MC-1: Cohort Gate Summary")
    out.extend(
        md_table(
            [
                "Cohort track",
                "Vendor scope",
                "Label source",
                "Track status",
                "SAE transfer status",
            ],
            [
                [
                    r["cohort_track"],
                    r["vendor_measurement_scope"],
                    r["primary_label_source"],
                    r["track_status"],
                    r["sae_transfer_status"],
                ]
                for r in rows
            ],
        )
    )
    return out


def render_crosswalk(root: Path) -> list[str]:
    rows = read_csv(root / "results" / "multicohort" / "concept_crosswalk_track_v.csv")
    by_category = Counter(r["category"] for r in rows)
    by_family_category: dict[tuple[str, str], int] = Counter((r["family"], r["category"]) for r in rows)
    eligible = [r["ptbxl_concept"] for r in rows if r["category"] == "track_v_eligible"]

    out = section("Table MC-2: Concept Crosswalk Summary")
    out.extend(
        md_table(
            ["Category", "Concept count"],
            [[k, str(v)] for k, v in sorted(by_category.items())],
        )
    )
    out.extend(["", "Family-level crosswalk:", ""])
    family_rows = []
    for (family, category), count in sorted(by_family_category.items()):
        family_rows.append([family, category, str(count)])
    out.extend(md_table(["Family", "Category", "Concept count"], family_rows))
    out.extend(["", "Track V eligible concepts:", "", ", ".join(eligible)])
    return out


def render_task_feasibility(root: Path) -> list[str]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    main_path = root / "results" / "multicohort" / "external_task_feasibility.csv"
    fallback_path = root / "results" / "multicohort" / "challenge_native_task_feasibility.csv"
    paths = [main_path]
    main_rows = read_csv(main_path) if main_path.exists() else []
    has_native_rows = any(row.get("cohort", "").strip() != "MIMIC-V" for row in main_rows)
    if not has_native_rows and fallback_path.exists():
        paths.append(fallback_path)

    for path in paths:
        if path.exists():
            for row in read_csv(path):
                key = (row.get("cohort", "").strip().lower(), row.get("task", "").strip().lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

    out = section("Table MC-3: External Task Feasibility")
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["cohort"],
                r["task"],
                r["label_independence_tier"],
                r["positive_count"],
                r["negative_count"],
                r["split_unit"],
                r["eligible_for_auroc"],
                r["task_measurement_family_covered"],
            ]
        )
    out.extend(
        md_table(
            [
                "Cohort",
                "Task",
                "Label tier",
                "Positive",
                "Negative",
                "Split unit",
                "AUROC eligible",
                "Measurement family coverage",
            ],
            table_rows,
        )
    )
    return out


def render_mimic_v_closure(root: Path) -> list[str]:
    rows = read_csv(root / "results" / "multicohort" / "mimic_v_closure" / "mimic_v_closure_summary.csv")
    out = section("MIMIC-V Track V Closure")
    out.extend(
        md_table(
            [
                "Task",
                "Scope",
                "Brand AUROC",
                "Bminimal AUROC",
                "Bcommon AUROC",
                "Bcommon - Brand",
            ],
            [
                [
                    r["task"],
                    r["task_scope"],
                    fmt_float(r["brand_test_auroc"]),
                    fmt_float(r["bminimal_test_auroc"]),
                    fmt_float(r["bcommon_test_auroc"]),
                    fmt_float(r["bcommon_minus_brand_auroc"]),
                ]
                for r in rows
            ],
        )
    )
    out.extend(
        [
            "",
            "Interpretation discipline: this is a measurement-vs-random external closure baseline, not an FM ClosureRatio.",
            "Primary Track V tasks are AF/rhythm and BBB/conduction. QT-related ICD rows are sensitivity-only because the label is measurement-proximal to QT/QTc concepts.",
        ]
    )
    return out


def render_track_f_closure(root: Path) -> list[str]:
    rows = read_csv(root / "results" / "multicohort" / "track_f_closure" / "closure_transfer_track_f.csv")
    out = section("Track F Waveform-Derived Closure")
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r["cohort"],
                r["task"],
                r["task_scope"],
                r["concepts"],
                r["concept_gate_statuses"],
                r["n_test"],
                fmt_float(r["btrackf_test_auroc"]),
                fmt_float(r["brand_test_auroc"]),
                fmt_float(r["closure_gain_vs_brand_auroc"]),
                r["quality_note"],
            ]
        )
    out.extend(
        md_table(
            [
                "Cohort",
                "Task",
                "Scope",
                "Concepts",
                "Concept gates",
                "N test",
                "Track F AUROC",
                "Brand AUROC",
                "Gain",
                "Quality note",
            ],
            table_rows,
        )
    )
    return out


def render_sae_gate(root: Path) -> list[str]:
    rows = read_csv(root / "results" / "multicohort" / "external_sae" / "external_sae_recon_gate.csv")
    status_counts = Counter(r["recon_gate_status"] for r in rows)
    activation_counts = Counter(r["external_activation_status"] for r in rows)
    passes = sum(r["recon_gate_pass"] == "true" for r in rows)

    by_model_cohort: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_model_cohort[(r["model"], r["external_cohort"])].append(r)

    out = section("Figure/Table MC-SAE: External SAE Reconstruction Gate")
    out.extend(
        md_table(
            ["Metric", "Value"],
            [
                ["Rows", str(len(rows))],
                ["Recon-gate passes", str(passes)],
                ["Recon statuses", "; ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))],
                ["Activation statuses", "; ".join(f"{k}: {v}" for k, v in sorted(activation_counts.items()))],
            ],
        )
    )

    table_rows = []
    for (model, cohort), group in sorted(by_model_cohort.items()):
        r2_values = sorted({r["external_recon_r2"] for r in group if r["external_recon_r2"]})
        table_rows.append(
            [
                model,
                cohort,
                str(len(group)),
                group[0]["external_activation_status"],
                "; ".join(sorted({r["recon_gate_status"] for r in group})),
                str(sum(r["recon_gate_pass"] == "true" for r in group)),
                ", ".join(r2_values[:4]) + (" ..." if len(r2_values) > 4 else ""),
            ]
        )
    out.extend(["", "Per model/cohort gate status:", ""])
    out.extend(
        md_table(
            ["Model", "External cohort", "Rows", "Activation", "Recon status", "Passes", "R2 values"],
            table_rows,
        )
    )
    out.extend(
        [
            "",
            "Claim discipline: current observed external SAE reconstruction results are CSFM-specific because available external activation rows are limited to CSFM. They support a reconstruction-fidelity non-transfer finding for PTB-XL-trained CSFM SAE dictionaries on the tested external cohorts, but no external SAE steering claim and no six-model SAE transfer generalization.",
        ]
    )
    clean_ningbo = root / "results" / "activations_external_v2" / "csfm_cu118_commons" / "ningbo_f"
    if clean_ningbo.exists() and not (clean_ningbo / "records.csv").exists():
        metadata_count = sum(1 for _ in clean_ningbo.glob("*/activation_metadata.json"))
        out.extend(
            [
                "",
                f"CSFM Ningbo-F note: current CSV still reflects the first-pass cache; clean v2 extraction has {metadata_count}/542 shards and recon-gate refresh is pending.",
            ]
        )
    return out


def render_claim_box() -> list[str]:
    return [
        "",
        "## Claim Discipline Box",
        "",
        "Allowed current claims:",
        "",
        "- MIMIC-V supports restricted interval/rate/axis external validation using ICD-linked labels.",
        "- MIMIC-V primary Track V closure is limited to AF/rhythm and BBB/conduction; QT-related ICD labels are sensitivity-only because they are measurement-proximal.",
        "- Track F waveform extraction and closure are available only as family/task-gated robustness checks.",
        "- Track F currently supports ST/T and RR-derived external robustness; QRS/PR weakness reflects the current lead-II NeuroKit2 delineation gate and is not a scientific null for conduction transfer.",
        "- Current observed external SAE reconstruction failures support a CSFM-specific PTB-XL SAE dictionary fidelity non-transfer finding on the tested external cohorts.",
        "- Failed or partial gates are informative limitations, not leaderboard scores.",
        "",
        "Disallowed current claims:",
        "",
        "- Full 49-concept external transfer from MIMIC vendor measurements.",
        "- ST/amplitude transfer from MIMIC `machine_measurements.csv`.",
        "- External SAE steering failure before an external SAE reconstruction gate passes.",
        "- Six-model external SAE transfer or steering generalization while non-CSFM external activation caches are missing.",
        "- Any external model or cohort leaderboard ranking.",
    ]


def render(root: Path) -> str:
    lines = [
        "# Multi-Cohort Paper-Ready Tables",
        "",
        "Auto-generated from current multi-cohort CSV artifacts. External cohorts are robustness checks, not a leaderboard.",
    ]
    for block in [
        render_cohort_gate,
        render_crosswalk,
        render_task_feasibility,
        render_mimic_v_closure,
        render_track_f_closure,
        render_sae_gate,
    ]:
        lines.extend(block(root))
    lines.extend(render_claim_box())
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-ready multi-cohort summary tables.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(render(ROOT))
    os.replace(tmp, args.out)
    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
