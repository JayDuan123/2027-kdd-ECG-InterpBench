#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "results" / "multicohort" / "track_f_full"
DEFAULT_SHARD_DIR = DEFAULT_ROOT / "shards"
COHORTS = ("ptbxl", "chapman", "cpsc", "ningbo", "mimic")


CONCEPTS = [
    {
        "concept": "rr_mean",
        "field": "rr_mean_ms",
        "family": "RATE_RHYTHM",
        "unit": "ms",
        "lo": 250.0,
        "hi": 2500.0,
    },
    {
        "concept": "qrs_duration",
        "field": "qrs_duration_ms",
        "family": "INTERVAL",
        "unit": "ms",
        "lo": 40.0,
        "hi": 220.0,
    },
    {
        "concept": "pr_interval",
        "field": "pr_interval_ms",
        "family": "INTERVAL",
        "unit": "ms",
        "lo": 60.0,
        "hi": 320.0,
    },
    {
        "concept": "qt_like",
        "field": "qt_like_ms",
        "family": "INTERVAL",
        "unit": "ms",
        "lo": 180.0,
        "hi": 700.0,
    },
    {
        "concept": "r_amp_global",
        "field": "r_amp_global_mv",
        "family": "AMPLITUDE",
        "unit": "mV",
        "lo": 0.01,
        "hi": 10.0,
        "abs_range": True,
    },
    {
        "concept": "st_amp_global",
        "field": "st_amp_global_mv",
        "family": "ST_T",
        "unit": "mV",
        "lo": -10.0,
        "hi": 10.0,
    },
    {
        "concept": "t_amp_global",
        "field": "t_amp_global_mv",
        "family": "ST_T",
        "unit": "mV",
        "lo": -10.0,
        "hi": 10.0,
    },
]


COMBINED_FIELDS = [
    "cohort",
    "record_id",
    "record_path",
    "subject_id",
    "study_id_or_record_key",
    "ecg_time",
    "status",
    "error",
    "fs",
    "sig_len",
    "n_sig",
    "lead_protocol",
    "rr_mean_ms",
    "qrs_duration_ms",
    "pr_interval_ms",
    "qt_like_ms",
    "r_amp_global_mv",
    "st_amp_global_mv",
    "t_amp_global_mv",
    "interval_pass",
    "axis_pass",
    "amplitude_pass",
    "st_t_pass",
    "concept_quality_flags",
    "vendor_comparison_status",
    "aggregation_parity",
    "tool",
]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if not math.isfinite(out):
        return None
    return out


def plausible(value: float | None, spec: dict[str, object]) -> bool:
    if value is None:
        return False
    lo = float(spec["lo"])
    hi = float(spec["hi"])
    if spec.get("abs_range"):
        value = abs(value)
    return lo <= value <= hi


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_concept_gate(root: Path) -> dict[str, dict[str, str]]:
    preferred = root / "results" / "multicohort" / "waveform_smoke_1k" / "waveform_concept_gate.csv"
    fallback = root / "results" / "multicohort" / "waveform_concept_gate.csv"
    rows = read_csv(preferred) or read_csv(fallback)
    return {row["concept"]: row for row in rows if row.get("concept")}


def load_family_gate(root: Path) -> dict[str, str]:
    preferred = root / "results" / "multicohort" / "waveform_smoke_1k" / "waveform_family_gate.csv"
    fallback = root / "results" / "multicohort" / "waveform_family_gate.csv"
    rows = read_csv(preferred) or read_csv(fallback)
    family_status: dict[str, str] = {}
    for row in rows:
        family = row.get("family", "")
        status = row.get("family_gate_status", "")
        if family and family not in family_status:
            family_status[family] = status
    return family_status


def shard_paths(shard_dir: Path, cohorts: list[str]) -> list[Path]:
    paths: list[Path] = []
    for cohort in cohorts:
        paths.extend(sorted(shard_dir.glob(f"{cohort}_shard_*.csv")))
    return paths


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def init_stats() -> dict[str, int]:
    return {
        "records": 0,
        "extractor_success": 0,
        "nonmissing": 0,
        "plausible": 0,
    }


def summarize_shards(
    paths: list[Path],
    out_dir: Path,
    write_combined: bool,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    stats: dict[tuple[str, str], dict[str, int]] = defaultdict(init_stats)
    cohort_stats: dict[str, dict[str, int]] = defaultdict(init_stats)
    error_counts: dict[str, int] = defaultdict(int)
    combined_tmp = out_dir / "waveform_concepts_by_record.csv.tmp"
    combined_final = out_dir / "waveform_concepts_by_record.csv"
    writer = None
    combined_file = None
    if write_combined:
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_file = combined_tmp.open("w", newline="")
        writer = csv.DictWriter(combined_file, fieldnames=COMBINED_FIELDS)
        writer.writeheader()

    try:
        for path in paths:
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cohort = row.get("cohort", "")
                    if not cohort:
                        continue
                    ok = row.get("status") == "ok"
                    cohort_stats[cohort]["records"] += 1
                    if ok:
                        cohort_stats[cohort]["extractor_success"] += 1
                    else:
                        error = row.get("error") or "unknown_error"
                        error_counts[f"{cohort}:{error.split(':', 1)[0]}"] += 1
                    for spec in CONCEPTS:
                        concept = str(spec["concept"])
                        key = (cohort, concept)
                        value = parse_float(row.get(str(spec["field"])))
                        stats[key]["records"] += 1
                        if ok:
                            stats[key]["extractor_success"] += 1
                        if value is not None:
                            stats[key]["nonmissing"] += 1
                        if plausible(value, spec):
                            stats[key]["plausible"] += 1
                    if writer is not None:
                        writer.writerow({field: row.get(field, "") for field in COMBINED_FIELDS})
    finally:
        if combined_file is not None:
            combined_file.close()

    if write_combined:
        os.replace(combined_tmp, combined_final)

    concept_gate = load_concept_gate(ROOT)
    family_gate = load_family_gate(ROOT)
    missingness_rows: list[dict[str, str]] = []
    quality_rows: list[dict[str, str]] = []
    for cohort in sorted(cohort_stats):
        for spec in CONCEPTS:
            concept = str(spec["concept"])
            family = str(spec["family"])
            key = (cohort, concept)
            s = stats[key]
            records = s["records"]
            nonmissing = s["nonmissing"]
            plausible_count = s["plausible"]
            extractor_success = s["extractor_success"]
            missingness_rows.append(
                {
                    "cohort": cohort,
                    "concept": concept,
                    "family": family,
                    "field": str(spec["field"]),
                    "unit": str(spec["unit"]),
                    "records": str(records),
                    "extractor_success": str(extractor_success),
                    "extractor_success_frac": f"{extractor_success / records:.6g}" if records else "",
                    "nonmissing_count": str(nonmissing),
                    "nonmissing_frac": f"{nonmissing / records:.6g}" if records else "",
                    "plausible_count": str(plausible_count),
                    "plausible_frac": f"{plausible_count / records:.6g}" if records else "",
                }
            )
            gate = concept_gate.get(concept, {})
            quality_status = gate.get("concept_gate_status", "not_assessed")
            success_ok = extractor_success / max(records, 1) >= 0.8
            plausible_ok = plausible_count / max(records, 1) >= 0.8
            plausible_partial = plausible_count / max(records, 1) >= 0.5
            if records == 0:
                full_status = "not_run"
                reason = "no shard rows found"
            elif quality_status == "smoke_fail":
                full_status = "excluded_smoke_fail"
                reason = gate.get("gate_reason", "small-sample extractor-vendor gate failed")
            elif quality_status == "partial" and plausible_ok and success_ok:
                full_status = "partial_full_candidate"
                reason = "full extraction coverage is acceptable, but PTB-XL extractor-vendor gate was partial"
            elif quality_status == "smoke_pass" and plausible_ok and success_ok:
                full_status = "full_candidate"
                reason = "full extraction has acceptable success and plausible-value coverage"
            elif quality_status == "not_assessed" and plausible_ok and success_ok:
                full_status = "coverage_only_no_vendor_gate"
                reason = "full extraction coverage is acceptable, but no PTB-XL extractor-vendor gate is available"
            elif plausible_partial:
                full_status = "partial_full_candidate"
                reason = "full extraction is usable only with concept-specific missingness handling"
            else:
                full_status = "full_fail"
                reason = "full extraction has weak plausible-value coverage"
            family_smoke = family_gate.get(family, "")
            if not family_smoke:
                family_smoke = f"concept_level_{quality_status}"
            quality_rows.append(
                {
                    "cohort": cohort,
                    "concept": concept,
                    "family": family,
                    "concept_gate_status_smoke": quality_status,
                    "family_gate_status_smoke": family_smoke,
                    "full_extraction_status": full_status,
                    "records": str(records),
                    "extractor_success_frac": f"{extractor_success / records:.6g}" if records else "",
                    "plausible_frac": f"{plausible_count / records:.6g}" if records else "",
                    "ptbxl_extractor_vendor_ratio": gate.get("corr_to_ceiling_ratio", ""),
                    "ptbxl_extractor_vendor_n": gate.get("n_pairs", ""),
                    "quality_reason": reason,
                }
            )
    return missingness_rows, quality_rows, dict(error_counts)


def render_report(
    paths: list[Path],
    missingness: list[dict[str, str]],
    quality: list[dict[str, str]],
    error_counts: dict[str, int],
    write_combined: bool,
) -> str:
    cohorts = sorted({row["cohort"] for row in missingness})
    lines = [
        "# Track F Full Waveform Extraction Summary",
        "",
        "This report summarizes the full checkpointed Track F waveform-derived concept extraction. It is a gate artifact, not a clinical result table.",
        "",
        f"- Shards read: {len(paths)}",
        f"- Record-level concept table written: {'yes' if write_combined else 'no'}",
        "- Axis concepts remain out of scope for this NeuroKit2 waveform extractor.",
        "- PTB-XL extractor-vendor gate status is inherited from the 1000-record smoke comparison.",
        "",
        "## Cohort Coverage",
        "",
        "| Cohort | Records | Extractor success | RR plausible | R amplitude plausible | ST plausible | T plausible |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_key = {(row["cohort"], row["concept"]): row for row in missingness}
    for cohort in cohorts:
        rr = by_key.get((cohort, "rr_mean"), {})
        r_amp = by_key.get((cohort, "r_amp_global"), {})
        st = by_key.get((cohort, "st_amp_global"), {})
        t_amp = by_key.get((cohort, "t_amp_global"), {})
        lines.append(
            f"| {cohort} | {rr.get('records', '')} | {rr.get('extractor_success_frac', '')} | "
            f"{rr.get('plausible_frac', '')} | {r_amp.get('plausible_frac', '')} | "
            f"{st.get('plausible_frac', '')} | {t_amp.get('plausible_frac', '')} |"
        )
    lines.extend(["", "## Concept Quality", "", "| Cohort | Concept | Smoke gate | Full status | Plausible frac | Reason |", "|---|---|---|---|---:|---|"])
    for row in quality:
        lines.append(
            f"| {row['cohort']} | {row['concept']} | {row['concept_gate_status_smoke']} | "
            f"{row['full_extraction_status']} | {row['plausible_frac']} | {row['quality_reason']} |"
        )
    if error_counts:
        lines.extend(["", "## Top Error Classes", "", "| Error | Count |", "|---|---:|"])
        for key, count in sorted(error_counts.items(), key=lambda item: item[1], reverse=True)[:20]:
            lines.append(f"| {key} | {count} |")
    lines.extend(
        [
            "",
            "## Claim Discipline",
            "",
            "- `full_candidate` allows downstream Track F closure only for the matching concept/family and cohort.",
            "- `excluded_smoke_fail` concepts remain excluded from Track F main claims even if full extraction produced values.",
            "- MIMIC vendor Track V and waveform Track F remain separate tracks.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize full Track F waveform concept shards.")
    parser.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=COHORTS)
    parser.add_argument("--no-combined", action="store_true", help="Skip writing the record-level combined table.")
    args = parser.parse_args()

    paths = shard_paths(args.shard_dir, args.cohorts)
    if not paths:
        raise SystemExit(f"No shard CSV files found in {args.shard_dir}")

    missingness, quality, error_counts = summarize_shards(paths, args.out_dir, not args.no_combined)
    write_csv_atomic(
        args.out_dir / "waveform_concept_missingness_by_cohort.csv",
        missingness,
        [
            "cohort",
            "concept",
            "family",
            "field",
            "unit",
            "records",
            "extractor_success",
            "extractor_success_frac",
            "nonmissing_count",
            "nonmissing_frac",
            "plausible_count",
            "plausible_frac",
        ],
    )
    write_csv_atomic(
        args.out_dir / "waveform_concept_quality.csv",
        quality,
        [
            "cohort",
            "concept",
            "family",
            "concept_gate_status_smoke",
            "family_gate_status_smoke",
            "full_extraction_status",
            "records",
            "extractor_success_frac",
            "plausible_frac",
            "ptbxl_extractor_vendor_ratio",
            "ptbxl_extractor_vendor_n",
            "quality_reason",
        ],
    )
    report = render_report(paths, missingness, quality, error_counts, not args.no_combined)
    report_path = args.out_dir / "track_f_full_extraction_report.md"
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(report)
    os.replace(tmp, report_path)
    print(f"read shards: {len(paths)}")
    print(f"wrote: {args.out_dir / 'waveform_concept_missingness_by_cohort.csv'}")
    print(f"wrote: {args.out_dir / 'waveform_concept_quality.csv'}")
    if not args.no_combined:
        print(f"wrote: {args.out_dir / 'waveform_concepts_by_record.csv'}")
    print(f"wrote: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
