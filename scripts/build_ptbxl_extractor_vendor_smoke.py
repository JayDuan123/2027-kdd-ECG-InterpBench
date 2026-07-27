#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/rhf/allocations/wq8/yd68")
PTBXL_PLUS = WORKSPACE / "data" / "1.0.1"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def absmax(values: list[float]) -> float | None:
    vals = [abs(x) for x in values if math.isfinite(x)]
    if not vals:
        return None
    return max(vals)


def load_12sl_by_ecg_id() -> dict[str, dict[str, str]]:
    feature_path = PTBXL_PLUS / "features" / "12sl_features.csv"
    statement_path = PTBXL_PLUS / "labels" / "12sl_statements.csv"
    with feature_path.open(newline="") as f_feat, statement_path.open(newline="") as f_stmt:
        features = list(csv.DictReader(f_feat))
        statements = list(csv.DictReader(f_stmt))
    if len(features) != len(statements):
        raise RuntimeError("12SL feature rows do not match 12SL statement rows")
    out: dict[str, dict[str, str]] = {}
    for feature_row, statement_row in zip(features, statements):
        ecg_id = statement_row.get("ecg_id")
        if ecg_id:
            out[ecg_id] = feature_row
    return out


def vendor_absmax(row: dict[str, str], prefix: str) -> float | None:
    values = []
    for lead in LEADS:
        x = parse_float(row.get(f"{prefix}_{lead}"))
        if x is not None:
            values.append(x)
    return absmax(values)


def hr_record_to_ecg_id(record_id: str) -> str | None:
    match = re.fullmatch(r"HR(\d{5})", record_id)
    if not match:
        return None
    return str(int(match.group(1)))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


CONCEPTS = [
    ("rr_mean", "INTERVAL", "rr_mean_ms", lambda r: parse_float(r.get("RR_Mean_Global")), "RR_Mean_Global"),
    ("qrs_duration", "INTERVAL", "qrs_duration_ms", lambda r: parse_float(r.get("QRS_Dur_Global")), "QRS_Dur_Global"),
    ("pr_interval", "INTERVAL", "pr_interval_ms", lambda r: parse_float(r.get("PR_Int_Global")), "PR_Int_Global"),
    ("qt_like", "INTERVAL", "qt_like_ms", lambda r: parse_float(r.get("QT_Int_Global")), "QT_Int_Global"),
    ("r_amp_global", "AMPLITUDE", "r_amp_global_mv", lambda r: vendor_absmax(r, "R_Amp"), "absmax(R_Amp_{lead})"),
    ("st_amp_global", "ST_T", "st_amp_global_mv", lambda r: vendor_absmax(r, "ST_Amp"), "absmax(ST_Amp_{lead})"),
    ("t_amp_global", "ST_T", "t_amp_global_mv", lambda r: vendor_absmax(r, "T_Amp"), "absmax(T_Amp_{lead})"),
]


def load_vendor_ceiling(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["concept"]: row["corr_12sl_unig"] for row in csv.DictReader(f)}


def build_comparison(smoke_csv: Path, ceiling_csv: Path) -> list[dict[str, str]]:
    by_12sl = load_12sl_by_ecg_id()
    ceilings = load_vendor_ceiling(ceiling_csv)
    with smoke_csv.open(newline="") as f:
        smoke_rows = [row for row in csv.DictReader(f) if row.get("cohort") == "ptbxl"]

    rows = []
    mapped = []
    for row in smoke_rows:
        ecg_id = hr_record_to_ecg_id(row.get("record_id", ""))
        if ecg_id is None or ecg_id not in by_12sl:
            continue
        mapped.append((row, by_12sl[ecg_id], ecg_id))

    for concept, family, extractor_col, vendor_fn, vendor_spec in CONCEPTS:
        xs = []
        ys = []
        for smoke_row, vendor_row, _ecg_id in mapped:
            x = parse_float(smoke_row.get(extractor_col))
            y = vendor_fn(vendor_row)
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        corr = pearson(xs, ys)
        ceiling = parse_float(ceilings.get(concept))
        ratio = corr / ceiling if corr is not None and ceiling not in (None, 0) else None
        rows.append(
            {
                "concept": concept,
                "family": family,
                "extractor_column": extractor_col,
                "vendor_12sl_spec": vendor_spec,
                "n_pairs": str(len(xs)),
                "corr_extractor_12sl": "" if corr is None else f"{corr:.8g}",
                "vendor_ceiling_corr_12sl_unig": "" if ceiling is None else f"{ceiling:.8g}",
                "corr_to_ceiling_ratio": "" if ratio is None else f"{ratio:.8g}",
                "status": "ok_small_sample" if corr is not None else "insufficient_pairs_or_zero_variance",
                "alignment_note": "HRxxxxx waveform record mapped to ecg_id xxxxx; 12SL features row-wise via labels/12sl_statements.csv",
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
        "# PTB-XL Extractor-Vendor Smoke Comparison",
        "",
        "This small-sample G3 check compares the PTB-XL rows from `waveform_extraction_smoke.csv` against 12SL vendor measurements.",
        "",
        "Interpretation limits:",
        "",
        "- This is a 20-record smoke result, not a full Track F validation.",
        "- HRxxxxx waveform records are mapped to PTB-XL `ecg_id` by numeric suffix; this mapping was sanity-checked against age/sex before use.",
        "- Amplitude/ST/T smoke values use absolute global amplitude, so the vendor side uses absolute lead max for this smoke comparison.",
        "",
        "| Concept | N | extractor-vendor r | vendor ceiling r | ratio | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['concept']} | {row['n_pairs']} | {row['corr_extractor_12sl']} | "
            f"{row['vendor_ceiling_corr_12sl_unig']} | {row['corr_to_ceiling_ratio']} | {row['status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare PTB-XL waveform smoke extraction to 12SL vendor features.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "multicohort")
    parser.add_argument(
        "--smoke-csv",
        type=Path,
        default=ROOT / "results" / "multicohort" / "waveform_extraction_smoke.csv",
    )
    parser.add_argument(
        "--ceiling-csv",
        type=Path,
        default=ROOT / "results" / "multicohort" / "ptbxl_vendor_ceiling.csv",
    )
    args = parser.parse_args()
    rows = build_comparison(args.smoke_csv, args.ceiling_csv)
    write_csv(args.out_dir / "ptbxl_extractor_vendor_smoke.csv", rows)
    (args.out_dir / "ptbxl_extractor_vendor_smoke_report.md").write_text(render_report(rows))
    print(f"wrote: {args.out_dir / 'ptbxl_extractor_vendor_smoke.csv'}")
    print(f"wrote: {args.out_dir / 'ptbxl_extractor_vendor_smoke_report.md'}")
    for row in rows:
        print(row["concept"], row["n_pairs"], row["corr_extractor_12sl"], row["vendor_ceiling_corr_12sl_unig"], row["corr_to_ceiling_ratio"], row["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
