#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/rhf/allocations/wq8/yd68")
PTBXL_PLUS = WORKSPACE / "data" / "1.0.1"
FEATURE_DIR = PTBXL_PLUS / "features"
LABEL_DIR = PTBXL_PLUS / "labels"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


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


def signed_absmax(values: list[float]) -> float | None:
    vals = [x for x in values if math.isfinite(x)]
    if not vals:
        return None
    return max(vals, key=lambda x: abs(x))


def load_12sl_by_ecg_id() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    feature_path = FEATURE_DIR / "12sl_features.csv"
    statement_path = LABEL_DIR / "12sl_statements.csv"
    with feature_path.open(newline="") as f_feat, statement_path.open(newline="") as f_stmt:
        features = list(csv.DictReader(f_feat))
        statements = list(csv.DictReader(f_stmt))
    if len(features) != len(statements):
        raise RuntimeError(
            f"12SL feature/statement row mismatch: {len(features)} vs {len(statements)}"
        )
    out = {}
    for feature_row, statement_row in zip(features, statements):
        ecg_id = statement_row.get("ecg_id")
        if not ecg_id:
            continue
        out[ecg_id] = feature_row
    meta = {
        "source": "12sl_features.csv has no ecg_id; aligned row-wise to labels/12sl_statements.csv",
        "feature_rows": str(len(features)),
        "statement_rows": str(len(statements)),
        "unique_ecg_ids": str(len(out)),
    }
    return out, meta


def load_unig_by_ecg_id() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    feature_path = FEATURE_DIR / "unig_features.csv"
    out = {}
    with feature_path.open(newline="") as f:
        for row in csv.DictReader(f):
            ecg_id = row.get("ecg_id")
            if ecg_id:
                out[ecg_id] = row
    meta = {
        "source": "unig_features.csv has explicit ecg_id",
        "feature_rows": str(len(out)),
        "unique_ecg_ids": str(len(out)),
    }
    return out, meta


def direct(column: str):
    return lambda row: parse_float(row.get(column))


def lead_signed_absmax(prefix: str):
    def fn(row: dict[str, str]) -> float | None:
        vals = []
        for lead in LEADS:
            value = parse_float(row.get(f"{prefix}_{lead}"))
            if value is not None:
                vals.append(value)
        return signed_absmax(vals)

    return fn


CONCEPTS = [
    {
        "concept": "rr_mean",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "RR_Mean_Global",
        "feature_unig": "RR_Mean_Global",
        "fn_12sl": direct("RR_Mean_Global"),
        "fn_unig": direct("RR_Mean_Global"),
    },
    {
        "concept": "qrs_duration",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "QRS_Dur_Global",
        "feature_unig": "QRS_Dur_Global",
        "fn_12sl": direct("QRS_Dur_Global"),
        "fn_unig": direct("QRS_Dur_Global"),
    },
    {
        "concept": "pr_interval",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "PR_Int_Global",
        "feature_unig": "PR_Int_Global",
        "fn_12sl": direct("PR_Int_Global"),
        "fn_unig": direct("PR_Int_Global"),
    },
    {
        "concept": "qt_like",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "QT_Int_Global",
        "feature_unig": "QT_Int_Global",
        "fn_12sl": direct("QT_Int_Global"),
        "fn_unig": direct("QT_Int_Global"),
    },
    {
        "concept": "qtc_bazett",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "QT_IntBazett_Global",
        "feature_unig": "QT_IntBazett_Global",
        "fn_12sl": direct("QT_IntBazett_Global"),
        "fn_unig": direct("QT_IntBazett_Global"),
    },
    {
        "concept": "qtc_fridericia",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "QT_IntFridericia_Global",
        "feature_unig": "QT_IntFridericia_Global",
        "fn_12sl": direct("QT_IntFridericia_Global"),
        "fn_unig": direct("QT_IntFridericia_Global"),
    },
    {
        "concept": "qtc_framingham",
        "family": "INTERVAL",
        "aggregation": "global",
        "feature_12sl": "QT_IntFramingham_Global",
        "feature_unig": "QT_IntFramingham_Global",
        "fn_12sl": direct("QT_IntFramingham_Global"),
        "fn_unig": direct("QT_IntFramingham_Global"),
    },
    {
        "concept": "r_amp_global",
        "family": "AMPLITUDE",
        "aggregation": "lead_signed_absmax",
        "feature_12sl": "R_Amp_{lead}",
        "feature_unig": "R_Amp_{lead}",
        "fn_12sl": lead_signed_absmax("R_Amp"),
        "fn_unig": lead_signed_absmax("R_Amp"),
    },
    {
        "concept": "st_amp_global",
        "family": "ST_T",
        "aggregation": "lead_signed_absmax",
        "feature_12sl": "ST_Amp_{lead}",
        "feature_unig": "ST_Amp_{lead}",
        "fn_12sl": lead_signed_absmax("ST_Amp"),
        "fn_unig": lead_signed_absmax("ST_Amp"),
    },
    {
        "concept": "t_amp_global",
        "family": "ST_T",
        "aggregation": "lead_signed_absmax",
        "feature_12sl": "T_Amp_{lead}",
        "feature_unig": "T_Amp_{lead}",
        "fn_12sl": lead_signed_absmax("T_Amp"),
        "fn_unig": lead_signed_absmax("T_Amp"),
    },
]


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    sx = math.sqrt(sum(x * x for x in dx))
    sy = math.sqrt(sum(y * y for y in dy))
    if sx == 0 or sy == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / (sx * sy)


def build_vendor_ceiling() -> tuple[list[dict[str, str]], dict[str, str]]:
    by_12sl, meta_12sl = load_12sl_by_ecg_id()
    by_unig, meta_unig = load_unig_by_ecg_id()
    common_ids = sorted(set(by_12sl) & set(by_unig), key=lambda x: int(x))
    rows = []
    for spec in CONCEPTS:
        xs: list[float] = []
        ys: list[float] = []
        for ecg_id in common_ids:
            x = spec["fn_12sl"](by_12sl[ecg_id])
            y = spec["fn_unig"](by_unig[ecg_id])
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
        corr = pearson(xs, ys)
        status = "ok" if corr is not None else "insufficient_pairs_or_zero_variance"
        rows.append(
            {
                "concept": str(spec["concept"]),
                "family": str(spec["family"]),
                "aggregation": str(spec["aggregation"]),
                "feature_12sl": str(spec["feature_12sl"]),
                "feature_unig": str(spec["feature_unig"]),
                "n_pairs": str(len(xs)),
                "corr_12sl_unig": "" if corr is None else f"{corr:.8g}",
                "status": status,
                "alignment_note": "12SL row-wise via 12sl_statements.csv; Uni-G by explicit ecg_id",
            }
        )
    meta = {
        "12sl_source": meta_12sl["source"],
        "12sl_rows": meta_12sl["feature_rows"],
        "12sl_unique_ecg_ids": meta_12sl["unique_ecg_ids"],
        "unig_source": meta_unig["source"],
        "unig_rows": meta_unig["feature_rows"],
        "unig_unique_ecg_ids": meta_unig["unique_ecg_ids"],
        "common_ids": str(len(common_ids)),
    }
    return rows, meta


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(rows: list[dict[str, str]], meta: dict[str, str]) -> str:
    lines = [
        "# PTB-XL+ Vendor-Vendor Ceiling",
        "",
        "This report estimates the attainable vendor agreement ceiling for G3 Track F smoke concepts by correlating PTB-XL+ 12SL and Uni-G measurements.",
        "",
        "## Alignment",
        "",
        f"- 12SL: {meta['12sl_source']}",
        f"- 12SL rows / unique IDs: {meta['12sl_rows']} / {meta['12sl_unique_ecg_ids']}",
        f"- Uni-G: {meta['unig_source']}",
        f"- Uni-G rows / unique IDs: {meta['unig_rows']} / {meta['unig_unique_ecg_ids']}",
        f"- Common IDs: {meta['common_ids']}",
        "",
        "## Correlations",
        "",
        "| Concept | Family | N | corr(12SL, Uni-G) | Status |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['concept']} | {row['family']} | {row['n_pairs']} | "
            f"{row['corr_12sl_unig']} | {row['status']} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Interpretation Discipline",
            "",
            "- These correlations are ceiling references, not extractor-vendor results.",
            "- Waveform-derived extractor correlations must be judged relative to this ceiling, not against a universal 0.90 threshold.",
            "- PTB-XL waveform-to-PTB-XL+ row alignment remains a separate gate before extractor-vendor correlations can be computed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PTB-XL+ 12SL-vs-Uni-G vendor ceiling for Track F smoke.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "multicohort")
    args = parser.parse_args()

    rows, meta = build_vendor_ceiling()
    write_csv(args.out_dir / "ptbxl_vendor_ceiling.csv", rows)
    (args.out_dir / "ptbxl_vendor_ceiling_report.md").write_text(render_report(rows, meta))
    print(f"wrote: {args.out_dir / 'ptbxl_vendor_ceiling.csv'}")
    print(f"wrote: {args.out_dir / 'ptbxl_vendor_ceiling_report.md'}")
    for row in rows:
        print(row["concept"], row["n_pairs"], row["corr_12sl_unig"], row["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
