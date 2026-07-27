#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/rhf/allocations/wq8/yd68")

COHORT_ROOTS = {
    "ptbxl": WORKSPACE / "data" / "ptb-xl",
    "chapman": WORKSPACE / "challenge-2021" / "training" / "chapman_shaoxing",
    "cpsc": WORKSPACE / "challenge-2021" / "training" / "cpsc_2018",
    "ningbo": WORKSPACE / "challenge-2021" / "training" / "ningbo",
}

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def import_runtime():
    try:
        import neurokit2 as nk
        import numpy as np
        import wfdb
    except Exception as exc:  # pragma: no cover - runtime environment check
        raise SystemExit(
            "Waveform smoke requires wfdb, neurokit2, and numpy. "
            "Use /rhf/allocations/wq8/yd68/venvs/st_mem_cu118/bin/python."
        ) from exc
    return nk, np, wfdb


def collect_record_bases(root: Path, limit: int) -> list[Path]:
    records = sorted(path.with_suffix("") for path in root.rglob("*.hea"))
    return records[:limit]


def finite_values(values: Iterable[object]) -> list[float]:
    out: list[float] = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def valid_interval_ms(values: Iterable[float], lo: float, hi: float) -> list[float]:
    return [x for x in values if lo <= x <= hi]


def sample_global_abs(signal, indices: list[int], np) -> float | None:
    vals: list[float] = []
    n = signal.shape[0]
    for idx in indices:
        if idx < 0 or idx >= n:
            continue
        row = signal[idx, :]
        if not np.isfinite(row).any():
            continue
        vals.append(float(np.nanmax(np.abs(row))))
    return median(vals)


def extract_record(cohort: str, base: Path, nk, np, wfdb) -> dict[str, str]:
    row = {
        "cohort": cohort,
        "record_id": base.name,
        "record_path": str(base),
        "status": "ok",
        "error": "",
        "fs": "",
        "sig_len": "",
        "n_sig": "",
        "lead_protocol": "",
        "rr_mean_ms": "",
        "qrs_duration_ms": "",
        "pr_interval_ms": "",
        "qt_like_ms": "",
        "r_amp_global_mv": "",
        "st_amp_global_mv": "",
        "t_amp_global_mv": "",
        "interval_pass": "false",
        "axis_pass": "not_assessed",
        "amplitude_pass": "false",
        "st_t_pass": "false",
        "vendor_comparison_status": "not_run_id_alignment_pending",
        "aggregation_parity": "global_absmax_for_amplitude_and_st_t",
        "tool": f"wfdb={getattr(wfdb, '__version__', '')};neurokit2={getattr(nk, '__version__', '')}",
    }
    try:
        rec = wfdb.rdrecord(str(base))
        signal = rec.p_signal
        fs = float(rec.fs)
        row["fs"] = str(fs)
        row["sig_len"] = str(rec.sig_len)
        row["n_sig"] = str(rec.n_sig)
        row["lead_protocol"] = "12_lead" if list(rec.sig_name) == LEADS else "|".join(rec.sig_name)
        if rec.n_sig < 12 or "II" not in rec.sig_name:
            raise ValueError(f"expected 12-lead record with lead II, found {rec.sig_name}")
        lead_ii = rec.sig_name.index("II")
        lead_signal = signal[:, lead_ii]
        if not np.isfinite(lead_signal).all():
            lead_signal = np.nan_to_num(lead_signal, nan=0.0)

        cleaned = nk.ecg_clean(lead_signal, sampling_rate=fs)
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = [int(x) for x in finite_values(peak_info.get("ECG_R_Peaks", []))]
        if len(rpeaks) < 3:
            raise ValueError("too few R peaks")
        _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")

        r_on = [int(x) for x in finite_values(waves.get("ECG_R_Onsets", []))]
        r_off = [int(x) for x in finite_values(waves.get("ECG_R_Offsets", []))]
        p_on = [int(x) for x in finite_values(waves.get("ECG_P_Onsets", []))]
        t_off = [int(x) for x in finite_values(waves.get("ECG_T_Offsets", []))]
        t_peak = [int(x) for x in finite_values(waves.get("ECG_T_Peaks", []))]

        rr_ms = valid_interval_ms(((b - a) / fs * 1000.0 for a, b in zip(rpeaks[:-1], rpeaks[1:])), 250, 2500)
        qrs_ms = valid_interval_ms(((b - a) / fs * 1000.0 for a, b in zip(r_on, r_off) if b > a), 40, 220)
        pr_ms = valid_interval_ms(((b - a) / fs * 1000.0 for a, b in zip(p_on, r_on) if b > a), 60, 320)
        qt_ms = valid_interval_ms(((b - a) / fs * 1000.0 for a, b in zip(r_on, t_off) if b > a), 180, 700)

        rr = median(rr_ms)
        qrs = median(qrs_ms)
        pr = median(pr_ms)
        qt = median(qt_ms)
        r_amp = sample_global_abs(signal, rpeaks, np)
        t_amp = sample_global_abs(signal, t_peak, np)
        st_indices = [idx + int(round(0.06 * fs)) for idx in r_off]
        st_amp = sample_global_abs(signal, st_indices, np)

        values = {
            "rr_mean_ms": rr,
            "qrs_duration_ms": qrs,
            "pr_interval_ms": pr,
            "qt_like_ms": qt,
            "r_amp_global_mv": r_amp,
            "st_amp_global_mv": st_amp,
            "t_amp_global_mv": t_amp,
        }
        for key, value in values.items():
            if value is not None and math.isfinite(value):
                row[key] = f"{value:.6g}"

        row["interval_pass"] = "true" if rr is not None and qrs is not None and qt is not None else "false"
        row["amplitude_pass"] = "true" if r_amp is not None and 0.01 <= r_amp <= 10 else "false"
        row["st_t_pass"] = (
            "true"
            if st_amp is not None
            and t_amp is not None
            and abs(st_amp) <= 10
            and abs(t_amp) <= 10
            else "false"
        )
    except Exception as exc:
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    cohorts = sorted({row["cohort"] for row in rows})
    for cohort in cohorts:
        subset = [row for row in rows if row["cohort"] == cohort]
        n = len(subset)
        ok = [row for row in subset if row["status"] == "ok"]
        summary = {
            "cohort": cohort,
            "records": str(n),
            "extractor_success": str(len(ok)),
            "extractor_success_frac": f"{len(ok) / n:.6g}" if n else "",
            "interval_pass_count": str(sum(row["interval_pass"] == "true" for row in subset)),
            "amplitude_pass_count": str(sum(row["amplitude_pass"] == "true" for row in subset)),
            "st_t_pass_count": str(sum(row["st_t_pass"] == "true" for row in subset)),
            "axis_pass_count": "0",
            "axis_status": "not_assessed_in_neurokit2_smoke",
            "vendor_comparison_status": "not_run_id_alignment_pending",
            "overall_track_f_smoke_status": "pass_partial" if ok else "fail",
        }
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_report(rows: list[dict[str, str]], summary: list[dict[str, str]]) -> str:
    ceiling_path = ROOT / "results" / "multicohort" / "ptbxl_vendor_ceiling.csv"
    ceiling_status = (
        f"available at `{ceiling_path}`"
        if ceiling_path.exists()
        else "not yet built; run `python scripts/build_ptbxl_vendor_ceiling.py`"
    )
    extractor_vendor_paths = [
        ROOT / "results" / "multicohort" / "ptbxl_extractor_vendor_smoke.csv",
        ROOT / "results" / "multicohort" / "waveform_smoke_1k" / "ptbxl_extractor_vendor_smoke.csv",
    ]
    extractor_vendor_existing = [path for path in extractor_vendor_paths if path.exists()]
    if extractor_vendor_existing:
        extractor_vendor_status = (
            "available separately at "
            + ", ".join(f"`{path}`" for path in extractor_vendor_existing)
            + "."
        )
    else:
        extractor_vendor_status = (
            "not yet built; run `python scripts/build_ptbxl_extractor_vendor_smoke.py` after waveform-to-PTB-XL+ ID alignment."
        )
    lines = [
        "# Waveform Extraction Smoke Gate",
        "",
        "This is the G3 Track F smoke test. It verifies that `wfdb` can load records and NeuroKit2 can extract basic lead-II fiducials on a small cohort sample.",
        "",
        "Limitations:",
        "",
        "- Axis concepts are not assessed in this NeuroKit2 smoke.",
        f"- PTB-XL+ vendor-vendor correlation ceiling is {ceiling_status}.",
        f"- PTB-XL waveform-derived extractor-vendor correlations are {extractor_vendor_status}",
        "- Amplitude/ST/T values use global absolute max aggregation at delineated indices for smoke only; full Track F must preserve the registered aggregation rules.",
        "",
        "## Summary",
        "",
        "| Cohort | Records | Success | Interval | Amplitude | ST/T | Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['cohort']} | {row['records']} | {row['extractor_success']} | "
            f"{row['interval_pass_count']} | {row['amplitude_pass_count']} | "
            f"{row['st_t_pass_count']} | {row['overall_track_f_smoke_status']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run small Track F waveform extraction smoke gate.")
    parser.add_argument("--records-per-cohort", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "multicohort")
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["ptbxl", "chapman", "cpsc", "ningbo"],
        choices=sorted(COHORT_ROOTS),
    )
    args = parser.parse_args()

    nk, np, wfdb = import_runtime()
    rows: list[dict[str, str]] = []
    for cohort in args.cohorts:
        for base in collect_record_bases(COHORT_ROOTS[cohort], args.records_per_cohort):
            rows.append(extract_record(cohort, base, nk, np, wfdb))
    summary = summarize(rows)
    write_csv(args.out_dir / "waveform_extraction_smoke.csv", rows)
    write_csv(args.out_dir / "waveform_extraction_smoke_summary.csv", summary)
    (args.out_dir / "waveform_extraction_smoke_report.md").write_text(render_report(rows, summary))
    print(f"wrote: {args.out_dir / 'waveform_extraction_smoke.csv'}")
    print(f"wrote: {args.out_dir / 'waveform_extraction_smoke_summary.csv'}")
    print(f"wrote: {args.out_dir / 'waveform_extraction_smoke_report.md'}")
    for row in summary:
        print(
            row["cohort"],
            row["records"],
            row["extractor_success"],
            row["interval_pass_count"],
            row["amplitude_pass_count"],
            row["st_t_pass_count"],
            row["overall_track_f_smoke_status"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
