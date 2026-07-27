#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/rhf/allocations/wq8/yd68")
MIMIC_ECG_ROOT = Path("/rhf/allocations/wq8/mimic_data/mimic-iv-ecg")
MIMIC_RECORD_LIST = MIMIC_ECG_ROOT / "record_list.csv"

sys.path.insert(0, str(ROOT / "scripts"))
from build_waveform_extraction_smoke import extract_record, import_runtime  # noqa: E402


COHORT_ROOTS = {
    "ptbxl": WORKSPACE / "data" / "ptb-xl",
    "chapman": WORKSPACE / "challenge-2021" / "training" / "chapman_shaoxing",
    "cpsc": WORKSPACE / "challenge-2021" / "training" / "cpsc_2018",
    "ningbo": WORKSPACE / "challenge-2021" / "training" / "ningbo",
    "mimic": MIMIC_ECG_ROOT,
}


OUT_FIELDS = [
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


def external_records(cohort: str) -> list[dict[str, str]]:
    root = COHORT_ROOTS[cohort]
    rows = []
    for path in sorted(root.rglob("*.hea")):
        base = path.with_suffix("")
        rows.append(
            {
                "cohort": cohort,
                "base": str(base),
                "subject_id": "",
                "study_id_or_record_key": base.name,
                "ecg_time": "",
            }
        )
    return rows


def mimic_records() -> list[dict[str, str]]:
    rows = []
    with MIMIC_RECORD_LIST.open(newline="") as f:
        for row in csv.DictReader(f):
            rel_path = row.get("path", "")
            if not rel_path:
                continue
            base = MIMIC_ECG_ROOT / rel_path
            rows.append(
                {
                    "cohort": "mimic",
                    "base": str(base),
                    "subject_id": row.get("subject_id", ""),
                    "study_id_or_record_key": row.get("study_id", ""),
                    "ecg_time": row.get("ecg_time", ""),
                }
            )
    return rows


def all_records(cohort: str) -> list[dict[str, str]]:
    if cohort == "mimic":
        return mimic_records()
    return external_records(cohort)


def quality_flags(row: dict[str, str]) -> str:
    flags = []
    if row.get("status") != "ok":
        flags.append("extractor_failed")
    if row.get("interval_pass") != "true":
        flags.append("interval_partial_or_missing")
    if row.get("amplitude_pass") != "true":
        flags.append("amplitude_missing_or_implausible")
    if row.get("st_t_pass") != "true":
        flags.append("st_t_missing_or_implausible")
    if row.get("axis_pass") == "not_assessed":
        flags.append("axis_not_assessed")
    return "|".join(flags) if flags else "ok"


def extract_rows(records: list[dict[str, str]]) -> list[dict[str, str]]:
    nk, np, wfdb = import_runtime()
    out = []
    for meta in records:
        row = extract_record(meta["cohort"], Path(meta["base"]), nk, np, wfdb)
        row["subject_id"] = meta.get("subject_id", "")
        row["study_id_or_record_key"] = meta.get("study_id_or_record_key", row.get("record_id", ""))
        row["ecg_time"] = meta.get("ecg_time", "")
        row["concept_quality_flags"] = quality_flags(row)
        out.append({field: row.get(field, "") for field in OUT_FIELDS})
    return out


def write_csv_atomic(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build checkpointed Track F waveform concept shards.")
    parser.add_argument("--cohort", required=True, choices=sorted(COHORT_ROOTS))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "multicohort" / "track_f_full" / "shards",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for smoke/debug runs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mplconfig = Path(
        os.environ.get(
            "MPLCONFIGDIR",
            f"/tmp/matplotlib-{os.environ.get('USER', 'yd68')}",
        )
    )
    mplconfig.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfig))
    out_path = args.out_dir / f"{args.cohort}_shard_{args.shard_index:06d}.csv"
    if args.skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"skip_existing: {out_path}")
        return 0

    records = all_records(args.cohort)
    if args.max_records is not None:
        records = records[: args.max_records]
    start = args.shard_index * args.shard_size
    end = min(start + args.shard_size, len(records))
    if start >= len(records):
        print(f"empty shard: cohort={args.cohort} shard={args.shard_index} total={len(records)}")
        write_csv_atomic(out_path, [])
        return 0

    shard = records[start:end]
    rows = extract_rows(shard)
    write_csv_atomic(out_path, rows)
    ok = sum(row["status"] == "ok" for row in rows)
    print(
        f"wrote {out_path} cohort={args.cohort} shard={args.shard_index} "
        f"records={len(rows)} ok={ok} start={start} end={end} total={len(records)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
