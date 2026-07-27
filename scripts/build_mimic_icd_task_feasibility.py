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
    build_external_task_feasibility_from_icd,
    write_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build MIMIC-V ICD-linked task feasibility counts for multi-cohort gates."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR / "external_task_feasibility.csv",
        help="Output CSV path. Defaults to results/multicohort/external_task_feasibility.csv.",
    )
    parser.add_argument("--min-positive", type=int, default=50)
    parser.add_argument("--min-negative", type=int, default=50)
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional smoke-test cap on ECG records. Omit for full MIMIC-V count.",
    )
    args = parser.parse_args()

    rows = build_external_task_feasibility_from_icd(
        min_positive=args.min_positive,
        min_negative=args.min_negative,
        max_records=args.max_records,
    )
    write_csv(args.out, rows)

    print(f"wrote: {args.out}")
    print(f"rows: {len(rows)}")
    if rows:
        print(f"records_seen: {rows[0].get('records_seen', '')}")
        print(f"records_linked_to_admission: {rows[0].get('records_linked_to_admission', '')}")
    for row in rows:
        if row["label_independence_tier"] != "primary_independent":
            continue
        print(
            ",".join(
                [
                    row["task"],
                    row["positive_count"],
                    row["negative_count"],
                    row["eligible_for_auroc"],
                    row["exclusion_reason"],
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
