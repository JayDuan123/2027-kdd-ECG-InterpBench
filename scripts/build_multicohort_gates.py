#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multicohort import DEFAULT_OUT_DIR, build_multicohort_gates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build multi-cohort gate artifacts for external ECG validation."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--overwrite-task-feasibility-template",
        action="store_true",
        help="Overwrite external_task_feasibility.csv with the initial template. By default an existing full G4 file is preserved.",
    )
    args = parser.parse_args()

    paths = build_multicohort_gates(
        args.out_dir,
        overwrite_task_feasibility_template=args.overwrite_task_feasibility_template,
    )
    print(f"mimic_measurement_audit: {paths.mimic_measurement_audit}")
    print(f"concept_crosswalk_track_v: {paths.concept_crosswalk_track_v}")
    print(f"cohort_gate_summary: {paths.cohort_gate_summary}")
    print(f"external_task_feasibility: {paths.external_task_feasibility}")
    print(f"report: {paths.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
