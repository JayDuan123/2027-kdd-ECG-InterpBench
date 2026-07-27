#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.build_manifest import DEFAULT_OUT_DIR, build_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build ECG FM benchmark v1 concept/task manifest.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    paths = build_manifest(args.out_dir)
    print(f"concepts_matrix: {paths.concepts_matrix}")
    print(f"tasks_matrix: {paths.tasks_matrix}")
    print(f"split: {paths.split}")
    print(f"concept_summary: {paths.concept_summary}")
    print(f"report: {paths.report}")
    print(f"provenance_report: {paths.provenance_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
