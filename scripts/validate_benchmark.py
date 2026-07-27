#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.validate import run_checks, write_report  # noqa: E402


def main() -> int:
    results = run_checks()
    report = write_report()
    for result in results:
        print(f"{'PASS' if result.ok else 'FAIL'} {result.name}")
    print(f"report: {report}")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
