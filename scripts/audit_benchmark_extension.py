#!/usr/bin/env python
"""Final completeness and checksum audit for benchmark_extension_v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "benchmark_extension_v1"
from scripts.build_benchmark_extension_report import END, START, audit_inputs  # noqa: E402


def main() -> None:
    experiment_metadata = audit_inputs()
    combined_path = BASE / "metadata.json"
    combined = json.loads(combined_path.read_text())
    if not combined.get("all_complete") or not combined.get("readme_updated"):
        raise RuntimeError(f"Combined metadata incomplete: {combined}")

    report = ROOT / combined["report"]
    figures = [ROOT / path for path in combined["figures"]]
    if not report.exists() or report.stat().st_size < 1000:
        raise RuntimeError(f"Missing or undersized report: {report}")
    for figure in figures:
        if not figure.exists() or figure.stat().st_size < 10000:
            raise RuntimeError(f"Missing or undersized figure: {figure}")
        if figure.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError(f"Invalid PNG signature: {figure}")

    readme = (ROOT / "README.md").read_text()
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise RuntimeError("README extension markers are missing or duplicated")
    required_phrases = (
        "Paired protocol comparison", "Dose、direction", "Norm-matched baselines",
        "24-pair transport ladder", "Controlled waveform interventions",
    )
    missing = [phrase for phrase in required_phrases if phrase not in readme]
    if missing:
        raise RuntimeError(f"README extension section is incomplete: {missing}")

    checksum_path = ROOT / combined["checksums"]
    checked = 0
    for line in checksum_path.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {relative}")
        checked += 1
    if checked < 10:
        raise RuntimeError(f"Checksum manifest unexpectedly small: {checked}")

    audit = {
        "schema_version": 1,
        "all_complete": True,
        "experiments": sorted(experiment_metadata),
        "checksum_files_verified": checked,
        "figures_verified": len(figures),
        "readme_markers": 1,
        "report_bytes": report.stat().st_size,
        "claim_boundary_present": "不是临床或生物机制因果证据" in report.read_text(),
    }
    if not audit["claim_boundary_present"]:
        raise RuntimeError("Final report claim boundary is missing")
    (BASE / "final_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
