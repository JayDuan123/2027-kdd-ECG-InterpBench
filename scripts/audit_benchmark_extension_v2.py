#!/usr/bin/env python
"""Final completeness, immutability, and checksum audit for extension v2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_v2_common import V1, V2, atomic_write_text, write_json  # noqa: E402


PRIMARY_FILES = (
    "hierarchical_robustness/transport_crossed_inference.csv",
    "hierarchical_robustness/transport_leave_one_out.csv",
    "hierarchical_robustness/protocol_factor_inference.csv",
    "hierarchical_robustness/protocol_leave_one_out.csv",
    "waveform_failure_bias/candidate_bias_covariates.csv",
    "waveform_failure_bias/propensity_diagnostics.csv",
    "waveform_failure_bias/waveform_ipw_sensitivity.csv",
    "sae_stability/capacity_seed_pair_stability.csv",
    "sae_stability/functional_top5_seed_pair_stability.csv",
    "waveform_triangle/triangle_paired_records.csv",
    "waveform_triangle/triangle_profile.csv",
    "waveform_triangle/triangle_summary.csv",
    "benchmark_extension_v2_report.md",
    "report_metadata.json",
)
FIGURES = (
    "docs/figures/benchmark_extension_v2_robustness.png",
    "docs/figures/benchmark_extension_v2_validation.png",
)
CODE_FILES = (
    "scripts/benchmark_extension_v2_common.py",
    "scripts/analyze_extension_v2_hierarchical_robustness.py",
    "scripts/analyze_waveform_failure_bias.py",
    "scripts/analyze_sae_stability_extension.py",
    "scripts/run_waveform_triangle_worker.py",
    "scripts/summarize_waveform_triangle.py",
    "scripts/build_benchmark_extension_v2_report.py",
    "scripts/audit_benchmark_extension_v2.py",
    "scripts/sbatch_extension_v2_hierarchical.sh",
    "scripts/sbatch_extension_v2_failure_bias.sh",
    "scripts/sbatch_extension_v2_sae_stability.sh",
    "scripts/sbatch_extension_v2_triangle_workers.sh",
    "scripts/sbatch_extension_v2_triangle_summary.sh",
    "scripts/sbatch_extension_v2_final_report.sh",
    "scripts/sbatch_extension_v2_final_audit.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=V2)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_v1_checksums() -> tuple[int, list[str]]:
    manifest = V1 / "artifact_checksums.sha256"
    verified = 0
    skipped = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        if relative == "README.md":
            skipped.append(relative)
            continue
        path = ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"v1 checksum changed: {path}")
        verified += 1
    return verified, skipped


def load_complete(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("all_complete") is not True:
        raise RuntimeError(f"Incomplete metadata: {path}")
    return payload


def main() -> None:
    args = parse_args()
    metadata_paths = {
        "hierarchical": args.base / "hierarchical_robustness" / "metadata.json",
        "failure_bias": args.base / "waveform_failure_bias" / "metadata.json",
        "sae_stability": args.base / "sae_stability" / "metadata.json",
        "triangle": args.base / "waveform_triangle" / "metadata.json",
        "report": args.base / "report_metadata.json",
    }
    metadata = {name: load_complete(path) for name, path in metadata_paths.items()}
    if metadata["triangle"].get("waveforms_written") is not False:
        raise RuntimeError("Triangle metadata does not prove waveform non-persistence")
    worker_complete = sorted((args.base / "waveform_triangle" / "workers").glob("*/*/*/complete.json"))
    if len(worker_complete) != 12:
        raise RuntimeError(f"Expected 12 triangle workers, got {len(worker_complete)}")
    reference_errors = []
    for path in worker_complete:
        worker = json.loads(path.read_text())
        if worker.get("status") != "complete" or worker.get("waveforms_written") is not False:
            raise RuntimeError(f"Invalid worker audit: {path}")
        reference_errors.append(float(worker["raw_head_reference_max_abs_difference"]))
    max_reference_error = max(reference_errors)
    if max_reference_error > 0.02:
        raise RuntimeError(
            f"Waveform re-inference drift exceeds 0.02 logit tolerance: {max_reference_error}"
        )

    expected_rows = {
        "hierarchical_robustness/transport_crossed_inference.csv": metadata["hierarchical"]["transport_inference_rows"],
        "hierarchical_robustness/transport_leave_one_out.csv": metadata["hierarchical"]["transport_leave_one_out_rows"],
        "hierarchical_robustness/protocol_factor_inference.csv": metadata["hierarchical"]["protocol_inference_rows"],
        "hierarchical_robustness/protocol_leave_one_out.csv": metadata["hierarchical"]["protocol_leave_one_out_rows"],
        "waveform_failure_bias/candidate_bias_covariates.csv": metadata["failure_bias"]["bias_rows"],
        "waveform_failure_bias/waveform_ipw_sensitivity.csv": metadata["failure_bias"]["ipw_rows"],
        "sae_stability/capacity_seed_pair_stability.csv": metadata["sae_stability"]["capacity_seed_pair_rows"],
        "sae_stability/functional_top5_seed_pair_stability.csv": metadata["sae_stability"]["functional_seed_pair_rows"],
        "waveform_triangle/triangle_paired_records.csv": metadata["triangle"]["seed_averaged_paired_records"],
        "waveform_triangle/triangle_profile.csv": metadata["triangle"]["profile_rows"],
    }
    for relative, expected in expected_rows.items():
        observed = len(pd.read_csv(args.base / relative))
        if observed != expected:
            raise RuntimeError(f"Row audit failed for {relative}: {observed} != {expected}")

    paths = [args.base / relative for relative in PRIMARY_FILES]
    paths += [ROOT / relative for relative in FIGURES]
    paths += [ROOT / relative for relative in CODE_FILES]
    for path in paths:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Missing/empty artifact: {path}")
    readme = (ROOT / "README.md").read_text()
    if readme.count("<!-- BENCHMARK_EXTENSION_V2_START -->") != 1 or readme.count(
        "<!-- BENCHMARK_EXTENSION_V2_END -->"
    ) != 1:
        raise RuntimeError("README v2 marker audit failed")
    report = (args.base / "benchmark_extension_v2_report.md").read_text()
    if "不证明临床干预效果" not in report:
        raise RuntimeError("Claim-boundary audit failed")

    checksum_lines = []
    for path in paths:
        checksum_lines.append(f"{sha256(path)}  {path.relative_to(ROOT)}")
    checksum_path = args.base / "artifact_checksums.sha256"
    atomic_write_text(checksum_path, "\n".join(checksum_lines) + "\n")
    v1_verified, v1_skipped = verify_v1_checksums()
    audit = {
        "schema_version": 1,
        "all_complete": True,
        "experiments": ["hierarchical", "failure_bias", "sae_stability", "triangle"],
        "triangle_workers_verified": len(worker_complete),
        "triangle_max_v1_logit_reproduction_error": max_reference_error,
        "triangle_v1_logit_reproduction_tolerance": 0.02,
        "row_audits_verified": len(expected_rows),
        "v2_checksum_files_written": len(checksum_lines),
        "v1_checksum_files_verified_unchanged": v1_verified,
        "v1_checksum_files_skipped_as_shared_mutable_document": v1_skipped,
        "figures_verified": len(FIGURES),
        "readme_markers": 1,
        "claim_boundary_present": True,
        "waveforms_written": False,
    }
    write_json(args.base / "final_audit.json", audit)
    print(audit)


if __name__ == "__main__":
    main()
