#!/usr/bin/env python
"""Final completion, integrity, and claim-boundary audit for method comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.method_comparison_common import (  # noqa: E402
    BASE,
    LABEL_BUDGETS,
    METHODS,
    atomic_write_text,
    write_json,
)
from scripts.build_method_comparison_report import END, START  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_old_checksums(path: Path) -> tuple[int, list[str]]:
    verified = 0
    skipped = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.strip()
        if relative == "README.md":
            skipped.append(relative)
            continue
        artifact = ROOT / relative
        if not artifact.exists() or sha256(artifact) != expected:
            raise RuntimeError(f"Prior artifact checksum mismatch: {artifact}")
        verified += 1
    return verified, skipped


def finite_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    for column in columns:
        if column not in frame or not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f"Non-finite or missing {label}.{column}")


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.base / "manifest.csv")
    worker_completes = sorted((args.base / "workers").glob("*/complete.json"))
    if len(manifest) != 72 or len(worker_completes) != 72:
        raise RuntimeError(
            f"Fit-worker audit failed: manifest={len(manifest)} complete={len(worker_completes)}"
        )
    worker_metadata = [json.loads(path.read_text()) for path in worker_completes]
    invalid_workers = [
        item
        for item in worker_metadata
        if item.get("status") != "complete"
        or item.get("waveforms_written") is not False
        or item.get("record_level_activations_written") is not False
        or item.get("data_files_modified") is not False
    ]
    if invalid_workers:
        raise RuntimeError(f"Invalid fit-worker metadata: {invalid_workers[:3]}")
    nonconverged_ica = [
        item["task_index"]
        for item in worker_metadata
        if not item["fit_diagnostics"]["ica64"]["converged"]
    ]
    if nonconverged_ica:
        raise RuntimeError(f"ICA did not converge for tasks: {nonconverged_ica}")

    summary_metadata = json.loads((args.base / "summary" / "metadata.json").read_text())
    if not summary_metadata.get("all_complete") or summary_metadata.get("complete_workers") != 72:
        raise RuntimeError(f"Summary metadata incomplete: {summary_metadata}")
    methods = pd.read_csv(args.base / "summary" / "method_seed_cells.csv")
    contrasts = pd.read_csv(args.base / "summary" / "paired_method_contrasts.csv")
    label_budget = pd.read_csv(args.base / "summary" / "label_budget_seed_cells.csv")
    reconstruction = pd.read_csv(args.base / "summary" / "reconstruction_seed_cells.csv")
    rate_distortion = pd.read_csv(args.base / "summary" / "rate_distortion_seed_cells.csv")
    matched = pd.read_csv(args.base / "summary" / "reconstruction_matched_operating_points.csv")
    hierarchical = pd.read_csv(args.base / "summary" / "hierarchical_method_inference.csv")
    dictionary = pd.read_csv(args.base / "summary" / "dictionary_seed_pair_stability.csv")
    functional = pd.read_csv(args.base / "summary" / "functional_seed_pair_stability.csv")
    if len(methods) != summary_metadata["method_rows"]:
        raise RuntimeError("Method row count mismatch")
    if len(contrasts) != summary_metadata["contrast_rows"]:
        raise RuntimeError("Contrast row count mismatch")
    if len(label_budget) != summary_metadata["label_budget_rows"]:
        raise RuntimeError("Label-budget row count mismatch")
    if set(label_budget.label_budget_requested.astype(int)) != set(LABEL_BUDGETS):
        raise RuntimeError("Label-budget levels are incomplete")
    if (label_budget.positive_labels <= 0).any() or (label_budget.negative_labels <= 0).any():
        raise RuntimeError("A label-budget cell lost one class")
    finite_columns(
        methods,
        ["ste", "otd_mean", "selectivity_margin", "wbi", "matched_l2_max_abs_error"],
        "methods",
    )
    finite_columns(
        reconstruction,
        ["dense_recon_r2", "topk_recon_r2", "dense_normalized_mse", "topk_normalized_mse"],
        "reconstruction",
    )
    finite_columns(rate_distortion, ["recon_r2", "normalized_mse", "cosine_mean"], "rate_distortion")
    finite_columns(matched, ["absolute_r2_gap", "selected_code_budget_k"], "matched")
    finite_columns(hierarchical, ["mean_delta", "ci_low", "ci_high", "q_two_sided"], "hierarchical")
    finite_columns(dictionary, ["matched_abs_cosine_mean", "subspace_overlap"], "dictionary")
    finite_columns(functional, ["functional_abs_cosine_mean", "functional_subspace_overlap"], "functional")
    max_l2_error = float(methods.matched_l2_max_abs_error.max())
    max_parity_error = float(methods.existing_sae_logit_parity_max_abs.max())
    if max_l2_error > 1e-3:
        raise RuntimeError(f"Per-record L2 matching failed: max error={max_l2_error}")
    if max_parity_error > 2e-4:
        raise RuntimeError(f"Existing SAE parity failed: max error={max_parity_error}")

    waveform_completes = sorted(
        (args.base / "waveform_triangle" / "workers").glob("*/*/*/complete.json")
    )
    if len(waveform_completes) != 12:
        raise RuntimeError(f"Waveform worker count is {len(waveform_completes)}, expected 12")
    waveform_worker_metadata = [json.loads(path.read_text()) for path in waveform_completes]
    invalid_waveform = [
        item
        for item in waveform_worker_metadata
        if item.get("status") != "complete"
        or item.get("waveforms_written") is not False
        or item.get("record_level_activations_written") is not False
        or item.get("data_files_modified") is not False
        or item.get("output_rows") != item.get("expected_output_rows")
    ]
    if invalid_waveform:
        raise RuntimeError(f"Invalid waveform workers: {invalid_waveform[:2]}")
    waveform_metadata = json.loads((args.base / "waveform_triangle" / "metadata.json").read_text())
    if not waveform_metadata.get("all_complete") or waveform_metadata.get("workers") != 12:
        raise RuntimeError(f"Waveform summary incomplete: {waveform_metadata}")
    if set(waveform_metadata["methods"]) != set((*METHODS, "sae_existing_8d")):
        raise RuntimeError("Waveform method set mismatch")

    report_metadata = json.loads((args.base / "report_metadata.json").read_text())
    report = (args.base / "method_comparison_report.md").read_text()
    readme = (ROOT / "README.md").read_text()
    if not report_metadata.get("all_complete") or not report_metadata.get("claim_boundary_present"):
        raise RuntimeError("Report metadata is incomplete")
    if readme.count(START) != 1 or readme.count(END) != 1:
        raise RuntimeError("README markers are missing or duplicated")
    required_claims = [
        "does not establish biological mechanism",
        "does not generate ECG waveforms",
        "SAE-specific value",
    ]
    missing_claims = [claim for claim in required_claims if claim not in report]
    if missing_claims:
        raise RuntimeError(f"Claim-boundary text missing: {missing_claims}")

    figures = [
        ROOT / "docs" / "figures" / "method_comparison_v1_main.png",
        ROOT / "docs" / "figures" / "method_comparison_v1_waveform.png",
    ]
    figure_dimensions = {}
    for figure in figures:
        with Image.open(figure) as image:
            image.verify()
        with Image.open(figure) as image:
            width, height = image.size
            if width < 1000 or height < 600:
                raise RuntimeError(f"Figure too small or invalid: {figure} {image.size}")
            figure_dimensions[str(figure.relative_to(ROOT))] = [width, height]

    prior_verified = 0
    prior_skipped = []
    for checksum_file in (
        ROOT / "results" / "benchmark_extension_v1" / "artifact_checksums.sha256",
        ROOT / "results" / "benchmark_extension_v2" / "artifact_checksums.sha256",
    ):
        verified, skipped = verify_old_checksums(checksum_file)
        prior_verified += verified
        prior_skipped.extend(f"{checksum_file.parent.name}:{item}" for item in skipped)

    artifacts = sorted((args.base / "summary").glob("*.csv")) + [
        args.base / "manifest.csv",
        args.base / "manifest.metadata.json",
        args.base / "summary" / "metadata.json",
        args.base / "waveform_triangle" / "method_triangle_profile.csv",
        args.base / "waveform_triangle" / "method_triangle_summary.csv",
        args.base / "waveform_triangle" / "method_triangle_hierarchical_inference.csv",
        args.base / "waveform_triangle" / "metadata.json",
        args.base / "method_comparison_report.md",
        args.base / "report_metadata.json",
        *figures,
    ]
    checksum_lines = [
        f"{sha256(path)}  {path.relative_to(ROOT)}" for path in artifacts
    ]
    atomic_write_text(args.base / "artifact_checksums.sha256", "\n".join(checksum_lines) + "\n")
    audit = {
        "schema_version": 1,
        "all_complete": True,
        "manifest_tasks": len(manifest),
        "fit_workers_verified": len(worker_metadata),
        "waveform_workers_verified": len(waveform_worker_metadata),
        "method_rows_verified": len(methods),
        "contrast_rows_verified": len(contrasts),
        "label_budget_rows_verified": len(label_budget),
        "reconstruction_rows_verified": len(reconstruction),
        "rate_distortion_rows_verified": len(rate_distortion),
        "hierarchical_rows_verified": len(hierarchical),
        "dictionary_stability_rows_verified": len(dictionary),
        "functional_stability_rows_verified": len(functional),
        "max_per_record_l2_match_error": max_l2_error,
        "max_existing_sae_logit_parity_error": max_parity_error,
        "ica_nonconverged_tasks": nonconverged_ica,
        "figures_verified": figure_dimensions,
        "prior_extension_files_verified_unchanged": prior_verified,
        "prior_readme_checksums_skipped_as_authorized_shared_document": prior_skipped,
        "new_artifact_checksums_written": len(artifacts),
        "readme_markers": readme.count(START),
        "claim_boundary_present": True,
        "waveforms_written": False,
        "record_level_activations_written": False,
        "data_files_modified": False,
    }
    write_json(args.base / "final_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
