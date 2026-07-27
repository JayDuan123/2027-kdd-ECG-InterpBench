#!/usr/bin/env python
"""Requirement-level release audit for the matched-scale ECG-FM benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_release_path(path: Path, root: Path = ROOT) -> tuple[Path, Path]:
    resolved_root = root.resolve()
    candidate = path if path.is_absolute() else resolved_root / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise RuntimeError(
            f"release artifact is outside benchmark root: {resolved}"
        ) from error
    return resolved, relative


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=ROOT / "results/multiscale_sae_fixed_k_over_n_middepth_v1",
    )
    args = parser.parse_args()

    errors: list[str] = []
    evidence: dict[str, Any] = {}
    checksum_paths: list[Path] = []

    def require_file(label: str, path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"{label}: missing or empty {path}")
            evidence[label] = {"path": str(path), "status": "missing"}
            return False
        evidence[label] = {
            "path": str(path),
            "status": "present",
            "bytes": path.stat().st_size,
        }
        checksum_paths.append(path)
        return True

    def require_json(label: str, path: Path, predicates: dict[str, Any]) -> dict[str, Any] | None:
        if not require_file(label, path):
            return None
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            errors.append(f"{label}: invalid JSON: {error}")
            return None
        mismatches = {
            key: {"observed": payload.get(key), "expected": expected}
            for key, expected in predicates.items()
            if payload.get(key) != expected
        }
        evidence[label]["predicates"] = predicates
        evidence[label]["mismatches"] = mismatches
        if mismatches:
            errors.append(f"{label}: predicate mismatch {mismatches}")
        return payload

    def require_csv(label: str, path: Path, expected_rows: int) -> list[dict[str, str]] | None:
        if not require_file(label, path):
            return None
        rows = csv_rows(path)
        evidence[label]["rows"] = len(rows)
        evidence[label]["expected_rows"] = expected_rows
        if len(rows) != expected_rows:
            errors.append(f"{label}: rows={len(rows)}, expected={expected_rows}")
        return rows

    protocol = require_json(
        "primary_protocol",
        args.root / "protocol.json",
        {
            "benchmark_object": "ECG foundation-model representations",
            "n_training_cells": 450,
            "sparsity_arm": "fixed_k_over_d",
            "exact_absolute_scale_matching": True,
        },
    )
    require_file("slurm_chain", args.root / "slurm_chain.json")
    if protocol is not None:
        expected_protocol = {
            "models": 6,
            "relative_depths": [0.0, 0.25, 0.5, 0.75, 1.0],
            "expansion_E": [1, 4, 8, 16, 32],
            "seeds": [4311, 4312, 4313],
        }
        for key, expected in expected_protocol.items():
            observed = len(protocol[key]) if key == "models" else protocol[key]
            if observed != expected:
                errors.append(f"primary_protocol: {key}={observed}, expected={expected}")
        if "per-model best-scale" not in protocol.get("selection_rule", ""):
            errors.append("primary_protocol: missing explicit per-model best-scale prohibition")

    require_json(
        "primary_audit",
        args.root / "audit.json",
        {
            "expected_cells": 450,
            "complete_cells": 450,
            "expected_matched_blocks": 75,
            "complete_matched_blocks": 75,
            "exact_absolute_scale_blocks": 75,
            "record_manifest_alignment_pass": True,
            "record_count": 21799,
            "matched_scale_grid_pass": True,
            "audit_pass": True,
        },
    )
    matched_rows = require_csv(
        "matched_scale_grid", args.root / "matched_scale_grid_audit.csv", 75
    )
    require_csv(
        "record_manifest_alignment", args.root / "record_manifest_audit.csv", 6
    )
    if matched_rows is not None:
        failed = [
            row
            for row in matched_rows
            if row.get("manifest_status") != "pass" or row.get("result_status") != "complete"
        ]
        if failed:
            errors.append(f"matched_scale_grid: {len(failed)} failed blocks")
    require_csv("primary_cells", args.root / "cell_metrics.csv", 450)
    require_csv("primary_surface", args.root / "layer_scale_surface.csv", 150)
    require_csv("primary_profiles", args.root / "model_profiles.csv", 6)

    require_json(
        "stability_audit",
        args.root / "stability_audit.json",
        {"status": "complete", "seed_pair_rows": 450, "layer_scale_rows": 150, "model_profiles": 6},
    )
    require_csv("stability_pairs", args.root / "stability_seed_pairs.csv", 450)
    require_csv("stability_surface", args.root / "stability_layer_scale.csv", 150)
    require_csv("stability_profiles", args.root / "stability_model_profiles.csv", 6)

    require_json(
        "design_inference_audit",
        args.root / "test_inference_audit.json",
        {
            "status": "complete",
            "profile_rows": 24,
            "pair_rows": 60,
            "rank_stability_rows": 40,
            "matched_scale_blocks": 75,
            "patient_level_inference": False,
        },
    )
    require_csv("design_profiles", args.root / "test_model_profile_inference.csv", 24)
    require_csv("design_pairs", args.root / "test_model_pair_inference.csv", 60)
    require_csv("scale_rank_stability", args.root / "test_scale_rank_stability.csv", 40)

    patient_audit = require_json(
        "patient_bootstrap_audit",
        args.root / "test_patient_bootstrap_audit.json",
        {
            "status": "complete",
            "verified_tasks": 450,
            "expected_tasks": 450,
            "matched_scale_blocks": 75,
            "bootstrap_samples": 2000,
            "surface_rows": 450,
            "matched_scale_rows": 90,
            "model_profile_rows": 18,
            "model_pair_rows": 45,
            "rank_probability_rows": 108,
            "concept_rows": 294,
            "per_model_best_scale_ranking": False,
            "patient_bootstrap_protocol": "patient_cluster_v2",
        },
    )
    if patient_audit is not None:
        for key in ("patient_cluster_hash", "bootstrap_design_hash"):
            value = patient_audit.get(key)
            if not isinstance(value, str) or len(value) != 64:
                errors.append(f"patient_bootstrap_audit: invalid {key}={value!r}")
    require_csv("patient_surface", args.root / "test_patient_bootstrap_surface.csv", 450)
    require_csv(
        "patient_matched_scales", args.root / "test_patient_bootstrap_matched_scales.csv", 90
    )
    require_csv(
        "patient_model_profiles", args.root / "test_patient_bootstrap_model_profiles.csv", 18
    )
    require_csv("patient_model_pairs", args.root / "test_patient_bootstrap_model_pairs.csv", 45)
    require_csv(
        "patient_rank_probabilities",
        args.root / "test_patient_bootstrap_rank_probabilities.csv",
        108,
    )
    require_csv("patient_concepts", args.root / "test_patient_bootstrap_concepts.csv", 294)

    operating = require_json(
        "operating_point_audit",
        args.root / "operating_point_audit.json",
        {"status": "complete", "selection_split": "validation", "test_metrics_used_for_selection": False},
    )
    if operating is not None:
        selected_points = int(operating.get("selected_points", 0))
        selected_checkpoints = int(operating.get("selected_checkpoints", 0))
        if selected_points <= 0 or selected_checkpoints != 3 * selected_points:
            errors.append(
                "operating_point_audit: selected checkpoints must contain three seeds per point"
            )
        require_csv(
            "selected_operating_points", args.root / "selected_operating_points.csv", selected_points
        )
        require_csv(
            "selected_checkpoint_manifest",
            args.root / "selected_checkpoint_manifest.csv",
            selected_checkpoints,
        )

    require_json(
        "sensitivity_training_audit",
        args.sensitivity_root / "audit.json",
        {
            "expected_cells": 90,
            "complete_cells": 90,
            "expected_matched_blocks": 15,
            "complete_matched_blocks": 15,
            "exact_absolute_scale_blocks": 15,
            "record_manifest_alignment_pass": True,
            "record_count": 21799,
            "matched_scale_grid_pass": True,
            "audit_pass": True,
        },
    )
    require_json(
        "sensitivity_analysis_audit",
        args.sensitivity_root / "test_sparsity_sensitivity_audit.json",
        {
            "status": "complete",
            "matched_cells_per_arm": 90,
            "rank_agreement_rows": 20,
            "cell_difference_rows": 120,
            "auc_difference_rows": 24,
            "patient_level_inference": False,
        },
    )
    require_csv(
        "sensitivity_rank_agreement",
        args.sensitivity_root / "test_sparsity_rank_agreement.csv",
        20,
    )
    require_csv(
        "sensitivity_auc_profiles",
        args.sensitivity_root / "test_sparsity_auc_profiles.csv",
        48,
    )
    require_csv(
        "sensitivity_auc_differences",
        args.sensitivity_root / "test_sparsity_auc_differences.csv",
        24,
    )

    figure_audit = require_json(
        "figure_audit", args.root / "figure_audit.json", {"status": "complete"}
    )
    expected_figures = {
        "multiscale_benchmark_workflow",
        "multiscale_reconstruction_atlas",
        "multiscale_semantic_atlas",
        "multiscale_dead_feature_atlas",
        "multiscale_model_curves",
        "multiscale_stability_curves",
        "multiscale_patient_matched_scale_curves",
        "multiscale_patient_common_scale_auc",
        "multiscale_sparsity_sensitivity",
    }
    for stem in sorted(expected_figures):
        require_file(f"figure_{stem}_png", args.root / "figures" / f"{stem}.png")
        require_file(f"figure_{stem}_pdf", args.root / "figures" / f"{stem}.pdf")
    require_file("paper_table_multiscale", args.root / "paper_table_multiscale.tex")
    require_file("paper_table_multiscale_patient", args.root / "paper_table_multiscale_patient.tex")
    if figure_audit is not None:
        emitted = {Path(name).stem for name in figure_audit.get("figures", [])}
        missing = expected_figures - emitted
        if missing:
            errors.append(f"figure_audit: missing expected figures {sorted(missing)}")
        if figure_audit.get("patient_bootstrap_audit") is None:
            errors.append("figure_audit: missing patient-bootstrap provenance")
        if figure_audit.get("sparsity_sensitivity_audit") is None:
            errors.append("figure_audit: missing sensitivity provenance")

    checksum_paths = sorted(set(checksum_paths))
    resolved_checksum_paths = [
        resolve_release_path(path) for path in checksum_paths
    ]
    checksum_lines = [
        f"{sha256(resolved)}  {relative}"
        for resolved, relative in resolved_checksum_paths
    ]
    atomic_text(args.root / "final_release_checksums.sha256", "\n".join(checksum_lines) + "\n")
    summary = {
        "status": "complete" if not errors else "failed",
        "audit_pass": not errors,
        "requirements_checked": len(evidence),
        "errors": errors,
        "checksum_files": len(checksum_paths),
        "comparison_rule": "same SAE architecture, d, absolute N, and k within every FM block; no per-model best-scale ranking",
        "evidence": evidence,
    }
    atomic_text(args.root / "final_release_audit.json", json.dumps(summary, indent=2) + "\n")
    report = [
        "# Multi-Scale SAE Final Release Audit",
        "",
        f"- Status: {summary['status']}",
        f"- Audit pass: {summary['audit_pass']}",
        f"- Requirements checked: {summary['requirements_checked']}",
        f"- Checksum files: {summary['checksum_files']}",
        "- Comparison rule: same SAE architecture, d, absolute N, and k within every FM block; per-model best-scale ranking is prohibited.",
        "",
        "## Errors",
        "",
    ]
    report.extend([f"- {error}" for error in errors] or ["- None."])
    atomic_text(args.root / "final_release_audit.md", "\n".join(report) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "evidence"}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
