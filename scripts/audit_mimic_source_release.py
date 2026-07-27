#!/usr/bin/env python
"""Final requirement audit for the 100k MIMIC source benchmark release."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/mimic_source_benchmark_100k_v1"


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def passed(payload: dict) -> bool:
    if "audit_pass" in payload:
        return bool(payload["audit_pass"])
    if "status" in payload:
        return payload["status"] in {"complete", "pass"}
    if "complete_cells" in payload and "expected_cells" in payload:
        return int(payload["complete_cells"]) == int(payload["expected_cells"])
    return True


def resolve_seed_checkpoint(seed_dir: Path) -> Path:
    """Resolve the single non-empty BatchTopK checkpoint for a reused seed."""
    checkpoints = sorted(
        path
        for path in seed_dir.glob("batchtopk_*.pt")
        if path.is_file() and path.stat().st_size > 0
    )
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected exactly one non-empty batchtopk_*.pt in {seed_dir}, "
            f"found {len(checkpoints)}"
        )
    return checkpoints[0]


def main() -> None:
    errors = []
    checks = {
        "multiscale": RESULT / "audit.json",
        "inference": RESULT / "test_inference_audit.json",
        "stability": RESULT / "stability_audit.json",
        "patient_bootstrap": RESULT / "test_patient_bootstrap_audit.json",
        "accessibility": RESULT / "accessibility/summary/audit.json",
        "dictionary": RESULT / "dictionary/summary/audit.json",
        "final_sparse": RESULT / "final_sparse/summary/audit.json",
        "feature_yield": RESULT / "final_sparse/feature_yield/audit.json",
    }
    check_rows = []
    for name, path in checks.items():
        try:
            payload = load(path)
            ok = passed(payload)
        except Exception as exc:
            payload = {}
            ok = False
            errors.append(f"{name}: {exc}")
        if not ok:
            errors.append(f"{name}: audit did not pass")
        check_rows.append({"component": name, "path": str(path), "pass": ok})

    protocol = load(RESULT / "protocol.json")
    expected_protocol = {
        "records": 100_000,
        "patients": 48_491,
        "model_depth_cells": 30,
        "waveform_concepts": 7,
        "diagnosis_targets": 5,
        "training_cells": 450,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            errors.append(f"protocol {key}={protocol.get(key)!r}, expected {expected!r}")

    reused = []
    for suffix in (
        "cardiac_fm_cu118_commons", "csfm_cu118_commons", "ecg_fm_cu118_commons",
        "ecg_jepa_cu118_commons", "hubert_ecg_cu118_commons", "st_mem_cu118_commons",
    ):
        pair = ROOT / "results/external_benchmark_v1" / suffix / "mimic_f"
        seed_dir = pair / "cohort_adapted_sae/seed4311"
        try:
            checkpoint = resolve_seed_checkpoint(seed_dir)
            checkpoint_relative = str(checkpoint.relative_to(pair))
        except RuntimeError as exc:
            checkpoint = seed_dir / "batchtopk_*.pt"
            checkpoint_relative = "cohort_adapted_sae/seed4311/batchtopk_*.pt"
            errors.append(f"invalid reused SAE artifact for {suffix}: {exc}")

        for relative, path in (
            ("frozen_heads.joblib", pair / "frozen_heads.joblib"),
            (checkpoint_relative, checkpoint),
            ("closure/closure_summary.json", pair / "closure/closure_summary.json"),
        ):
            ok = path.exists() and path.stat().st_size > 0
            reused.append({"model_suffix": suffix, "artifact": relative, "path": str(path), "pass": ok})
            if not ok:
                errors.append(f"missing reused artifact: {path}")
    global_reused = (
        ROOT / "results/mimic_final_layer_live_atom_matched_effect_100k_v1/summary/report.md",
        ROOT / "results/external_benchmark_v1/summary/external_steering_cells.csv",
        ROOT / "results/external_benchmark_v1/summary/external_steering_target_profile.csv",
    )
    for path in global_reused:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing reused MIMIC benchmark artifact: {path}")

    with (RESULT / "release_component_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(check_rows[0]))
        writer.writeheader()
        writer.writerows(check_rows)
    with (RESULT / "reused_mimic_artifacts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(reused[0]))
        writer.writeheader()
        writer.writerows(reused)
    report = {
        "status": "pass" if not errors else "fail",
        "audit_pass": not errors,
        "errors": errors,
        "new_components": len(check_rows),
        "reused_artifacts": len(reused) + len(global_reused),
        "claim_boundary": "MIMIC-equivalent 7+5 target panel; not literal PTB-XL+ 49+9 target parity",
    }
    (RESULT / "release_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
