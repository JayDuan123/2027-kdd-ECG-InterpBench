#!/usr/bin/env python
"""Requirement-level audit for the compiled multi-scale SAE manuscript."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_ROOT = ROOT.parent / "overleaf_paper_benchmark"
FIGURE_STEMS = (
    "multiscale_benchmark_workflow",
    "multiscale_reconstruction_atlas",
    "multiscale_semantic_atlas",
    "multiscale_dead_feature_atlas",
    "multiscale_model_curves",
    "multiscale_stability_curves",
    "multiscale_patient_matched_scale_curves",
    "multiscale_patient_common_scale_auc",
    "multiscale_sparsity_sensitivity",
)
FORBIDDEN_PATTERNS = (
    "SAE has no selectivity",
    "no significant selectivity advantage",
    "not universally more selective",
    "PCA and ICA also have higher",
    "all 90 model--depth--seed cells",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    args = parser.parse_args()
    paper = args.paper_root
    errors: list[str] = []

    sync_path = paper / "generated/multiscale_sync_audit.json"
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else {}
    if sync.get("status") != "complete" or not sync.get("release_audit_pass"):
        errors.append("artifact sync audit is missing or failed")

    required = [
        paper / "main.tex",
        paper / "main.pdf",
        paper / "main.log",
        paper / "main.bbl",
        paper / "generated/paper_macros.tex",
        paper / "generated/paper_table_multiscale_patient.tex",
        paper / "generated/paper_table_accessibility_calibration.tex",
        paper / "generated/accessibility_calibration_audit.json",
        paper / "generated/dictionary_accessibility_audit.json",
        paper / "figures/final_layer_sae_dense_concept_count.pdf",
        paper / "figures/final_layer_sae_dense_concept_count.png",
    ]
    required.extend(
        paper / "figures" / f"{stem}.{extension}"
        for stem in FIGURE_STEMS
        for extension in ("pdf", "png")
    )
    for path in required:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing or empty manuscript artifact: {path}")

    source_paths = [paper / "main.tex", *sorted((paper / "sections").glob("*multiscale.tex"))]
    source_text = "\n".join(path.read_text() for path in source_paths if path.exists())
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in source_text.lower():
            errors.append(f"forbidden legacy claim remains: {pattern}")
    if "ECG-FM-InterpBench" not in source_text:
        errors.append("new matched-scale benchmark title/identity is missing")

    macros_path = paper / "generated/paper_macros.tex"
    if macros_path.exists() and "TBD" in macros_path.read_text():
        errors.append("audited paper macros still contain TBD")

    calibration_path = paper / "generated/accessibility_calibration_audit.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    if (
        not calibration.get("audit_pass")
        or calibration.get("protocol") != "accessibility_calibration_e8_v2"
        or int(calibration.get("complete_groups", 0)) != 30
        or int(calibration.get("random_replicates_per_group", 0)) != 20
    ):
        errors.append(f"v2 accessibility calibration audit is missing or invalid: {calibration}")
    for required_claim in (
        "20 matched random dictionaries",
        "native FM coordinates outperform SAE coordinates",
    ):
        if required_claim not in source_text:
            errors.append(f"required calibration claim is missing: {required_claim}")

    dictionary_path = paper / "generated/dictionary_accessibility_audit.json"
    dictionary = json.loads(dictionary_path.read_text()) if dictionary_path.exists() else {}
    if (
        not dictionary.get("audit_pass")
        or dictionary.get("protocol") != "dictionary_accessibility_e8_v1"
        or int(dictionary.get("complete_groups", 0)) != 30
        or int(dictionary.get("feature_rows", 0)) != 6240
        or int(dictionary.get("target_rows", 0)) != 180960
    ):
        errors.append(f"dictionary accessibility audit is missing or invalid: {dictionary}")
    for required_claim in (
        "CSFM L6 and L12 for the other encoders",
        "Native coordinates cover more concepts than SAE coordinates for every model",
    ):
        if required_claim not in source_text:
            errors.append(f"required dictionary claim is missing: {required_claim}")

    log_path = paper / "main.log"
    log_text = log_path.read_text(errors="replace") if log_path.exists() else ""
    for marker in (
        "Fatal error occurred",
        "! LaTeX Error:",
        "There were undefined references",
        "undefined on input line",
    ):
        if marker in log_text:
            errors.append(f"LaTeX log contains failure marker: {marker}")
    overfull = len(re.findall(r"Overfull \\hbox", log_text))
    widths = [
        float(value)
        for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)
    ]
    max_overfull_pt = max(widths or [0.0])

    pages = 0
    pdf_bytes = 0
    pdf_path = paper / "main.pdf"
    if pdf_path.exists():
        pdf_bytes = pdf_path.stat().st_size
        info = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        match = re.search(r"^Pages:\s+(\d+)$", info, flags=re.MULTILINE)
        pages = int(match.group(1)) if match else 0
        if pages <= 0 or pdf_bytes < 100_000:
            errors.append(f"compiled PDF is implausible: pages={pages}, bytes={pdf_bytes}")

    audit = {
        "status": "complete" if not errors else "failed",
        "audit_pass": not errors,
        "errors": errors,
        "paper_root": str(paper),
        "pages_total": pages,
        "pdf_bytes": pdf_bytes,
        "overfull_hbox_count": overfull,
        "max_overfull_hbox_pt": max_overfull_pt,
        "required_artifacts": len(required),
        "legacy_claims_absent": not any(
            pattern.lower() in source_text.lower() for pattern in FORBIDDEN_PATTERNS
        ),
        "audited_macros_present": macros_path.exists(),
        "accessibility_calibration_v2_present": (
            calibration.get("protocol") == "accessibility_calibration_e8_v2"
        ),
        "dictionary_accessibility_v1_present": (
            dictionary.get("protocol") == "dictionary_accessibility_e8_v1"
        ),
    }
    atomic_json(paper / "generated/paper_build_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
