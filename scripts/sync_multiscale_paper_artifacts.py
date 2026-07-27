#!/usr/bin/env python
"""Atomically sync audited multi-scale SAE paper artifacts into the Overleaf tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> dict[str, Any]:
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty source artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"artifact checksum mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "destination": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": source_hash,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    args = parser.parse_args()

    release = json.loads((args.root / "final_release_audit.json").read_text())
    facts_audit = json.loads((args.root / "paper_facts_audit.json").read_text())
    if not release.get("audit_pass"):
        raise RuntimeError(f"release audit has not passed: {release.get('errors')}")
    if facts_audit.get("status") != "complete" or not facts_audit.get("release_audit_pass"):
        raise RuntimeError(f"paper facts audit has not passed: {facts_audit}")

    mappings = [
        (args.root / "paper_macros.tex", args.paper_root / "generated/paper_macros.tex"),
        (args.root / "paper_table_multiscale.tex", args.paper_root / "generated/paper_table_multiscale.tex"),
        (
            args.root / "paper_table_multiscale_patient.tex",
            args.paper_root / "generated/paper_table_multiscale_patient.tex",
        ),
        (
            ROOT
            / "results/accessibility_calibration_e8_v2/summary/paper_table_accessibility_calibration_v2.tex",
            args.paper_root / "generated/paper_table_accessibility_calibration.tex",
        ),
        (
            ROOT / "results/accessibility_calibration_e8_v2/summary/audit.json",
            args.paper_root / "generated/accessibility_calibration_audit.json",
        ),
        (
            ROOT / "results/dictionary_accessibility_e8_v1/summary/audit.json",
            args.paper_root / "generated/dictionary_accessibility_audit.json",
        ),
    ]
    for stem in ("final_layer_sae_dense_concept_count",):
        for extension in ("pdf", "png"):
            mappings.append(
                (
                    ROOT
                    / "results/dictionary_accessibility_e8_v1/summary"
                    / f"{stem}.{extension}",
                    args.paper_root / "figures" / f"{stem}.{extension}",
                )
            )
    for name in (
        "paper_facts.json",
        "paper_facts.md",
        "paper_facts_audit.json",
        "final_release_audit.json",
        "final_release_checksums.sha256",
        "figure_audit.json",
    ):
        mappings.append((args.root / name, args.paper_root / "generated" / name))
    for stem in FIGURE_STEMS:
        for extension in ("pdf", "png"):
            mappings.append(
                (
                    args.root / "figures" / f"{stem}.{extension}",
                    args.paper_root / "figures" / f"{stem}.{extension}",
                )
            )

    copied = [atomic_copy(source, destination) for source, destination in mappings]
    audit = {
        "status": "complete",
        "release_audit_pass": True,
        "paper_facts_audit_pass": True,
        "source_root": str(args.root),
        "paper_root": str(args.paper_root),
        "copied_artifacts": len(copied),
        "artifacts": copied,
    }
    atomic_json(args.paper_root / "generated/multiscale_sync_audit.json", audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "artifacts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
