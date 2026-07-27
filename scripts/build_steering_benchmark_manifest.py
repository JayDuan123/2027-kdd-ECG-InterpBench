#!/usr/bin/env python
"""Build the frozen target registry and row-aligned steering manifest."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/sae_reconciliation/phenotype_steering"
OUT = ROOT / "results/sae_reconciliation/steering_benchmark_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phenotype-manifest", type=Path, default=OLD / "manifest.csv")
    p.add_argument("--concepts", type=Path, default=ROOT / "results/manifest/concepts_matrix.csv")
    p.add_argument("--metadata", type=Path, default=Path("/rhf/allocations/wq8/yd68/data/ptb-xl/1.0.3/ptbxl_database.csv"))
    p.add_argument("--out-dir", type=Path, default=OUT)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    base = pd.read_csv(a.phenotype_manifest)
    concepts = pd.read_csv(a.concepts)
    wanted = ["hr_ventricular", "qrs_duration", "qtc_fridericia", "st_amp_global", "qrst_angle"]
    missing = sorted(set(wanted) - set(concepts.columns))
    if missing:
        raise RuntimeError(f"Missing concept columns: {missing}")
    merged = base.merge(concepts[["ecg_id", *wanted]], on="ecg_id", how="left", validate="one_to_one")

    meta = pd.read_csv(a.metadata)
    meta["baseline_drift_present"] = meta["baseline_drift"].fillna("").astype(str).str.strip().ne("").astype(float)
    merged = merged.merge(
        meta[["ecg_id", "age", "sex", "baseline_drift_present"]],
        on="ecg_id", how="left", validate="one_to_one",
    )
    if not np.array_equal(merged["row_index"].to_numpy(), np.arange(len(merged))):
        raise RuntimeError("Manifest row order no longer matches the activation cache")

    registry = pd.DataFrame([
        ("lbbb", "binary", "conduction", "main", "LBBB; ILBBB excluded from negatives"),
        ("rbbb", "binary", "conduction", "main", "RBBB; IRBBB excluded from negatives"),
        ("pvc", "binary", "ectopy", "main", "Premature ventricular complexes"),
        ("avb1", "binary", "conduction", "main", "First-degree AV block"),
        ("lafb", "binary", "conduction", "main", "Left anterior fascicular block"),
        ("afib", "binary", "rate_rhythm", "positive_control", "Definition-proximal rhythm control"),
        ("hr_ventricular", "continuous", "rate_rhythm", "main", "Ventricular heart rate"),
        ("qrs_duration", "continuous", "interval", "main", "QRS duration"),
        ("qtc_fridericia", "continuous", "interval", "main", "Fridericia-corrected QT"),
        ("st_amp_global", "continuous", "st_t", "main", "Global ST amplitude"),
        ("qrst_angle", "continuous", "axis", "main", "QRS-T angle"),
        ("age", "continuous", "metadata", "nuisance_control", "Patient age"),
        ("sex", "binary", "metadata", "nuisance_control", "Recorded sex"),
        ("baseline_drift_present", "binary", "artifact", "nuisance_control", "PTB-XL baseline-drift flag"),
    ], columns=["target", "target_type", "family", "analysis_role", "notes"])
    registry["eligible"] = True
    registry["novelty_denominator"] = registry["analysis_role"].eq("main")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(a.out_dir / "manifest.csv", index=False)
    registry.to_csv(a.out_dir / "target_registry.csv", index=False)
    coverage = []
    for row in registry.itertuples(index=False):
        y = pd.to_numeric(merged[row.target], errors="coerce")
        coverage.append({
            "target": row.target,
            "target_type": row.target_type,
            "n_total": int(y.notna().sum()),
            "n_train": int(y[merged.split.eq("train")].notna().sum()),
            "n_val": int(y[merged.split.eq("val")].notna().sum()),
            "n_test": int(y[merged.split.eq("test")].notna().sum()),
            "test_positive": int((y[merged.split.eq("test")] == 1).sum()) if row.target_type == "binary" else "",
        })
    pd.DataFrame(coverage).to_csv(a.out_dir / "target_coverage.csv", index=False)
    print(registry.to_string(index=False))


if __name__ == "__main__":
    main()
