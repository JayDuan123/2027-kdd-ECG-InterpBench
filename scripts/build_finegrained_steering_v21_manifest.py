#!/usr/bin/env python
"""Extend the frozen v2 registry with supported fine-grained SCP targets."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
OUT = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_1_finegrained"
LABELS = Path("/rhf/allocations/wq8/yd68/data/1.0.1/labels/ptbxl_statements.csv")

# These are the previously unregistered SCP statements that pass the frozen
# prevalence floor: >=150 train positives and >=30 positive test patients.
FINEGRAINED = {
    "irbbb": ("IRBBB", "conduction", "incomplete right bundle branch block"),
    "crbbb": ("CRBBB", "conduction", "complete right bundle branch block"),
    "clbbb": ("CLBBB", "conduction", "complete left bundle branch block"),
    "almi": ("ALMI", "mi_ischemia", "anterolateral myocardial infarction"),
    "injas": ("INJAS", "mi_ischemia", "anteroseptal subendocardial injury"),
    "iscin": ("ISCIN", "mi_ischemia", "inferior-lead ischemia"),
}


def main() -> None:
    manifest = pd.read_csv(V2 / "manifest.csv")
    registry = pd.read_csv(V2 / "candidate_target_registry.csv")
    labels = pd.read_csv(LABELS, usecols=["ecg_id", "scp_codes"])
    labels["ecg_id"] = labels.ecg_id.astype(int)
    parsed = labels.scp_codes.map(
        lambda value: {str(code): float(weight) for code, weight in ast.literal_eval(value)}
    )
    added = pd.DataFrame({"ecg_id": labels.ecg_id})
    rows = []
    for target, (code, family, description) in FINEGRAINED.items():
        added[target] = parsed.map(lambda values, key=code: float(values.get(key, 0.0) > 0.0))
        rows.append(
            {
                "target": target,
                "target_type": "binary",
                "family": family,
                "analysis_role": "main",
                "notes": f"Fine-grained PTB-XL SCP statement: {description}",
                "eligible": True,
                "novelty_denominator": True,
                "source": "ptbxl_scp_codes_v21",
                "source_key": code,
            }
        )

    manifest = manifest.merge(added, on="ecg_id", how="left", validate="one_to_one")
    new_registry = pd.DataFrame(rows)
    registry = pd.concat([registry, new_registry], ignore_index=True)
    if registry.target.duplicated().any():
        duplicates = registry.loc[registry.target.duplicated(), "target"].tolist()
        raise RuntimeError(f"Duplicate v2.1 targets: {duplicates}")
    if manifest[list(FINEGRAINED)].isna().any().any():
        raise RuntimeError("Missing v2.1 labels after ECG-ID alignment")

    OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT / "manifest.csv", index=False)
    registry.to_csv(OUT / "candidate_target_registry.csv", index=False)
    registry.to_csv(OUT / "target_registry.csv", index=False)
    pd.read_csv(V2 / "selected_operating_points.csv").to_csv(
        OUT / "selected_operating_points.csv", index=False
    )
    pd.read_csv(V2 / "sae_seed_fidelity_audit.csv").to_csv(
        OUT / "sae_seed_fidelity_audit.csv", index=False
    )

    coverage = []
    for spec in registry.itertuples(index=False):
        y = pd.to_numeric(manifest[spec.target], errors="coerce")
        row = {
            "target": spec.target,
            "target_type": spec.target_type,
            "family": spec.family,
            "analysis_role": spec.analysis_role,
            "source": spec.source,
        }
        for split in ("train", "val", "test"):
            valid = manifest.split.eq(split) & y.notna()
            row[f"{split}_valid"] = int(valid.sum())
            if spec.target_type == "binary":
                positive = valid & y.eq(1)
                row[f"{split}_positive"] = int(positive.sum())
                row[f"{split}_positive_patients"] = int(
                    manifest.loc[positive, "patient_id"].nunique()
                )
        row["missing_fraction"] = float(y.isna().mean())
        coverage.append(row)
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(OUT / "candidate_target_coverage.csv", index=False)
    added_coverage = coverage[coverage.target.isin(FINEGRAINED)]
    print(f"v2.1 targets={len(registry)} newly_added={len(FINEGRAINED)}")
    print(
        added_coverage[
            [
                "target",
                "family",
                "train_positive",
                "test_positive",
                "test_positive_patients",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
