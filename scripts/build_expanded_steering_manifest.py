#!/usr/bin/env python
"""Build the frozen 55-target candidate registry for expanded steering."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"
OUT = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"
LABELS = Path("/rhf/allocations/wq8/yd68/data/1.0.1/labels/ptbxl_statements.csv")

DIAGNOSTIC = {
    "sarrh": ("SARRH", "rhythm", "main"), "stach": ("STACH", "rhythm", "main"),
    "sbrad": ("SBRAD", "rhythm", "main"), "pac": ("PAC", "ectopy", "main"),
    "ivcd": ("IVCD", "conduction", "main"), "lpr": ("LPR", "conduction", "definition_control"),
    "imi": ("IMI", "mi_ischemia", "main"), "asmi": ("ASMI", "mi_ischemia", "main"),
    "ilmi": ("ILMI", "mi_ischemia", "main"), "ami": ("AMI", "mi_ischemia", "main"),
    "isc_generic": ("ISC_", "mi_ischemia", "main"), "iscal": ("ISCAL", "mi_ischemia", "main"),
    "lvh": ("LVH", "hypertrophy", "main"), "lao_lae": ("LAO/LAE", "hypertrophy", "main"),
    "vclvh": ("VCLVH", "hypertrophy", "definition_control"),
    "abqrs": ("ABQRS", "morphology", "main"), "ndt": ("NDT", "st_t_form", "main"),
    "std_generic": ("STD_", "st_t_form", "main"), "nst_generic": ("NST_", "st_t_form", "main"),
    "qwave": ("QWAVE", "morphology", "main"), "nt_generic": ("NT_", "st_t_form", "main"),
    "lowt": ("LOWT", "st_t_form", "main"), "invt": ("INVT", "st_t_form", "main"),
    "norm": ("NORM", "diagnostic_control", "positive_control"),
    "sr": ("SR", "rhythm", "positive_control"), "pace": ("PACE", "device", "nuisance_control"),
}

MEASUREMENTS = {
    "rr_iqr": ("rate_rhythm", "main"), "p_found": ("rate_rhythm", "main"),
    "pq_interval": ("interval", "main"), "p_duration_global": ("interval", "main"),
    "p_axis_front": ("axis", "main"), "qrs_axis_front": ("axis", "main"),
    "t_axis_front": ("axis", "main"), "r_amp_precordial": ("amplitude", "main"),
    "q_amp_global": ("amplitude", "main"), "t_amp_global": ("st_t", "main"),
    "st_elev_inferior": ("st_t", "main"), "st_elev_anterior": ("st_t", "main"),
    "qrs_area_global": ("amplitude", "main"), "t_duration_global": ("st_t", "main"),
    "qt_interval": ("interval", "redundancy_control"),
}


def main() -> None:
    base = pd.read_csv(V1 / "manifest.csv"); registry = pd.read_csv(V1 / "target_registry.csv")
    labels = pd.read_csv(LABELS); labels.ecg_id = labels.ecg_id.astype(int)
    parsed = labels.scp_codes.map(lambda v: {str(k): float(x) for k, x in ast.literal_eval(v)})
    diag = pd.DataFrame({"ecg_id": labels.ecg_id})
    for target, (code, _, _) in DIAGNOSTIC.items():
        diag[target] = parsed.map(lambda d, c=code: float(d.get(c, 0) > 0))
    concepts = pd.read_csv(ROOT / "results/manifest/concepts_matrix.csv")
    expanded = base.merge(diag, on="ecg_id", how="left", validate="one_to_one")
    expanded = expanded.merge(concepts[["ecg_id", *MEASUREMENTS]], on="ecg_id", how="left", validate="one_to_one")
    if expanded[list(DIAGNOSTIC)].isna().any().any(): raise RuntimeError("Missing diagnostic labels after alignment")

    extra = []
    for target, (code, family, role) in DIAGNOSTIC.items():
        extra.append({"target": target, "target_type": "binary", "family": family, "analysis_role": role,
                      "notes": f"PTB-XL SCP code {code}", "eligible": True, "novelty_denominator": role == "main",
                      "source": "ptbxl_scp_codes", "source_key": code})
    for target, (family, role) in MEASUREMENTS.items():
        extra.append({"target": target, "target_type": "continuous", "family": family, "analysis_role": role,
                      "notes": "PTB-XL+ measurement", "eligible": True, "novelty_denominator": role == "main",
                      "source": "ptbxl_plus_measurement", "source_key": target})
    registry["source"] = registry.target.map(lambda t: "existing_v1")
    registry["source_key"] = registry.target
    registry = pd.concat([registry, pd.DataFrame(extra)], ignore_index=True)
    if registry.target.duplicated().any(): raise RuntimeError(f"Duplicate targets: {registry.loc[registry.target.duplicated(), 'target'].tolist()}")

    OUT.mkdir(parents=True, exist_ok=True); expanded.to_csv(OUT / "manifest.csv", index=False)
    registry.to_csv(OUT / "candidate_target_registry.csv", index=False)
    registry.to_csv(OUT / "target_registry.csv", index=False)
    pd.read_csv(V1 / "selected_operating_points.csv").to_csv(OUT / "selected_operating_points.csv", index=False)
    pd.read_csv(V1 / "sae_seed_fidelity_audit.csv").to_csv(OUT / "sae_seed_fidelity_audit.csv", index=False)
    coverage = []
    for spec in registry.itertuples(index=False):
        y = pd.to_numeric(expanded[spec.target], errors="coerce")
        row = {"target": spec.target, "target_type": spec.target_type, "family": spec.family,
               "analysis_role": spec.analysis_role, "source": spec.source}
        for split in ("train", "val", "test"):
            mask = expanded.split.eq(split) & y.notna(); row[f"{split}_valid"] = int(mask.sum())
            if spec.target_type == "binary":
                pos = mask & y.eq(1); row[f"{split}_positive"] = int(pos.sum())
                row[f"{split}_positive_patients"] = int(expanded.loc[pos, "patient_id"].nunique())
        row["missing_fraction"] = float(y.isna().mean()); coverage.append(row)
    coverage = pd.DataFrame(coverage); coverage.to_csv(OUT / "candidate_target_coverage.csv", index=False)
    print(f"candidate targets={len(registry)} main={int(registry.novelty_denominator.sum())}")
    print(coverage[["target", "target_type", "family", "analysis_role", "test_valid", "test_positive", "test_positive_patients", "missing_fraction"]]
          .fillna("").to_string(index=False))


if __name__ == "__main__": main()
