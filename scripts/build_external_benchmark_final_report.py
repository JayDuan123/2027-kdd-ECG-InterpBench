#!/usr/bin/env python
"""Build an authoritative interim/final report for the 6x5 benchmark matrix."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"results/external_benchmark_v1"; OUT=BASE/"final"


def table(frame:pd.DataFrame)->str:
    cols=list(frame.columns); lines=["| "+" | ".join(cols)+" |","| "+" | ".join(["---"]*len(cols))+" |"]
    for values in frame.itertuples(index=False,name=None): lines.append("| "+" | ".join(str(v) for v in values)+" |")
    return "\n".join(lines)


def main():
    import subprocess
    subprocess.run([sys.executable,str(ROOT/"scripts/build_30pair_execution_matrix.py")],check=True)
    subprocess.run([sys.executable,str(ROOT/"scripts/audit_30pair_completion.py"),"--allow-incomplete",
                    "--out",str(BASE/"final/completion_audit.csv")],check=True)
    matrix=pd.read_csv(BASE/"execution_matrix_30pairs.csv"); external=matrix[matrix.cohort.ne("ptbxl")].copy()
    completion=json.loads((BASE/"final/completion_audit.json").read_text())
    complete=int(completion["complete_pairs"]); final=bool(completion["all_complete"])
    steering_path=BASE/"summary/external_steering_target_profile.csv"
    steering=pd.read_csv(steering_path) if steering_path.exists() else pd.DataFrame()
    rows=[]
    for r in external.itertuples(index=False):
        head_path=BASE/r.model_suffix/r.cohort/"frozen_heads_metrics.csv"
        heads=pd.read_csv(head_path) if head_path.exists() else pd.DataFrame()
        ok=heads[heads.status.eq("ok")] if not heads.empty else heads
        leace_path=BASE/r.model_suffix/r.cohort/"pooled_leace/summary/pooled_leace_cells.csv"
        leace=pd.read_csv(leace_path) if leace_path.exists() else pd.DataFrame()
        sae_metrics=list((BASE/r.model_suffix/r.cohort/"cohort_adapted_sae").glob("seed*/*.metrics.json"))
        sae_pass=0; sae_warnings=0
        for path in sae_metrics:
            x=json.loads(path.read_text()); sae_pass+=int(float(x["explained_variance"])>=.90)
            sae_warnings+=int(float(x["dead_fraction"])>.20)
        profile=steering[(steering.model==r.model)&(steering.cohort==r.cohort)] if not steering.empty else steering
        frozen=profile[profile.protocol.eq("frozen_atom")] if not profile.empty else profile
        adapted=profile[profile.protocol.eq("cohort_adapted_atom")] if not profile.empty else profile
        rows.append({
            "model":r.model,"cohort":r.cohort,"status":r.status,
            "records":int(r.pooled_records_indexed),"head_tasks":int(len(ok)),
            "min_head_AUROC":round(float(ok.test_auroc.min()),3) if len(ok) else np.nan,
            "layer_probe":bool(r.layer_probe_complete),
            "strict_measurements":int(r.strict_encoded_measurements) if pd.notna(r.strict_encoded_measurements) else np.nan,
            "LEACE_causal":int(leace.representation_causal.sum()) if len(leace) else 0,
            "LEACE_low_coupling":int((leace.representation_causal & leace.coupling_role.eq("low_coupling_candidate")).sum()) if len(leace) else 0,
            "closure_stable":int(r.closure_stable_ratios) if pd.notna(r.closure_stable_ratios) else 0,
            "frozen_T2_robust":int(frozen.robustness.eq("robust_3_of_3").sum()) if len(frozen) else 0,
            "adapted_SAE_pass":f"{sae_pass}/3" if sae_metrics else "0/3",
            "adapted_dead_warnings":sae_warnings,
            "adapted_T2_robust":int(adapted.robustness.eq("robust_3_of_3").sum()) if len(adapted) else 0,
        })
    profile=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); profile.to_csv(OUT/"external_pair_profile.csv",index=False)
    protocol_counts=pd.DataFrame()
    cells_path=BASE/"summary/external_steering_cells.csv"
    if cells_path.exists():
        cells=pd.read_csv(cells_path); protocol_counts=cells.groupby("protocol",as_index=False).agg(
            cells=("target","size"),tier0=("tier0_fidelity","sum"),tier1=("tier1_sparse_attribution","sum"),
            tier2=("tier2_selective_steering","sum"),tier3=("tier3_behavior_changing","sum"))
        protocol_counts.to_csv(OUT/"steering_protocol_counts.csv",index=False)
    lines=["# ECG FM Multi-Cohort Interpretability Benchmark","",
           f"**Status: {'FINAL' if final else 'INTERIM'} ({complete}/30 model-cohort pairs complete).**","",
           "No cross-model denominator or final consistency claim is frozen until all 30 pairs pass execution audit.","",
           "## Frozen Protocol","",
           "- Models: CSFM, CARDIAC-FM, ECG-FM, ECG-JEPA, HuBERT-ECG, ST-MEM.",
           "- Cohorts: PTB-XL, Chapman, CPSC, Ningbo, MIMIC-IV-ECG.",
           "- External concepts are waveform-derived continuous measurements; diagnoses remain tasks.",
           "- Probe uses train-only scaling, validation selection, shuffled/Gaussian controls, and held-out test reporting.",
           "- Pooled LEACE requires strict encoding, effective residual erasure, paired group bootstrap, random-rank control, and BH-FDR.",
           "- Closure compares B0/Ball/Benc/Brep/Bfam/Brand/FM; unstable denominators are not reported as ratios.",
           "- Frozen-Atom tests source dictionary transport; source-local re-ranks that fixed dictionary; cohort-adapted retrains the same relative SAE scale.",
           "- Chapman/CPSC/Ningbo use record-level inference; MIMIC uses patient-level splitting/bootstrap.",
           "- Pooled LEACE does not claim internal-layer continuation; SAE edits do not claim waveform-level causality.","",
           "## Pair Profile","",table(profile),""]
    if len(protocol_counts): lines += ["## Steering Protocol Counts","",table(protocol_counts),""]
    lines += ["## Claim Boundary","",
              "Gate failures are fidelity, transport, readout, or task-support limitations. They are not evidence that physiological information is absent.",
              "Cross-cohort results from waveform-derived measurements are Track-F evidence and are not presented as vendor-equivalent PTB-XL+ measurements."]
    (OUT/"external_benchmark_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":"final" if final else "interim","complete_pairs":complete,"total_pairs":30,"profile_rows":len(profile)}))


if __name__=="__main__": main()
