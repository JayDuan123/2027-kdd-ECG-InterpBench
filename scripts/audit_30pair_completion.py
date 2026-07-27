#!/usr/bin/env python
"""Requirement-level completion audit for the 6-model x 5-cohort benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT/"results/external_benchmark_v1"
MODELS={"CSFM":"csfm_cu118_commons","CARDIAC-FM":"cardiac_fm_cu118_commons","ECG-FM":"ecg_fm_cu118_commons",
        "ECG-JEPA":"ecg_jepa_cu118_commons","HuBERT-ECG":"hubert_ecg_cu118_commons","ST-MEM":"st_mem_cu118_commons"}
COHORTS={"chapman_f":(10208,4,7),"cpsc_f":(6876,2,3),"ningbo_f":(34631,4,7),"mimic_f":(100000,5,8)}
POOLED_AUDITS={
    "chapman_f":BASE/"chapman_cpsc_pooled_shard_audit.csv","cpsc_f":BASE/"chapman_cpsc_pooled_shard_audit.csv",
    "ningbo_f":BASE/"ningbo_pooled_shard_audit.csv","mimic_f":BASE/"mimic_100k_pooled_shard_audit.csv"}
LAYER_AUDITS={
    "chapman_f":BASE/"chapman_cpsc_layer_atlas_audit.csv","cpsc_f":BASE/"chapman_cpsc_layer_atlas_audit.csv",
    "ningbo_f":BASE/"ningbo_layer_atlas_audit.csv","mimic_f":BASE/"mimic_layer_atlas_audit.csv"}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=BASE/"final/completion_audit.csv")
    p.add_argument("--allow-incomplete",action="store_true"); return p.parse_args()


def payload(path:Path)->dict:
    try: return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError): return {}


def main():
    a=parse_args(); rows=[]
    comparison=ROOT/"results/analysis/model_comparison"
    ptb_manifest=pd.read_csv(comparison/"final_v1_manifest.csv").set_index("model")
    ptb_probe=pd.read_csv(comparison/"cleanup_audit/probe_encoding_strict_summary.csv").set_index("model")
    ptb_leace=pd.read_csv(comparison/"cleanup_audit/continuation_canonical_strict_summary.csv").set_index("model")
    ptb_closure=pd.read_csv(comparison/"figure5_closure_by_task.csv")
    source_sae=pd.read_csv(ROOT/"results/sae_reconciliation/matched_scale_v1/training_manifest.csv")
    for model,suffix in MODELS.items():
        ptb_missing=[]
        required=("probe","linear_task_head","closure_ball","continuation_erasure","bootstrap_ci","bh_fdr")
        if model not in ptb_manifest.index or any(str(ptb_manifest.loc[model,key]).lower()!="yes" for key in required):
            ptb_missing.append("final_v1_manifest")
        if model not in ptb_probe.index or int(ptb_probe.loc[model,"concept_count"])!=49: ptb_missing.append("strict_probe_49")
        if model not in ptb_leace.index or int(ptb_leace.loc[model,"tested_after_task_dedup"])<=0: ptb_missing.append("leace_summary")
        if ptb_closure[ptb_closure.model.eq(model)].task_id.nunique()<=0: ptb_missing.append("closure_tasks")
        sae_rows=source_sae[source_sae.feature_suffix.eq(suffix)]; valid_source=0
        for row in sae_rows.itertuples(index=False):
            checkpoint=Path(row.checkpoint); metrics=checkpoint.with_suffix(".metrics.json"); x=payload(metrics)
            valid_source+=int(checkpoint.exists() and x.get("step")==8000 and int(x.get("N",0))==8*int(x.get("d",0)))
        if valid_source!=3: ptb_missing.append(f"source_sae:{valid_source}/3")
        rows.append({"model":model,"cohort":"ptbxl","complete":not ptb_missing,"missing":";".join(ptb_missing)})
        for cohort,(expected_records,expected_tasks,expected_leace) in COHORTS.items():
            pair=BASE/suffix/cohort; missing=[]
            pooled=payload(ROOT/f"results/activations_external_full_v1/pooled/{suffix}/{cohort}/index_report.json")
            layer=payload(ROOT/f"results/activations_external_full_v1/layer_atlas/{suffix}/{cohort}/index_report.json")
            if int(pooled.get("n_records",0))!=expected_records: missing.append(f"pooled_records:{pooled.get('n_records',0)}/{expected_records}")
            if int(layer.get("n_records",0))!=4096: missing.append(f"layer_records:{layer.get('n_records',0)}/4096")
            for label,path,target in (("pooled_audit",POOLED_AUDITS[cohort],expected_records),("layer_audit",LAYER_AUDITS[cohort],4096)):
                if not path.exists():
                    missing.append(label+":missing"); continue
                audit=pd.read_csv(path); part=audit[(audit.model_suffix==suffix)&(audit.cohort==cohort)]
                valid=not part.empty and part.status.eq("complete").all() and int(pd.to_numeric(part.loaded).sum())==target
                if not valid: missing.append(f"{label}:invalid_or_incomplete")
            head=payload(pair/"head_summary.json")
            if int(head.get("tasks_trained",0))!=expected_tasks: missing.append(f"head_tasks:{head.get('tasks_trained',0)}/{expected_tasks}")
            if cohort=="mimic_f" and head.get("split_unit")!="patient": missing.append("mimic_split_not_patient")
            probe=payload(BASE/"layer_probe"/suffix/cohort/"probe_report.json")
            if int(probe.get("concepts_scored",0))!=9: missing.append(f"layer_concepts:{probe.get('concepts_scored',0)}/9")
            leace=payload(pair/"pooled_leace/summary/pooled_leace_summary.json")
            if int(leace.get("cells",0))!=expected_leace: missing.append(f"leace_cells:{leace.get('cells',0)}/{expected_leace}")
            closure=payload(pair/"closure/closure_summary.json")
            if int(closure.get("tasks",0))!=expected_tasks: missing.append(f"closure_tasks:{closure.get('tasks',0)}/{expected_tasks}")
            metrics=list((pair/"cohort_adapted_sae").glob("seed*/*.metrics.json")); valid_metrics=0
            for path in metrics:
                x=payload(path); d=int(x.get("d",0)); N=int(x.get("N",0)); k=int(x.get("k",0))
                valid_metrics+=int(x.get("step")==8000 and N==8*d and k*8==d and path.with_name(path.name.replace(".metrics.json",".pt")).exists())
            if valid_metrics!=3: missing.append(f"adapted_sae:{valid_metrics}/3")
            expected_seed_cells=expected_tasks*3; steering=pair/"steering"
            skipped=len(list(steering.glob("skipped/seed*/*.json")))
            for protocol in ("frozen_atom","local_atom","cohort_adapted_atom"):
                count=len(list(steering.glob(f"{protocol}/seed*/*/result.json")))
                if count+skipped<expected_seed_cells: missing.append(f"{protocol}:{count}+{skipped}/{expected_seed_cells}")
            rows.append({"model":model,"cohort":cohort,"complete":not missing,"missing":";".join(missing)})
    frame=pd.DataFrame(rows); a.out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.out,index=False)
    report={"pairs":len(frame),"complete_pairs":int(frame.complete.sum()),"incomplete_pairs":int((~frame.complete).sum()),
            "all_complete":bool(frame.complete.all())}
    a.out.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report)); print(frame[~frame.complete].to_string(index=False))
    if not report["all_complete"] and not a.allow_incomplete: raise SystemExit(2)


if __name__=="__main__": main()
