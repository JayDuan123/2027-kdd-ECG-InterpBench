#!/usr/bin/env python
"""Transparent measurement closure for one external model/cohort pair."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
BASE=ROOT/"results/external_benchmark_v1"
CONCEPTS=("rr_mean_ms","heart_rate_bpm","qrs_duration_ms","pr_interval_ms","qt_like_ms",
          "qtc_bazett_ms","r_amp_global_mv","st_amp_global_mv","t_amp_global_mv")
B0=("rr_mean_ms","qrs_duration_ms","qt_like_ms")
FAMILY={"rr_mean_ms":"rate_rhythm","heart_rate_bpm":"rate_rhythm",
        "qrs_duration_ms":"interval","pr_interval_ms":"interval","qt_like_ms":"interval","qtc_bazett_ms":"interval",
        "r_amp_global_mv":"amplitude","st_amp_global_mv":"st_t","t_amp_global_mv":"st_t"}
MIN_DENOM=0.02


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--model-suffix",required=True); p.add_argument("--cohort",required=True)
    p.add_argument("--seed",type=int,default=20260712); return p.parse_args()


def fit_block(x:np.ndarray,y:np.ndarray,tr:np.ndarray,va:np.ndarray,te:np.ndarray):
    best=None
    for c in (.01,.1,1.,10.):
        model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                            LogisticRegression(C=c,class_weight="balanced",max_iter=3000,solver="liblinear",random_state=4311))
        model.fit(x[tr],y[tr]); score=roc_auc_score(y[va],model.decision_function(x[va]))
        if best is None or score>best[0]: best=(score,c,model)
    pred=best[2].decision_function(x[te]); return {"val_auroc":best[0],"best_C":best[1],
        "test_auroc":float(roc_auc_score(y[te],pred)),"test_auprc":float(average_precision_score(y[te],pred))}


def main():
    a=parse_args(); pair=BASE/a.model_suffix/a.cohort; bundle=joblib.load(pair/"frozen_heads.joblib")
    from scripts.run_external_layer_probe import concept_frame
    from scripts.run_external_pooled_leace import FULL_MANIFESTS
    ids=np.asarray(bundle["record_ids"]).astype(str); frame=concept_frame(FULL_MANIFESTS[a.cohort],a.cohort,ids)
    matrix=frame[list(CONCEPTS)].to_numpy(dtype=float); split=np.asarray(bundle["split"])
    tr=split=="train"; va=split=="val"; te=split=="test"; rng=np.random.default_rng(a.seed)
    leace_files=sorted((pair/"pooled_leace/summary").glob("pooled_leace_cells.csv"))
    if not leace_files: raise RuntimeError("Pooled LEACE summary required before closure")
    leace=pd.read_csv(leace_files[0]); encoded=sorted(set(leace.loc[leace.pooled_strict_encoded,"concept"]))
    rows=[]
    for task,head in bundle["heads"].items():
        y=np.asarray(head["labels"],dtype=int)
        rep=sorted(set(leace.loc[(leace.task==task)&leace.representation_causal,"concept"]))
        rep_families={FAMILY[name] for name in rep}
        bfam=[name for name in CONCEPTS if FAMILY[name] in rep_families]
        blocks={"B0":list(B0),"Ball":list(CONCEPTS),"Benc":encoded,"Brep":rep,"Bfam":bfam}
        metrics={}
        for name,names in blocks.items():
            if not names:
                metrics[name]={"status":"empty","test_auroc":np.nan,"test_auprc":np.nan,"dimension":0}; continue
            idx=[CONCEPTS.index(value) for value in names]; result=fit_block(matrix[:,idx],y,tr,va,te)
            metrics[name]={"status":"ok","dimension":len(names),"concepts":"|".join(names),**result}
        brand_dim=max(len(rep),1); brand=rng.normal(size=(len(y),brand_dim)); metrics["Brand"]={"status":"ok","dimension":brand_dim,**fit_block(brand,y,tr,va,te)}
        fm=float(head["metrics"]["test_auroc"]); brand_score=float(metrics["Brand"]["test_auroc"])
        brep=float(metrics["Brep"]["test_auroc"]) if metrics["Brep"]["status"]=="ok" else np.nan
        denom=fm-brand_score; ratio=(brep-brand_score)/denom if np.isfinite(brep) and denom>=MIN_DENOM else np.nan
        for block,result in metrics.items(): rows.append({"model_suffix":a.model_suffix,"cohort":a.cohort,"task":task,"block":block,**result})
        rows.append({"model_suffix":a.model_suffix,"cohort":a.cohort,"task":task,"block":"FM","status":"ok","dimension":np.nan,
                     "test_auroc":fm,"test_auprc":float(head["metrics"]["test_auprc"])})
        rows.append({"model_suffix":a.model_suffix,"cohort":a.cohort,"task":task,"block":"ClosureRatio","status":"stable" if np.isfinite(ratio) else "unstable_or_empty",
                     "dimension":len(rep),"test_auroc":ratio,"test_auprc":np.nan,"fm_minus_brand":denom})
    out=pair/"closure"; out.mkdir(parents=True,exist_ok=True); result=pd.DataFrame(rows)
    tmp=out/f"closure_results.csv.tmp.{os.getpid()}"; result.to_csv(tmp,index=False); tmp.replace(out/"closure_results.csv")
    ratios=result[result.block.eq("ClosureRatio")]; report={"model_suffix":a.model_suffix,"cohort":a.cohort,
        "tasks":int(ratios.task.nunique()),"stable_ratios":int(ratios.status.eq("stable").sum()),
        "concept_source":"waveform-derived external measurements","min_fm_minus_brand":MIN_DENOM}
    tmp=out/f"closure_summary.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(report,indent=2)+"\n"); tmp.replace(out/"closure_summary.json")
    print(json.dumps(report))


if __name__=="__main__": main()
