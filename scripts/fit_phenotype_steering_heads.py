#!/usr/bin/env python
from __future__ import annotations

import argparse,csv,json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/sae_reconciliation/phenotype_steering"
TARGETS=("lbbb","rbbb","pvc","avb1","lafb","afib")


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--acts",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    p.add_argument("--manifest",type=Path,default=BASE/"manifest.csv")
    p.add_argument("--out",type=Path,default=BASE/"frozen_heads.joblib")
    p.add_argument("--seed",type=int,default=4311)
    return p.parse_args()


def main():
    a=parse_args()
    import joblib
    from joblib import Parallel,delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
    from sklearn.preprocessing import StandardScaler
    rows=list(csv.DictReader(a.manifest.open())); x=np.asarray(np.load(a.acts,mmap_mode="r"),dtype=np.float32)
    split=np.array([r["split"] for r in rows]); scaler=StandardScaler().fit(x[split=="train"]); xs=scaler.transform(x)
    workers=min(12,int(os.environ.get("SLURM_CPUS_PER_TASK","1")))
    heads={}; metrics={}
    for ti,target in enumerate(TARGETS):
        y=np.array([np.nan if r[target]=="" else float(r[target]) for r in rows])
        tr=np.where((split=="train")&np.isfinite(y))[0]; va=np.where((split=="val")&np.isfinite(y))[0]; te=np.where((split=="test")&np.isfinite(y))[0]
        def fit_one(C,l1):
            clf=LogisticRegression(penalty="elasticnet",solver="saga",C=C,l1_ratio=l1,class_weight="balanced",
                                   max_iter=3000,random_state=a.seed+ti,n_jobs=1)
            clf.fit(xs[tr],y[tr].astype(int)); pv=clf.predict_proba(xs[va])[:,1]
            return roc_auc_score(y[va],pv),C,l1,clf
        grid=[(C,l1) for C in (.01,.1,1.,10.) for l1 in (.1,.5,.9)]
        fits=Parallel(n_jobs=workers,prefer="threads")(delayed(fit_one)(C,l1) for C,l1 in grid)
        val_auc,C,l1,clf=max(fits,key=lambda z:z[0]); pt=clf.predict_proba(xs[te])[:,1]
        heads[target]={"clf":clf,"valid_mask":np.isfinite(y),"labels":y}
        metrics[target]={"C":C,"l1_ratio":l1,"val_auroc":float(val_auc),
                         "test_auroc":float(roc_auc_score(y[te],pt)),
                         "test_auprc":float(average_precision_score(y[te],pt)),
                         "test_brier":float(brier_score_loss(y[te],pt)),
                         "n_iter":int(clf.n_iter_[0]),"converged":bool(clf.n_iter_[0]<clf.max_iter),
                         "n_train":len(tr),"n_val":len(va),"n_test":len(te)}
        print(target,json.dumps(metrics[target]),flush=True)
    payload={"scaler":scaler,"heads":heads,"metrics":metrics,"targets":TARGETS,
             "provenance":{"head_input":"raw CSFM Layer-6 mean embedding","selection":"validation AUROC","seed":a.seed}}
    a.out.parent.mkdir(parents=True,exist_ok=True); tmp=a.out.with_suffix(a.out.suffix+f".tmp.{os.getpid()}")
    joblib.dump(payload,tmp); tmp.replace(a.out)
    a.out.with_suffix(".metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")


if __name__=="__main__": main()
