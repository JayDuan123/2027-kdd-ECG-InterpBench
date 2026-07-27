#!/usr/bin/env python
from __future__ import annotations

import csv,json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/sae_reconciliation/phenotype_steering"


def main():
    import joblib
    from joblib import Parallel,delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
    path=BASE/"frozen_heads.joblib"; bundle=joblib.load(path)
    rows=list(csv.DictReader((BASE/"manifest.csv").open())); split=np.array([r["split"] for r in rows])
    x=np.asarray(np.load(ROOT/"results/sae_reconciliation/lbbb_fig6/layer6_mean.npy",mmap_mode="r"),dtype=np.float32)
    xs=bundle["scaler"].transform(x)
    def refit(target,idx):
        m=dict(bundle["metrics"][target]); item=bundle["heads"][target]; y=np.asarray(item["labels"],float)
        tr=np.where((split=="train")&np.isfinite(y))[0]; va=np.where((split=="val")&np.isfinite(y))[0]; te=np.where((split=="test")&np.isfinite(y))[0]
        clf=LogisticRegression(penalty="elasticnet",solver="saga",C=float(m["C"]),l1_ratio=float(m["l1_ratio"]),
                               class_weight="balanced",max_iter=5000,tol=1e-3,random_state=4311+idx,n_jobs=1)
        clf.fit(xs[tr],y[tr].astype(int)); pv=clf.predict_proba(xs[va])[:,1]; pt=clf.predict_proba(xs[te])[:,1]
        m.update({"val_auroc_final":float(roc_auc_score(y[va],pv)),"test_auroc":float(roc_auc_score(y[te],pt)),
                  "test_auprc":float(average_precision_score(y[te],pt)),"test_brier":float(brier_score_loss(y[te],pt)),
                  "n_iter_final":int(clf.n_iter_[0]),"converged":bool(clf.n_iter_[0]<clf.max_iter),
                  "final_max_iter":5000,"final_tol":1e-3})
        return target,clf,m
    fitted=Parallel(n_jobs=len(bundle["metrics"]),prefer="threads")(
        delayed(refit)(target,idx) for idx,target in enumerate(bundle["metrics"]))
    repaired=[]
    for target,clf,m in fitted:
        if not m["converged"]: raise RuntimeError(f"{target} still did not converge under uniform final fit")
        bundle["heads"][target]["clf"]=clf; bundle["metrics"][target]=m; repaired.append(target)
        print(target,json.dumps(m),flush=True)
    tmp=path.with_suffix(path.suffix+f".tmp.{os.getpid()}"); joblib.dump(bundle,tmp); tmp.replace(path)
    path.with_suffix(".metrics.json").write_text(json.dumps(bundle["metrics"],indent=2)+"\n")
    (BASE/"head_repair.json").write_text(json.dumps({"repaired":repaired,"uniform_final_fit":{"tol":1e-3,"max_iter":5000},
                                                      "all_converged":all(m["converged"] for m in bundle["metrics"].values())},indent=2)+"\n")


if __name__=="__main__": main()
