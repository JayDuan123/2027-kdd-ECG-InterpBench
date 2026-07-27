#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE


def parse_args():
    p=argparse.ArgumentParser()
    base=ROOT/"results/sae_reconciliation/lbbb_fig6"
    p.add_argument("--acts",type=Path,default=base/"layer6_mean.npy")
    p.add_argument("--manifest",type=Path,default=base/"manifest.csv")
    p.add_argument("--concepts",type=Path,default=ROOT/"results/manifest/concepts_matrix.csv")
    p.add_argument("--checkpoint",type=Path,default=base/"checkpoints/batchtopk_8192_k128.pt")
    p.add_argument("--predictors",type=Path,default=base/"frozen_predictors.joblib")
    p.add_argument("--step1",type=Path,default=base/"step1_result.json")
    p.add_argument("--out",type=Path,default=base/"step2_knob_results.json")
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=4311)
    p.add_argument("--device",default="cuda")
    return p.parse_args()


def encode_batches(sae,x,batch,device):
    import torch
    out=[]; sae.eval()
    with torch.no_grad():
        for lo in range(0,len(x),batch):
            raw=torch.from_numpy(np.asarray(x[lo:lo+batch],dtype=np.float32)).to(device)
            out.append(sae.encode(raw).cpu().numpy())
    return np.concatenate(out)


def decode_batches(sae,z,batch,device):
    import torch
    out=[]
    with torch.no_grad():
        for lo in range(0,len(z),batch):
            code=torch.from_numpy(np.asarray(z[lo:lo+batch],dtype=np.float32)).to(device)
            out.append(sae.decode(code).cpu().numpy())
    return np.concatenate(out)


def patient_ci(values,patients,n_boot,rng):
    unique=np.unique(patients); stats=[]
    for _ in range(n_boot):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        stats.append(float(np.concatenate([values[patients==p] for p in sampled]).mean()))
    return [float(np.quantile(stats,.025)),float(np.quantile(stats,.975))]


def summarize(values,patients,n_boot,rng):
    return {"mean_delta":float(values.mean()),"median_delta":float(np.median(values)),
            "patient_bootstrap_95ci":patient_ci(values,patients,n_boot,rng)}


def main():
    a=parse_args(); import joblib,torch
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score,roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rows=list(csv.DictReader(a.manifest.open()))
    x=np.asarray(np.load(a.acts,mmap_mode="r"),dtype=np.float32)
    split=np.array([r["split"] for r in rows]); patients=np.array([r["patient_id"] for r in rows])
    lbbb=np.array([int(r["lbbb"]) for r in rows]); af=np.array([int(r["af"]) for r in rows])
    ecg_ids=[r["ecg_id"] for r in rows]
    cache=joblib.load(a.predictors)
    l_scaler,l_clf,l_info=cache["lbbb"]; af_scaler,af_clf,af_info=cache["af"]
    ck=torch.load(a.checkpoint,map_location=a.device); cfg=ck["config"]
    sae=BatchTopKSAE(x.shape[1],int(cfg["n_features"]),int(cfg["k"])).to(a.device)
    sae.load_state_dict(ck["model"]); sae.eval()
    dec=sae.W_dec.detach().cpu().numpy(); sigma=sae.sigma.detach().cpu().numpy()
    tr=np.where(split=="train")[0]; te=np.where(split=="test")[0]
    ztr=encode_batches(sae,x[tr],256,a.device)
    grad_l=(l_clf.coef_[0]/l_scaler.scale_)*sigma@dec
    lpos_local=np.where(lbbb[tr]==1)[0]; lneg_local=np.where(lbbb[tr]==0)[0]
    ig=(ztr[lpos_local]*grad_l).mean(0)
    ig_rank=np.argsort(ig)[::-1]
    target=int(ig_rank[0])
    step1=json.loads(a.step1.read_text())
    random_atom=int(step1["atoms"]["matched_random"])
    af_atom=int(step1["atoms"]["orthogonal_af"])

    pos_idx=np.where((split=="test")&(lbbb==1))[0]
    pos_idx=pos_idx[np.argsort(np.array([int(ecg_ids[i]) for i in pos_idx]))][:80]
    zpos=encode_batches(sae,x[pos_idx],len(pos_idx),a.device)
    base_raw=decode_batches(sae,zpos,len(zpos),a.device)
    base_l=l_clf.decision_function(l_scaler.transform(base_raw))
    base_af=af_clf.decision_function(af_scaler.transform(base_raw))
    rng=np.random.default_rng(a.seed)

    def eval_edit(name,indices,values):
        edited=zpos.copy()
        if np.isscalar(values): edited[:,indices]=values
        else: edited[:,indices]=np.asarray(values)[indices]
        raw=decode_batches(sae,edited,len(edited),a.device)
        dl=l_clf.decision_function(l_scaler.transform(raw))-base_l
        da=af_clf.decision_function(af_scaler.transform(raw))-base_af
        return {"name":name,"atoms":[int(i) for i in np.atleast_1d(indices)],
                "lbbb":summarize(dl,patients[pos_idx],a.bootstrap,rng),
                "af_offtarget":summarize(da,patients[pos_idx],a.bootstrap,rng)}

    knob_2a={}
    for n in (1,5,10): knob_2a[f"top{n}"]=eval_edit(f"IG top-{n} zero",ig_rank[:n],0.0)
    centroid=ztr.mean(0)
    knob_2b={"zero":eval_edit("single IG zero",[target],0.0),
             "population_centroid":eval_edit("single IG train-population centroid",[target],centroid)}

    cav_atom=int(np.argmax(grad_l))
    mean_diff=ztr[lpos_local].mean(0)-ztr[lneg_local].mean(0)
    pooled_sd=np.maximum(ztr.std(0),1e-8)
    activation_atom=int(np.argmax(mean_diff/pooled_sd))
    knob_2c={"ig":eval_edit("IG selection",[target],0.0),
             "cav_geometry":eval_edit("frozen-head CAV geometry selection",[cav_atom],0.0),
             "activation_association":eval_edit("standardized activation mean-difference selection",[activation_atom],0.0)}

    # Continuous ST knob: train-only standardized target, validation alpha.
    concepts={r["ecg_id"]:r for r in csv.DictReader(a.concepts.open())}
    st=np.array([float(concepts[e]["st_amp_global"]) for e in ecg_ids],dtype=np.float64)
    y_scaler=StandardScaler().fit(st[tr,None]); sty=y_scaler.transform(st[:,None]).ravel()
    va=np.where(split=="val")[0]
    xs=l_scaler.transform(x)
    best=None
    for alpha in (0.1,1.0,10.0,100.0):
        reg=Ridge(alpha=alpha,solver="lsqr").fit(xs[tr],sty[tr])
        score=r2_score(sty[va],reg.predict(xs[va]))
        if best is None or score>best[0]: best=(score,alpha,reg)
    st_val_r2,st_alpha,st_reg=best
    st_grad=(st_reg.coef_/l_scaler.scale_)*sigma@dec
    high_st_local=np.where(sty[tr]>=np.quantile(sty[tr],.75))[0]
    st_ig=(ztr[high_st_local]*st_grad).mean(0); st_atom=int(np.argmax(st_ig))
    st_test=te[np.argsort(st[te])[::-1]][:80]
    zst=encode_batches(sae,x[st_test],len(st_test),a.device)
    st_base_raw=decode_batches(sae,zst,len(zst),a.device)
    st_base=st_reg.predict(l_scaler.transform(st_base_raw))
    freq=(ztr>0).mean(0)
    mag=np.divide(ztr.sum(0),(ztr>0).sum(0),out=np.zeros(sae.N),where=(ztr>0).sum(0)>0)
    st_dist=np.abs(np.log((freq+1e-6)/(freq[st_atom]+1e-6)))+np.abs(np.log((mag+1e-6)/(mag[st_atom]+1e-6)))
    st_dist[[st_atom,target]]=np.inf; st_random=int(np.argmin(st_dist))
    def st_effect(atom):
        zedit=zst.copy(); zedit[:,atom]=0
        edit_raw=decode_batches(sae,zedit,len(zedit),a.device)
        delta=st_reg.predict(l_scaler.transform(edit_raw))-st_base
        return summarize(delta,patients[st_test],a.bootstrap,rng)
    knob_2d={"concept":"st_amp_global","selection":"IG on standardized ridge output among train top-quartile ST",
             "atom":st_atom,"alpha":st_alpha,"val_r2":float(st_val_r2),
             "test_r2_raw_embedding":float(r2_score(sty[te],st_reg.predict(xs[te]))),
             "effects_standardized_ST":{"target_st_atom":st_effect(st_atom),
                                         "frequency_magnitude_matched_random":st_effect(st_random),
                                         "other_concept_lbbb_atom":st_effect(target)},
             "control_atoms":{"matched_random":st_random,"lbbb_other":target,
                              "target_random_frequency_ratio":float(freq[st_random]/max(freq[st_atom],1e-12)),
                              "target_random_magnitude_ratio":float(mag[st_random]/max(mag[st_atom],1e-12))}}

    # Strict readout/WBI knob on the complete held-out test split.
    zte=encode_batches(sae,x[te],256,a.device); recon=decode_batches(sae,zte,256,a.device)
    clean_l=l_clf.decision_function(l_scaler.transform(recon)); clean_af=af_clf.decision_function(af_scaler.transform(recon))
    def task_metrics(atom):
        zz=zte.copy(); zz[:,atom]=0; raw=decode_batches(sae,zz,256,a.device)
        pred_l=l_clf.decision_function(l_scaler.transform(raw)); pred_af=af_clf.decision_function(af_scaler.transform(raw))
        base_l_auc=roc_auc_score(lbbb[te],clean_l); base_af_auc=roc_auc_score(af[te],clean_af)
        dl=float(base_l_auc-roc_auc_score(lbbb[te],pred_l)); da=float(base_af_auc-roc_auc_score(af[te],pred_af))
        return {"lbbb_auroc_drop":dl,"af_auroc_drop":da,"wbi_abs_af_over_lbbb":abs(da)/(abs(dl)+1e-8)}
    raw_l_auc=roc_auc_score(lbbb[te],l_clf.decision_function(l_scaler.transform(x[te])))
    raw_af_auc=roc_auc_score(af[te],af_clf.decision_function(af_scaler.transform(x[te])))
    recon_l_auc=roc_auc_score(lbbb[te],clean_l); recon_af_auc=roc_auc_score(af[te],clean_af)
    knob_2e={"scope":"final-layer frozen readouts; AF is the off-target task",
             "reconstruction_retention":{"lbbb_raw_auroc":float(raw_l_auc),"lbbb_recon_auroc":float(recon_l_auc),
                                         "af_raw_auroc":float(raw_af_auc),"af_recon_auroc":float(recon_af_auc)},
             "ig_target_atom":task_metrics(target),"matched_random_atom":task_metrics(random_atom),
             "orthogonal_af_atom":task_metrics(af_atom)}

    result={"step1_replication_pass":bool(step1["replication_pass"]),
            "2a_single_to_group":knob_2a,"2b_zero_to_centroid":knob_2b,
            "2c_ig_to_other_selection":knob_2c,"2d_discrete_to_continuous_ST":knob_2d,
            "2e_frozen_logit_to_strict_WBI":knob_2e,
            "guards":{"selection_split":"train only","evaluation_split":"held-out test",
                      "patient_bootstrap_samples":a.bootstrap,"checkpoint":"self-trained architecture-matched BatchTopK"}}
    a.out.parent.mkdir(parents=True,exist_ok=True); tmp=a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result,indent=2)+"\n"); tmp.replace(a.out)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
