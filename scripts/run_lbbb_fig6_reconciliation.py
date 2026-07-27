#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json, os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--acts",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    p.add_argument("--manifest",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/manifest.csv")
    p.add_argument("--checkpoint",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/checkpoints/batchtopk_8192_k128.pt")
    p.add_argument("--out",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/step1_result.json")
    p.add_argument("--predictor-cache",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/frozen_predictors.joblib")
    p.add_argument("--n-test-positive",type=int,default=80)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--seed",type=int,default=4311)
    p.add_argument("--device",default="cuda")
    return p.parse_args()


def fit_predictor(x, y, split, seed):
    from joblib import Parallel, delayed
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    tr=np.where(split=="train")[0]; va=np.where(split=="val")[0]; te=np.where(split=="test")[0]
    scaler=StandardScaler().fit(x[tr]); xs=scaler.transform(x)
    def fit_one(C, l1):
        clf=LogisticRegression(penalty="elasticnet",solver="saga",C=C,l1_ratio=l1,
                               class_weight="balanced",max_iter=3000,random_state=seed,n_jobs=1)
        clf.fit(xs[tr],y[tr])
        score=roc_auc_score(y[va],clf.predict_proba(xs[va])[:,1])
        return score,C,l1,clf
    grid=[(C,l1) for C in (0.01,0.1,1.0,10.0) for l1 in (0.1,0.5,0.9)]
    workers=min(len(grid), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    fits=Parallel(n_jobs=workers, prefer="threads")(delayed(fit_one)(C,l1) for C,l1 in grid)
    score,C,l1,clf=max(fits,key=lambda item:item[0])
    return scaler,clf,{"C":C,"l1_ratio":l1,"val_auroc":score,
                      "test_auroc":float(roc_auc_score(y[te],clf.predict_proba(xs[te])[:,1])),
                      "n_iter":int(clf.n_iter_[0]),"converged":bool(clf.n_iter_[0] < clf.max_iter),
                      "grid_workers":workers}


def encode_batches(model, x, batch, device):
    import torch
    zs=[]
    model.eval()
    with torch.no_grad():
        for lo in range(0,len(x),batch):
            zs.append(model.encode(torch.from_numpy(np.asarray(x[lo:lo+batch],dtype=np.float32)).to(device)).cpu().numpy())
    return np.concatenate(zs)


def patient_ci(values, patients, n_boot, rng):
    unique=np.unique(patients); stats=[]
    for _ in range(n_boot):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        chunks=[values[patients==p] for p in sampled]
        stats.append(float(np.concatenate(chunks).mean()))
    return [float(np.quantile(stats,.025)),float(np.quantile(stats,.975))]


def main():
    a=parse_args(); import torch
    rows=list(csv.DictReader(a.manifest.open())); x=np.asarray(np.load(a.acts,mmap_mode="r"),dtype=np.float32)
    split=np.array([r["split"] for r in rows]); lbbb=np.array([int(r["lbbb"]) for r in rows]); af=np.array([int(r["af"]) for r in rows])
    patients=np.array([r["patient_id"] for r in rows]); ecg=np.array([int(r["ecg_id"]) for r in rows])
    import joblib
    if a.predictor_cache.exists():
        cache=joblib.load(a.predictor_cache)
        l_scaler,l_clf,l_info=cache["lbbb"]
        af_scaler,af_clf,af_info=cache["af"]
    else:
        l_scaler,l_clf,l_info=fit_predictor(x,lbbb,split,a.seed)
        af_scaler,af_clf,af_info=fit_predictor(x,af,split,a.seed+1)
        a.predictor_cache.parent.mkdir(parents=True,exist_ok=True)
        cache_tmp=a.predictor_cache.with_suffix(a.predictor_cache.suffix+".tmp")
        joblib.dump({"lbbb":(l_scaler,l_clf,l_info),"af":(af_scaler,af_clf,af_info)},cache_tmp)
        cache_tmp.replace(a.predictor_cache)
    ck=torch.load(a.checkpoint,map_location=a.device); cfg=ck["config"]
    sae=BatchTopKSAE(x.shape[1],int(cfg["n_features"]),int(cfg["k"])).to(a.device)
    sae.load_state_dict(ck["model"]); sae.eval()
    tr=np.where(split=="train")[0]
    z_train=encode_batches(sae,x[tr],256,a.device)
    # For a linear frozen predictor, integrated gradients from zero in atom
    # space are exact: IG_i = z_i * d(logit)/dz_i.
    dec=sae.W_dec.detach().cpu().numpy(); sigma=sae.sigma.detach().cpu().numpy()
    grad_l=(l_clf.coef_[0]/l_scaler.scale_)*sigma @ dec
    grad_af=(af_clf.coef_[0]/af_scaler.scale_)*sigma @ dec
    tr_lpos=np.where((split[tr]=="train") & (lbbb[tr]==1))[0]
    tr_afpos=np.where((split[tr]=="train") & (af[tr]==1))[0]
    attr_l=(z_train[tr_lpos]*grad_l).mean(0); attr_af=(z_train[tr_afpos]*grad_af).mean(0)
    target=int(np.argmax(attr_l))
    directions=dec/np.maximum(np.linalg.norm(dec,axis=0,keepdims=True),1e-12)
    cosine=np.abs(directions[:,target]@directions)
    near_zero_lbbb=np.abs(attr_l)<=np.quantile(np.abs(attr_l),.10)
    af_candidates=np.where((cosine<0.05)&near_zero_lbbb)[0]
    if len(af_candidates)==0:
        raise RuntimeError("no AF atom satisfies decoder orthogonality and near-zero LBBB attribution")
    other=int(af_candidates[np.argmax(attr_af[af_candidates])])
    freq=(z_train>0).mean(0); mag=np.divide(z_train.sum(0),(z_train>0).sum(0),out=np.zeros(sae.N),where=(z_train>0).sum(0)>0)
    dist=np.abs(np.log((freq+1e-6)/(freq[target]+1e-6)))+np.abs(np.log((mag+1e-6)/(mag[target]+1e-6)))
    dist[[target,other]]=np.inf; random_atom=int(np.argmin(dist))

    test_pos=np.where((split=="test")&(lbbb==1))[0]
    test_pos=test_pos[np.argsort(ecg[test_pos])][:a.n_test_positive]
    z=encode_batches(sae,x[test_pos],len(test_pos),a.device)
    def logits_from_raw(raw):
        return l_clf.decision_function(l_scaler.transform(raw))
    def logits(zcode):
        with torch.no_grad():
            raw=sae.decode(torch.from_numpy(zcode.astype(np.float32)).to(a.device)).cpu().numpy()
        return logits_from_raw(raw)
    base=logits(z)
    effects={}
    for name,atom in (("target_ig_lbbb",target),("matched_random",random_atom),("orthogonal_af",other)):
        edited=z.copy(); edited[:,atom]=0
        effects[name]=logits(edited)-base
    # Dense circularity control. Project the reconstructed activation along the
    # frozen predictor-gradient direction in SAE-normalised dense space. Scale
    # this edit so its mean unscaled attribution matches the selected atom's
    # mean attribution, then actually decode and evaluate it through the head.
    with torch.no_grad():
        z_t=torch.from_numpy(z.astype(np.float32)).to(a.device)
        h_norm=sae.decode_normalised(z_t).cpu().numpy()
    dense_grad=(l_clf.coef_[0]/l_scaler.scale_)*sigma
    dense_unit=dense_grad/max(np.linalg.norm(dense_grad),1e-12)
    projection=h_norm@dense_unit
    dense_unscaled_delta=-projection*np.dot(dense_grad,dense_unit)
    atom_mean=float(effects["target_ig_lbbb"].mean())
    dense_mean=float(dense_unscaled_delta.mean())
    dense_scale=atom_mean/dense_mean if abs(dense_mean)>1e-12 else 0.0
    dense_edited_norm=h_norm-dense_scale*projection[:,None]*dense_unit[None,:]
    with torch.no_grad():
        dense_raw=sae.denormalise(torch.from_numpy(dense_edited_norm.astype(np.float32)).to(a.device)).cpu().numpy()
    dense_matched=logits_from_raw(dense_raw)-base
    rng=np.random.default_rng(a.seed)
    summary={}
    for name,vals in {**effects,"dense_matched_attribution":dense_matched}.items():
        summary[name]={"mean_delta_logit":float(vals.mean()),"median_delta_logit":float(np.median(vals)),
                       "patient_bootstrap_95ci":patient_ci(vals,patients[test_pos],a.bootstrap,rng)}
    target_drop=summary["target_ig_lbbb"]["mean_delta_logit"]
    controls=max(abs(summary["matched_random"]["mean_delta_logit"]),abs(summary["orthogonal_af"]["mean_delta_logit"]))
    result={
      "protocol":"architecture- and metric-matched replication; original paper SAE checkpoint unavailable",
      "label_definition":"CLBBB score > 0 in PTB-XL+ scp_codes", "n_test_positive_ecgs":len(test_pos),
      "n_test_positive_patients":int(len(np.unique(patients[test_pos]))),
      "sae":{"architecture":"BatchTopK","N":sae.N,"k":sae.k,"checkpoint_step":int(ck["step"])},
      "lbbb_predictor":l_info,"af_predictor":af_info,
      "atoms":{"target_ig_lbbb":target,"matched_random":random_atom,"orthogonal_af":other,
               "target_random_frequency_ratio":float(freq[random_atom]/max(freq[target],1e-12)),
               "target_random_magnitude_ratio":float(mag[random_atom]/max(mag[target],1e-12)),
               "target_af_decoder_abs_cosine":float(cosine[other]),
               "af_atom_train_mean_lbbb_ig":float(attr_l[other]),
               "af_atom_train_mean_af_ig":float(attr_af[other]),
               "af_control_rule":"max AF IG among atoms with decoder |cosine|<0.05 and |LBBB IG| in bottom 10%"},
      "effects":summary,
      "dense_control":{"direction":"frozen LBBB predictor gradient in SAE-normalised dense space",
                       "attribution_matching_scale":float(dense_scale)},
      "paper_scale_reference_delta":-0.62,
      "replication_pass":bool(target_drop < -0.1 and controls < 0.01),
      "pass_rule":"target mean delta < -0.1 and both control absolute mean deltas < 0.01",
      "interpretation_guard":"Predictor-space intervention; not ECG-waveform causality. Dense matched control quantifies selection-readout circularity.",
    }
    a.out.parent.mkdir(parents=True,exist_ok=True); tmp=a.out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result,indent=2)+"\n"); tmp.replace(a.out)
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
