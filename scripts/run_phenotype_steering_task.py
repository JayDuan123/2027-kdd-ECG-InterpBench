#!/usr/bin/env python
from __future__ import annotations

import argparse,csv,json,os,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE=ROOT/"results/sae_reconciliation/phenotype_steering"
TARGETS=("lbbb","rbbb","pvc","avb1","lafb")
ALL_TARGETS=TARGETS+("afib",)
OTHER={"lbbb":"afib","rbbb":"pvc","pvc":"avb1","avb1":"pvc","lafb":"afib"}


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--target",choices=TARGETS,required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--acts",type=Path,default=ROOT/"results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    p.add_argument("--manifest",type=Path,default=BASE/"manifest.csv")
    p.add_argument("--heads",type=Path,default=BASE/"frozen_heads.joblib")
    p.add_argument("--out-dir",type=Path,default=BASE/"tasks")
    p.add_argument("--device",default="cuda")
    p.add_argument("--n-random",type=int,default=20)
    return p.parse_args()


def encode_batches(sae,x,batch,device):
    import torch
    out=[]; sae.eval()
    with torch.no_grad():
        for lo in range(0,len(x),batch):
            out.append(sae.encode(torch.from_numpy(np.asarray(x[lo:lo+batch],dtype=np.float32)).to(device)).cpu().numpy())
    return np.concatenate(out)


def sigmoid(x):
    x=np.clip(x,-50,50); return 1/(1+np.exp(-x))


def main():
    a=parse_args(); import joblib,torch
    from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE
    out=a.out_dir/f"seed{a.seed}"/a.target; out.mkdir(parents=True,exist_ok=True)
    final=out/"result.json"
    if final.exists() and (out/"records.npz").exists():
        print(f"already complete: {out}"); return

    rows=list(csv.DictReader(a.manifest.open())); split=np.array([r["split"] for r in rows])
    patients=np.array([r["patient_id"] for r in rows]); x=np.asarray(np.load(a.acts,mmap_mode="r"),dtype=np.float32)
    head_bundle=joblib.load(a.heads); scaler=head_bundle["scaler"]; heads=head_bundle["heads"]
    labels={t:np.asarray(heads[t]["labels"],dtype=float) for t in ALL_TARGETS}
    ck=torch.load(a.checkpoint,map_location=a.device); cfg=ck["config"]
    sae=BatchTopKSAE(x.shape[1],int(cfg["n_features"]),int(cfg["k"])).to(a.device)
    sae.load_state_dict(ck["model"]); sae.eval()
    tr=np.where(split=="train")[0]; te=np.where(split=="test")[0]
    ztr=encode_batches(sae,x[tr],256,a.device); zte=encode_batches(sae,x[te],256,a.device)
    dec=sae.W_dec.detach().cpu().numpy(); sigma=sae.sigma.detach().cpu().numpy()
    gradients={t:((heads[t]["clf"].coef_[0]/scaler.scale_)*sigma)@dec for t in ALL_TARGETS}
    ytr=labels[a.target][tr]; valid_tr=np.isfinite(ytr); pos=np.where(valid_tr&(ytr==1))[0]; neg=np.where(valid_tr&(ytr==0))[0]
    ig=(ztr[pos]*gradients[a.target]).mean(0); rank=np.argsort(ig)[::-1]; target_atom=int(rank[0])
    cav_atom=int(np.argmax(gradients[a.target]))
    assoc=(ztr[pos].mean(0)-ztr[neg].mean(0))/np.maximum(ztr[valid_tr].std(0),1e-8)
    activation_atom=int(np.argmax(assoc))

    other_name=OTHER[a.target]; yother=labels[other_name][tr]; opos=np.where(np.isfinite(yother)&(yother==1))[0]
    other_ig=(ztr[opos]*gradients[other_name]).mean(0)
    dirs=dec/np.maximum(np.linalg.norm(dec,axis=0,keepdims=True),1e-12)
    cosine=np.abs(dirs[:,target_atom]@dirs)
    candidates=np.where((cosine<.05)&(np.abs(ig)<=np.quantile(np.abs(ig),.10)))[0]
    if not len(candidates): raise RuntimeError("no orthogonal other-phenotype atom")
    other_atom=int(candidates[np.argmax(other_ig[candidates])])

    rng=np.random.default_rng(a.seed+sum(map(ord,a.target)))
    pseudo=np.zeros(valid_tr.sum(),dtype=int); pseudo[:len(pos)]=1; rng.shuffle(pseudo)
    valid_idx=np.where(valid_tr)[0]; pseudo_pos=valid_idx[pseudo==1]
    shuffled_ig=(ztr[pseudo_pos]*gradients[a.target]).mean(0); shuffled_ig[rank[:10]]=-np.inf
    shuffled_atom=int(np.argmax(shuffled_ig))

    freq=(ztr>0).mean(0); mag=np.divide(ztr.sum(0),(ztr>0).sum(0),out=np.zeros(sae.N),where=(ztr>0).sum(0)>0)
    dist=np.abs(np.log((freq+1e-6)/(freq[target_atom]+1e-6)))+np.abs(np.log((mag+1e-6)/(mag[target_atom]+1e-6)))
    dist[np.unique(np.r_[rank[:10],other_atom,shuffled_atom])]=np.inf
    random_atoms=np.argsort(dist)[:a.n_random].astype(int)

    # Decode once. Every intervention below is then propagated analytically
    # through the linear SAE decoder and frozen linear heads; this is exactly
    # equivalent to materialising each decoded representation.
    with torch.no_grad():
        recon=sae.decode(torch.from_numpy(zte).to(a.device)).cpu().numpy()
    baseline_logits={t:heads[t]["clf"].decision_function(scaler.transform(recon)) for t in ALL_TARGETS}

    def delta_for(indices,new_values):
        idx=np.atleast_1d(indices).astype(int)
        if np.isscalar(new_values): new=np.full((len(zte),len(idx)),float(new_values),dtype=np.float32)
        else:
            vals=np.asarray(new_values,dtype=np.float32)
            new=vals[:,None] if vals.ndim==1 and len(idx)==1 else vals
        dz=new-zte[:,idx]
        return {t:dz@gradients[t][idx] for t in ALL_TARGETS}

    def task_metrics(delta):
        result={}
        for t in ALL_TARGETS:
            y=labels[t][te]; valid=np.isfinite(y); yy=y[valid].astype(int)
            base=baseline_logits[t][valid]; edit=base+delta[t][valid]
            result[t]={"baseline_auroc":float(roc_auc_score(yy,base)),
                       "edited_auroc":float(roc_auc_score(yy,edit)),
                       "auroc_drop":float(roc_auc_score(yy,base)-roc_auc_score(yy,edit)),
                       "baseline_auprc":float(average_precision_score(yy,sigmoid(base))),
                       "edited_auprc":float(average_precision_score(yy,sigmoid(edit))),
                       "brier_change":float(brier_score_loss(yy,sigmoid(edit))-brier_score_loss(yy,sigmoid(base)))}
        return result

    target_test=labels[a.target][te]; target_pos=np.isfinite(target_test)&(target_test==1)
    def summarize_edit(name,indices,new_values):
        delta=delta_for(indices,new_values); td=delta[a.target][target_pos]
        return {"name":name,"atoms":[int(i) for i in np.atleast_1d(indices)],
                "target_positive_mean_delta_logit":float(td.mean()),
                "target_positive_median_delta_logit":float(np.median(td)),
                "task_metrics":task_metrics(delta)},delta

    interventions={}; saved_delta={}
    for alpha in (0.,.25,.5,.75,1.):
        values=zte[:,target_atom]*(1-alpha)
        obj,d=summarize_edit(f"top1_suppression_alpha_{alpha:g}",[target_atom],values)
        interventions[f"top1_alpha_{alpha:g}"]=obj
        if alpha==1.: saved_delta["target"]=d
    for n in (5,10):
        obj,_=summarize_edit(f"IG_top{n}_zero",rank[:n],0.); interventions[f"top{n}_zero"]=obj
    centroid=ztr.mean(0); obj,_=summarize_edit("top1_population_centroid",[target_atom],np.full(len(zte),centroid[target_atom])); interventions["top1_centroid"]=obj
    active=ztr[:,target_atom][ztr[:,target_atom]>0]; active_std=float(active.std()) if len(active)>1 else 0.
    low,high=np.quantile(ztr[:,target_atom],[.01,.99])
    for beta in (.5,1.,2.):
        values=np.clip(zte[:,target_atom]+beta*active_std,low,high)
        obj,_=summarize_edit(f"top1_enhance_beta_{beta:g}",[target_atom],values); interventions[f"enhance_beta_{beta:g}"]=obj
    for label,atom in (("cav_zero",cav_atom),("activation_zero",activation_atom),("other_zero",other_atom),("shuffled_zero",shuffled_atom)):
        obj,d=summarize_edit(label,[atom],0.); interventions[label]=obj
        if label=="other_zero": saved_delta["other"]=d
        if label=="shuffled_zero": saved_delta["shuffled"]=d

    random_results=[]; random_target_deltas=[]
    for atom in random_atoms:
        obj,d=summarize_edit("matched_random_zero",[int(atom)],0.)
        random_results.append({"atom":int(atom),"target_positive_mean_delta_logit":obj["target_positive_mean_delta_logit"],
                               "target_metrics":obj["task_metrics"][a.target]})
        random_target_deltas.append(d[a.target])

    # Matched-attribution dense direction circularity control.
    hnorm=zte@dec.T+sae.b_dec.detach().cpu().numpy(); dense_grad=(heads[a.target]["clf"].coef_[0]/scaler.scale_)*sigma
    unit=dense_grad/max(np.linalg.norm(dense_grad),1e-12); projection=hnorm@unit
    unscaled=-projection*np.dot(dense_grad,unit); target_mean=float(saved_delta["target"][a.target][target_pos].mean())
    scale=target_mean/float(unscaled[target_pos].mean()) if abs(unscaled[target_pos].mean())>1e-12 else 0.
    dense_delta=scale*unscaled

    metric_path=a.checkpoint.with_suffix(".metrics.json")
    sae_metrics=json.loads(metric_path.read_text()) if metric_path.exists() else {}
    result={"target":a.target,"seed":a.seed,"checkpoint":str(a.checkpoint),"sae_metrics":sae_metrics,
            "atoms":{"ig_target":target_atom,"ig_top5":[int(i) for i in rank[:5]],"ig_top10":[int(i) for i in rank[:10]],
                     "cav":cav_atom,"activation":activation_atom,"other_target":other_name,"other":other_atom,
                     "other_decoder_abs_cosine":float(cosine[other_atom]),"other_train_target_ig":float(ig[other_atom]),
                     "shuffled":shuffled_atom,"matched_random":[int(i) for i in random_atoms]},
            "interventions":interventions,"random_controls":random_results,
            "dense_control":{"mean_target_positive_delta_logit":float(dense_delta[target_pos].mean()),"matching_scale":float(scale)},
            "guards":{"selection":"train only","evaluation":"held-out test","analytic_linear_continuation":True,
                      "n_test_positive_records":int(target_pos.sum()),"n_test_positive_patients":len(set(patients[te][target_pos]))}}
    tmp=final.with_suffix(f".json.tmp.{os.getpid()}"); tmp.write_text(json.dumps(result,indent=2)+"\n"); tmp.replace(final)
    npz_tmp=out/f"records.npz.tmp.{os.getpid()}"
    with npz_tmp.open("wb") as f:
        np.savez_compressed(f,patient_ids=patients[te],target_labels=target_test,
                            baseline_target_logits=baseline_logits[a.target],
                            target_delta_logits=saved_delta["target"][a.target],
                            random_delta_logits=np.stack(random_target_deltas,axis=1),
                            other_delta_logits=saved_delta["other"][a.target],
                            shuffled_delta_logits=saved_delta["shuffled"][a.target])
    npz_tmp.replace(out/"records.npz")
    print(json.dumps({"target":a.target,"seed":a.seed,"atom":target_atom,
                      "target_delta":interventions["top1_alpha_1"]["target_positive_mean_delta_logit"]}))


if __name__=="__main__": main()
