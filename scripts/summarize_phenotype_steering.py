#!/usr/bin/env python
from __future__ import annotations

import argparse,csv,json,os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/sae_reconciliation/phenotype_steering"
TARGETS=("lbbb","rbbb","pvc","avb1","lafb")
SEEDS=(4311,4312,4313)


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--base",type=Path,default=BASE)
    p.add_argument("--bootstrap",type=int,default=2000); p.add_argument("--jobs",type=int,default=15)
    return p.parse_args()


def bh(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); q=np.empty(n); running=1.
    for rank in range(n-1,-1,-1):
        idx=order[rank]; running=min(running,p[idx]*n/(rank+1)); q[idx]=running
    return q


def cell_summary(task_dir,target,seed,n_boot):
    from sklearn.metrics import roc_auc_score
    meta=json.loads((task_dir/"result.json").read_text()); d=np.load(task_dir/"records.npz",allow_pickle=True)
    patients=d["patient_ids"].astype(str); y=d["target_labels"].astype(float); valid=np.isfinite(y)
    base=d["baseline_target_logits"].astype(float); td=d["target_delta_logits"].astype(float)
    rd=d["random_delta_logits"].astype(float); od=d["other_delta_logits"].astype(float)
    pos=valid&(y==1); rng=np.random.default_rng(seed+sum(map(ord,target)))

    def groups(mask):
        unique=np.unique(patients[mask]); return unique,[np.where(mask&(patients==p))[0] for p in unique]
    up,gp=groups(pos); uv,gv=groups(valid)
    effect=[]; excess_a=[]; excess_b=[]
    for _ in range(n_boot):
        ip=np.concatenate([gp[i] for i in rng.integers(0,len(gp),len(gp))])
        effect.append(float(td[ip].mean()))
        excess_a.append(float(-(td[ip].mean()-rd[ip].mean(axis=1).mean())))
        iv=np.concatenate([gv[i] for i in rng.integers(0,len(gv),len(gv))])
        yy=y[iv].astype(int)
        if len(np.unique(yy))<2: continue
        target_drop=roc_auc_score(yy,base[iv])-roc_auc_score(yy,base[iv]+td[iv])
        random_drop=roc_auc_score(yy,base[iv])-roc_auc_score(yy,base[iv]+rd[iv].mean(axis=1))
        excess_b.append(float(target_drop-random_drop))
    effect=np.asarray(effect); excess_a=np.asarray(excess_a); excess_b=np.asarray(excess_b)
    dose=[meta["interventions"][f"top1_alpha_{v}"]["target_positive_mean_delta_logit"] for v in ("0","0.25","0.5","0.75","1")]
    drops=-np.asarray(dose); monotonic=bool(np.all(np.diff(drops)>=-1e-6))
    random_means=rd[pos].mean(axis=0); target_mean=float(td[pos].mean()); other_mean=float(od[pos].mean())
    target_drop=float(meta["interventions"]["top1_alpha_1"]["task_metrics"][target]["auroc_drop"])
    off=[abs(v["auroc_drop"]) for t,v in meta["interventions"]["top1_alpha_1"]["task_metrics"].items() if t!=target]
    a_ci=np.quantile(excess_a,[.025,.975]); b_ci=np.quantile(excess_b,[.025,.975]); e_ci=np.quantile(effect,[.025,.975])
    tier_a=bool(a_ci[0]>0 and abs(other_mean)<.01 and monotonic and -target_mean>np.quantile(-random_means,.95))
    return {"target":target,"seed":seed,"atom":meta["atoms"]["ig_target"],
            "sae_ev":meta.get("sae_metrics",{}).get("explained_variance"),
            "sae_dead_fraction":meta.get("sae_metrics",{}).get("dead_fraction"),
            "target_mean_delta_logit":target_mean,"target_effect_ci_low":float(e_ci[0]),"target_effect_ci_high":float(e_ci[1]),
            "random_mean_delta_logit":float(random_means.mean()),"other_mean_delta_logit":other_mean,
            "tier_a_excess_mean":float(excess_a.mean()),"tier_a_excess_ci_low":float(a_ci[0]),"tier_a_excess_ci_high":float(a_ci[1]),
            "dose_monotonic":monotonic,"tier_a_pass":tier_a,
            "target_auroc_drop":target_drop,"mean_abs_offtarget_auroc_change":float(np.mean(off)),
            "wbi":float(np.mean(off)/(abs(target_drop)+1e-8)),
            "tier_b_excess_mean":float(excess_b.mean()),"tier_b_excess_ci_low":float(b_ci[0]),"tier_b_excess_ci_high":float(b_ci[1]),
            "tier_b_p_one_sided":float((np.sum(excess_b<=0)+1)/(len(excess_b)+1)),
            "tier_b_pre_fdr":bool(b_ci[0]>0),"bootstrap_valid":len(excess_b)}


def main():
    a=parse_args(); from joblib import Parallel,delayed
    specs=[(a.base/"tasks"/f"seed{s}"/t,t,s) for s in SEEDS for t in TARGETS]
    missing=[str(p) for p,_,_ in specs if not (p/"result.json").exists() or not (p/"records.npz").exists()]
    if missing: raise RuntimeError("missing steering tasks: "+", ".join(missing))
    rows=Parallel(n_jobs=min(a.jobs,len(specs)))(delayed(cell_summary)(p,t,s,a.bootstrap) for p,t,s in specs)
    q=bh([r["tier_b_p_one_sided"] for r in rows])
    for r,qq in zip(rows,q): r["tier_b_q"]=float(qq); r["tier_b_pass"]=bool(r["tier_b_pre_fdr"] and qq<.05)
    out=a.base/"summary"; out.mkdir(parents=True,exist_ok=True)
    csv_path=out/"phenotype_steering_cells.csv"; tmp=csv_path.with_suffix(f".csv.tmp.{os.getpid()}")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(csv_path)
    aggregate=[]
    for t in TARGETS:
        rr=[r for r in rows if r["target"]==t]
        aggregate.append({"target":t,"seeds":len(rr),"tier_a_pass_seeds":sum(r["tier_a_pass"] for r in rr),
                          "tier_b_pass_seeds":sum(r["tier_b_pass"] for r in rr),
                          "median_target_delta_logit":float(np.median([r["target_mean_delta_logit"] for r in rr])),
                          "median_target_auroc_drop":float(np.median([r["target_auroc_drop"] for r in rr])),
                          "median_wbi":float(np.median([r["wbi"] for r in rr]))})
    agg=out/"phenotype_steering_by_target.csv"; tmp=agg.with_suffix(f".csv.tmp.{os.getpid()}")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0])); w.writeheader(); w.writerows(aggregate)
    tmp.replace(agg)
    lines=["# Discrete ECG Phenotype SAE Steering Audit","",
           "Primary unit: target x SAE seed. Tier A is frozen-readout-selective steering; Tier B requires a held-out AUROC effect above matched random with patient bootstrap and BH-FDR.","",
           "| Target | Tier A seeds | Tier B seeds | Median delta logit | Median AUROC drop | Median WBI |","|---|---:|---:|---:|---:|---:|"]
    for r in aggregate:
        lines.append(f"| {r['target']} | {r['tier_a_pass_seeds']}/3 | {r['tier_b_pass_seeds']}/3 | {r['median_target_delta_logit']:.4f} | {r['median_target_auroc_drop']:.6f} | {r['median_wbi']:.3f} |")
    lines += ["","## Interpretation discipline","","- Tier A without Tier B means readout calibration steering, not changed diagnostic ranking.",
              "- Tier B is still representation-level causality; no waveform-level causal claim is made.",
              "- Atom identities are seed-specific and are not aligned across dictionaries.",""]
    (out/"phenotype_steering_report.md").write_text("\n".join(lines))
    print(json.dumps({"cells":len(rows),"tier_a_pass":sum(r["tier_a_pass"] for r in rows),
                      "tier_b_pass":sum(r["tier_b_pass"] for r in rows),"aggregate":aggregate},indent=2))


if __name__=="__main__": main()
