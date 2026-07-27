#!/usr/bin/env python
from __future__ import annotations

import csv,glob,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/sae_reconciliation/phenotype_steering"
TARGETS=("lbbb","rbbb","pvc","avb1","lafb")


def main():
    import matplotlib.pyplot as plt
    out=BASE/"summary"; out.mkdir(parents=True,exist_ok=True)
    rows=[]; atoms=[]
    for path in sorted(glob.glob(str(BASE/"tasks/seed*/*/result.json"))):
        d=json.load(open(path)); t=d["target"]; seed=int(d["seed"]); atoms.append({"seed":seed,"target":t,"atom":d["atoms"]["ig_target"]})
        for alpha in (0,.25,.5,.75,1):
            key=f"top1_alpha_{alpha:g}"; rows.append({"target":t,"seed":seed,"alpha":alpha,
                "logit_drop":-float(d["interventions"][key]["target_positive_mean_delta_logit"])})
    with (out/"dose_response.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with (out/"atom_registry.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(atoms[0])); w.writeheader(); w.writerows(atoms)
    cells=list(csv.DictReader((out/"phenotype_steering_cells.csv").open()))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    colors={"lbbb":"#2166ac","rbbb":"#67a9cf","pvc":"#b2182b","avb1":"#ef8a62","lafb":"#4d9221"}
    for t in TARGETS:
        med=[]; lo=[]; hi=[]
        for alpha in (0,.25,.5,.75,1):
            v=[r["logit_drop"] for r in rows if r["target"]==t and r["alpha"]==alpha]
            med.append(np.median(v)); lo.append(np.min(v)); hi.append(np.max(v))
        axes[0].plot((0,.25,.5,.75,1),med,marker="o",label=t.upper(),color=colors[t])
        axes[0].fill_between((0,.25,.5,.75,1),lo,hi,color=colors[t],alpha=.12)
    axes[0].set(xlabel="Suppression strength",ylabel="Target-positive logit drop",title="Top-1 atom dose response")
    axes[0].legend(frameon=False,ncol=2)
    x=np.arange(len(TARGETS)); width=.34
    tier_a=[sum(r["tier_a_pass"]=="True" for r in cells if r["target"]==t) for t in TARGETS]
    tier_b=[sum(r["tier_b_pass"]=="True" for r in cells if r["target"]==t) for t in TARGETS]
    axes[1].bar(x-width/2,tier_a,width,label="Tier A",color="#4c78a8")
    axes[1].bar(x+width/2,tier_b,width,label="Tier B",color="#f58518")
    axes[1].set_xticks(x,TARGETS); axes[1].set_ylim(0,3.4); axes[1].set_ylabel("Passing SAE seeds (of 3)")
    axes[1].set_title("Readout attribution vs behavior change"); axes[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig(out/"phenotype_steering_main.png",dpi=220); fig.savefig(out/"phenotype_steering_main.pdf")
    print(out/"phenotype_steering_main.png")


if __name__=="__main__": main()
