#!/usr/bin/env python
from __future__ import annotations

import ast, csv, json, os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SPLIT=ROOT/"results/manifest/split.csv"
LABELS=Path("/rhf/allocations/wq8/yd68/data/1.0.1/labels/ptbxl_statements.csv")
OUT=ROOT/"results/sae_reconciliation/phenotype_steering/manifest.csv"

TARGETS=("lbbb","rbbb","pvc","avb1","lafb","afib")


def main():
    split={r["ecg_id"]:r for r in csv.DictReader(SPLIT.open())}
    by_id={}
    for row in csv.DictReader(LABELS.open()):
        codes={k:float(v) for k,v in ast.literal_eval(row["scp_codes"])}
        def present(code): return codes.get(code,0.0)>0
        by_id[row["ecg_id"]]={
            # Incomplete bundle-branch labels are excluded from the negative
            # class of their corresponding complete-block task.
            "lbbb":"1" if present("CLBBB") else ("" if present("ILBBB") else "0"),
            "rbbb":"1" if present("CRBBB") else ("" if present("IRBBB") else "0"),
            "pvc":str(int(present("PVC"))),
            "avb1":str(int(present("1AVB"))),
            "lafb":str(int(present("LAFB"))),
            "afib":str(int(present("AFIB"))),
        }
    rows=[]
    for ecg_id,s in sorted(split.items(),key=lambda kv:int(kv[0])):
        if ecg_id not in by_id: continue
        rows.append({"row_index":len(rows),"ecg_id":ecg_id,"patient_id":s["patient_id"],
                     "split":s["split"],**by_id[ecg_id]})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    tmp=OUT.with_suffix(f".csv.tmp.{os.getpid()}")
    with tmp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tmp.replace(OUT)
    counts={}
    for target in TARGETS:
        counts[target]={}
        for sp in ("train","val","test"):
            valid=[r for r in rows if r["split"]==sp and r[target]!=""]
            pos=[r for r in valid if r[target]=="1"]
            counts[target][sp]={"valid_records":len(valid),"positive_records":len(pos),
                                "positive_patients":len({r["patient_id"] for r in pos})}
    meta={"targets":list(TARGETS),"label_source":str(LABELS),
          "exclusions":{"lbbb_negative":"exclude ILBBB","rbbb_negative":"exclude IRBBB"},
          "counts":counts}
    OUT.with_suffix(".json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps(meta,indent=2))


if __name__=="__main__": main()
