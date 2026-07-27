#!/usr/bin/env python
"""Build a deterministic 100k ICD-linked patient-splittable MIMIC extraction plan."""
from __future__ import annotations

import csv
import hashlib
import heapq
import math
from pathlib import Path
import sys

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SOURCE=ROOT/"results/multicohort/track_f_full/waveform_concepts_by_record.csv"
LABELS=ROOT/"results/multicohort/mimic_icd_label_matrix.csv"
OUT=ROOT/"results/activations_external_full_v1"
PLAN=OUT/"plan_mimic_100k"
MODELS=("csfm","cardiac_fm","ecg_fm","ecg_jepa","hubert_ecg","st_mem")
N=100_000; POOLED_BATCH=128; LAYER_N=4096; LAYER_BATCH=16


def rank(key: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8],"big")


def main() -> None:
    from scripts.plan_external_activation_extraction import command,index_command
    label_ids=set(pd.read_csv(LABELS,usecols=["study_id"]).study_id.astype(str))
    heap=[]; counter=0; fieldnames=[]
    with SOURCE.open(newline="") as handle:
        reader=csv.DictReader(handle); fieldnames=list(reader.fieldnames or [])
        for row in reader:
            if row.get("cohort")!="mimic" or row.get("status")!="ok" or row.get("study_id_or_record_key") not in label_ids: continue
            value=rank(row["study_id_or_record_key"],"mimic-main-v1"); item=(-value,counter,row); counter+=1
            if len(heap)<N: heapq.heappush(heap,item)
            elif value < -heap[0][0]: heapq.heapreplace(heap,item)
    rows=[item[2] for item in sorted(heap,key=lambda x:-x[0])]
    if len(rows)!=N: raise RuntimeError(f"Expected {N} linked records, got {len(rows)}")
    PLAN.mkdir(parents=True,exist_ok=True)
    full=PLAN/"mimic_main_manifest.csv"
    with full.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    layer_rows=sorted(rows,key=lambda r:rank(r["study_id_or_record_key"],"mimic-layer-v1"))[:LAYER_N]
    layer=PLAN/"mimic_layer_manifest.csv"
    with layer.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fieldnames); w.writeheader(); w.writerows(layer_rows)
    pooled=[]; layers=[]; pindex=[]; lindex=[]; smoke=[]
    for model in MODELS:
        first=None
        for offset in range(0,N,POOLED_BATCH):
            cmd=command(model,"mimic",offset,min(POOLED_BATCH,N-offset),"pooled",OUT/"pooled",full,"cuda")
            pooled.append(cmd)
            if first is None: first=cmd
        smoke.append(first)
        for offset in range(0,LAYER_N,LAYER_BATCH):
            layers.append(command(model,"mimic",offset,min(LAYER_BATCH,LAYER_N-offset),"all",OUT/"layer_atlas",layer,"cuda")+" --pool-layer-activations")
        pindex.append(index_command(model,"mimic",OUT/"pooled")); lindex.append(index_command(model,"mimic",OUT/"layer_atlas"))
    for name,values in (("pooled_commands.txt",pooled),("pooled_smoke_commands.txt",smoke),("layer_commands.txt",layers),
                        ("pooled_index_commands.txt",pindex),("layer_index_commands.txt",lindex)):
        (PLAN/name).write_text("\n".join(values)+"\n")
    pd.DataFrame([
        {"model":model,"cohort":"mimic","records":N,"pooled_batch":POOLED_BATCH,
         "pooled_shards":math.ceil(N/POOLED_BATCH),"layer_records":LAYER_N,
         "layer_batch":LAYER_BATCH,"layer_shards":math.ceil(LAYER_N/LAYER_BATCH)}
        for model in MODELS
    ]).to_csv(PLAN/"plan_summary.csv",index=False)
    (PLAN/"plan_report.md").write_text(
        "# MIMIC External Main Plan\n\n"
        f"- ICD-linked deterministic records: {N}\n- Full pooled commands: {len(pooled)}\n"
        f"- Layer-atlas records: {LAYER_N}\n- Layer commands: {len(layers)}\n"
        "- Split is assigned later by subject_id, never by study_id.\n")
    print(f"eligible_seen={counter} selected={len(rows)} pooled={len(pooled)} layers={len(layers)}")


if __name__=="__main__": main()
