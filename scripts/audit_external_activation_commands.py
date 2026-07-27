#!/usr/bin/env python
"""Audit every planned external activation shard against its command contract."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(); p.add_argument("--commands",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--check-finite",action="store_true"); return p.parse_args()


def option(parts:list[str],name:str) -> str:
    i=parts.index(name); return parts[i+1]


def main() -> None:
    a=parse_args(); rows=[]
    for task_index,line in enumerate(a.commands.read_text().splitlines()):
        if not line.strip(): continue
        parts=shlex.split(line); out=Path(option(parts,"--out-dir")); model=option(parts,"--model")
        cohort=option(parts,"--cohort"); shard=option(parts,"--shard-name"); expected=int(option(parts,"--limit"))
        root=out/model/cohort/shard; meta=root/"activation_metadata.json"; pooled=root/"pooled.npy"; ids=root/"record_ids.csv"
        status="missing"; loaded=None; finite=None; shape=""; reason="activation_metadata missing"
        layer_files_expected=0; layer_files_valid=0; layer_shapes=""
        if meta.exists():
            try:
                payload=json.loads(meta.read_text()); loaded=int(payload["loaded_records"]); arr=np.load(pooled,mmap_mode="r")
                shape="x".join(map(str,arr.shape)); id_count=sum(1 for _ in csv.DictReader(ids.open()))
                finite=bool(np.isfinite(arr).all()) if a.check_finite else None
                layer_ids=[int(value) for value in payload.get("layers",[])]
                layer_files_expected=len(layer_ids); observed_shapes=[]; layer_errors=[]
                pooled_layers=payload.get("layer_aggregation")=="token_mean"
                for layer_id in layer_ids:
                    layer_path=root/f"layer_{layer_id:02d}.npy"
                    if not layer_path.exists():
                        layer_errors.append(f"missing {layer_path.name}"); continue
                    layer=np.load(layer_path,mmap_mode="r"); observed_shapes.append(f"{layer_id}:"+"x".join(map(str,layer.shape)))
                    layer_finite=bool(np.isfinite(layer).all()) if a.check_finite else True
                    valid=(len(layer)==loaded and layer_finite and (not pooled_layers or layer.ndim==2))
                    declared_shape=payload.get("layer_shapes",{}).get(str(layer_id))
                    if declared_shape is not None:
                        valid=valid and list(layer.shape)==list(declared_shape)
                    if valid: layer_files_valid+=1
                    else: layer_errors.append(f"invalid {layer_path.name} shape={layer.shape} finite={layer_finite}")
                layer_shapes="|".join(observed_shapes)
                layers_ok=layer_files_valid==layer_files_expected and not layer_errors
                ok=(len(arr)==loaded==id_count and loaded<=expected and (finite is not False) and layers_ok)
                status="complete" if ok else "invalid"
                reason="" if ok else (
                    f"shape/ID/finite/layer mismatch: rows={len(arr)} loaded={loaded} ids={id_count} "
                    f"finite={finite} layers={layer_files_valid}/{layer_files_expected}; " + "; ".join(layer_errors)
                )
            except Exception as exc:
                status="invalid"; reason=f"{type(exc).__name__}: {exc}"
        rows.append({"task_index":task_index,"model_suffix":model,"cohort":cohort,"shard":shard,
                     "expected":expected,"loaded":loaded,"shape":shape,"finite":finite,
                     "layer_files_expected":layer_files_expected,"layer_files_valid":layer_files_valid,
                     "layer_shapes":layer_shapes,"status":status,"reason":reason})
    frame=pd.DataFrame(rows); a.out.parent.mkdir(parents=True,exist_ok=True); frame.to_csv(a.out,index=False)
    counts=frame.status.value_counts().to_dict(); print({"planned":len(frame),**counts})
    if not frame.status.eq("complete").all(): raise SystemExit(2)


if __name__=="__main__": main()
