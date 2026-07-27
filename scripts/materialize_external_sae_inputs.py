#!/usr/bin/env python
"""Materialize one indexed external pair for cohort-adapted SAE training."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE=ROOT/"results/external_benchmark_v1"
ACT=ROOT/"results/activations_external_full_v1/pooled"


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--model-suffix",required=True); p.add_argument("--cohort",required=True); a=p.parse_args()
    from scripts.train_external_frozen_heads import load_activations
    pair=BASE/a.model_suffix/a.cohort; bundle=joblib.load(pair/"frozen_heads.joblib")
    acts,ids=load_activations(ACT/a.model_suffix/a.cohort)
    if not np.array_equal(ids.astype(str),np.asarray(bundle["record_ids"]).astype(str)): raise RuntimeError("record order mismatch")
    out=pair/"cohort_adapted_sae"; out.mkdir(parents=True,exist_ok=True)
    tmp=out/f"pooled_all.npy.tmp.{os.getpid()}"
    with tmp.open("wb") as handle: np.save(handle,acts.astype(np.float32))
    tmp.replace(out/"pooled_all.npy")
    manifest=pd.DataFrame({"record_id":ids.astype(str),"group_id":np.asarray(bundle.get("group_ids",ids)).astype(str),"split":bundle["split"]})
    manifest.to_csv(out/"sae_manifest.csv",index=False)
    finite=bool(np.isfinite(acts).all()); summary={"records":len(acts),"hidden_dim":acts.shape[1],"all_finite":finite,"split_unit":bundle["split_unit"]}
    (out/"materialization.json").write_text(__import__("json").dumps(summary,indent=2)+"\n")
    if not finite: raise RuntimeError("non-finite materialized activations")
    print(summary)


if __name__=="__main__": main()
