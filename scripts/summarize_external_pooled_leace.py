#!/usr/bin/env python
"""Patient/record paired-bootstrap summary for external pooled LEACE cells."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/external_benchmark_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-suffix", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260712)
    return p.parse_args()


def bh(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float); order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values)+1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(values)); out[order] = np.minimum(ranked, 1.0)
    return out


def auc_from_weights(weights: np.ndarray, y: np.ndarray, score: np.ndarray) -> np.ndarray:
    order = np.argsort(score, kind="mergesort"); ys = y[order]; scores = score[order]
    starts = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1]
    w = weights[:, order]
    pos = np.add.reduceat(w * ys[None, :], starts, axis=1)
    neg = np.add.reduceat(w * (1-ys)[None, :], starts, axis=1)
    neg_before = np.cumsum(neg, axis=1) - neg
    numerator = np.sum(pos * (neg_before + 0.5 * neg), axis=1)
    denominator = pos.sum(1) * neg.sum(1)
    return np.divide(numerator, denominator, out=np.full(len(weights), np.nan), where=denominator>0)


def bootstrap_adjusted(data: dict[str, np.ndarray], n_boot: int, seed: int) -> np.ndarray:
    group_ids = data["group_ids"].astype(str); y = data["y"].astype(int)
    unique, inverse = np.unique(group_ids, return_inverse=True); n_groups = len(unique)
    rng = np.random.default_rng(seed); chunks=[]
    for lo in range(0, n_boot, 100):
        size = min(100, n_boot-lo)
        group_weights = rng.multinomial(n_groups, np.full(n_groups,1/n_groups), size=size).astype(np.float64)
        weights = group_weights[:, inverse]
        erased_auc = auc_from_weights(weights, y, data["erased_score"])
        random_auc = np.column_stack([
            auc_from_weights(weights, y, data["random_scores"][:,j])
            for j in range(data["random_scores"].shape[1])
        ]).mean(1)
        chunks.append(random_auc-erased_auc)
    return np.concatenate(chunks)


def main() -> None:
    args=parse_args(); root=BASE/args.model_suffix/args.cohort/"pooled_leace"; rows=[]
    for result_path in sorted(root.glob("*__to__*/result.json")):
        result=json.loads(result_path.read_text()); records=result_path.with_name("records.npz")
        if not records.exists(): continue
        with np.load(records,allow_pickle=False) as loaded: data={key:loaded[key] for key in loaded.files}
        values=bootstrap_adjusted(data,args.bootstrap,args.seed+sum(map(ord,result["concept"]+result["task"])))
        low,high=np.nanquantile(values,[.025,.975]); p=(1+float(np.sum(values<=0)))/(len(values)+1)
        rows.append({**result,"adjusted_delta_ci_low":low,"adjusted_delta_ci_high":high,
                     "adjusted_delta_p_one_sided":p,"bootstrap_samples":args.bootstrap})
    if not rows: raise RuntimeError(f"No pooled LEACE results in {root}")
    frame=pd.DataFrame(rows); frame["adjusted_delta_q"]=bh(frame.adjusted_delta_p_one_sided.to_numpy())
    frame["head_quality_pass"]=frame.base_auroc.ge(.70)
    frame["representation_causal"]=(
        frame.pooled_strict_encoded & frame.eraser_effective & frame.head_quality_pass
        & frame.adjusted_delta_ci_low.gt(0) & frame.adjusted_delta_q.lt(.05)
    )
    out=root/"summary"; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"pooled_leace_cells.csv",index=False)
    report={"model_suffix":args.model_suffix,"cohort":args.cohort,"cells":len(frame),
            "strict_encoded":int(frame.pooled_strict_encoded.sum()),
            "eraser_effective":int(frame.eraser_effective.sum()),
            "representation_causal":int(frame.representation_causal.sum()),
            "low_coupling_causal":int((frame.representation_causal & frame.coupling_role.eq("low_coupling_candidate")).sum()),
            "split_unit":str(frame.split_unit.iloc[0]),"bootstrap_samples":args.bootstrap}
    (out/"pooled_leace_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# External Pooled-Representation LEACE Audit","",
           "This audit intervenes at the pooled head input. It does not claim internal-layer continuation causality.","",
           f"- Cells: {report['cells']}",f"- Strict encoded: {report['strict_encoded']}",
           f"- Effective erasure: {report['eraser_effective']}",
           f"- Representation-causal: {report['representation_causal']}",
           f"- Low-coupling causal: {report['low_coupling_causal']}",
           f"- Bootstrap unit: {report['split_unit']}",""]
    (out/"pooled_leace_report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(report))


if __name__=="__main__": main()
