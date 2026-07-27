#!/usr/bin/env python
"""Record-bootstrap/FDR summary for external frozen/local atom steering."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/external_benchmark_v1"
spec=importlib.util.spec_from_file_location("single",ROOT/"scripts/summarize_steering_benchmark.py")
single=importlib.util.module_from_spec(spec); spec.loader.exec_module(single)


def main() -> None:
    entries=[]
    for path in sorted(BASE.glob("*/*/steering/*/seed*/*/result.json")):
        result=json.loads(path.read_text()); npz=path.with_name("records.npz")
        if not npz.exists(): continue
        with np.load(npz,allow_pickle=False) as loaded: data={k:loaded[k] for k in loaded.files}
        entries.append((result,data))
    if not entries: raise RuntimeError("No external steering results")
    rows=[]
    for result,data in entries:
        wrong=[od["top5_delta"] for other,od in entries if other["model"]==result["model"] and other["cohort"]==result["cohort"]
               and other["protocol"]==result["protocol"] and int(other["seed"])==int(result["seed"]) and other["target"]!=result["target"]]
        point=single.one_stats(data,result,wrong)
        rng=np.random.default_rng(20260712+int(result["seed"])+sum(map(ord,result["model"]+result["cohort"]+result["target"]+result["protocol"])))
        samples=single.vectorized_bootstrap(data,result,wrong,2000,rng)
        row={
            "model":result["model"],"model_suffix":result["model_suffix"],"cohort":result["cohort"],
            "protocol":result["protocol"],"target":result["target"],"family":result["family"],"seed":result["seed"],
            "raw_head_test_auroc":result["raw_head_test_auroc"],"sae_recon_head_test_auroc":result["sae_recon_head_test_auroc"],
            "sae_readout_retention":result["sae_readout_retention"],"source_fidelity_eligible":result["source_fidelity_eligible"],
            "transport_eligible":result["transport_eligible"],
            "transport_gate_applicable":result.get("transport_gate_applicable", True),
            "dictionary_source":result.get("dictionary_source", "source"),
            "dictionary_training_recon_r2":result.get("dictionary_training_recon_r2"),
            "dictionary_training_dead_fraction":result.get("dictionary_training_dead_fraction"),
            "dictionary_quality_warning":result.get("dictionary_quality_warning", False),
            "split_unit":result["split_unit"],"bootstrap_samples":2000,**point,
        }
        for metric in ("tier1_excess_attribution","excess_selectivity","wbi_improvement","wrong_atom_margin","behavior_excess"):
            values=np.asarray(samples[metric]); row[f"{metric}_ci_low"],row[f"{metric}_ci_high"]=np.quantile(values,[.025,.975])
            row[f"{metric}_p_one_sided"]=(1+float((values<=0).sum()))/(len(values)+1)
        rows.append(row)
    cells=pd.DataFrame(rows)
    for metric in ("tier1_excess_attribution","excess_selectivity","wbi_improvement","wrong_atom_margin","behavior_excess"):
        cells[f"{metric}_q"]=np.nan
        for _,idx in cells.groupby(["model","cohort","protocol"]).groups.items():
            cells.loc[idx,f"{metric}_q"]=single.bh(cells.loc[idx,f"{metric}_p_one_sided"].to_numpy())
    cells["head_quality_pass"]=cells.raw_head_test_auroc.ge(.70)
    cells["reconstruction_retention_pass"]=cells.sae_readout_retention.ge(.95)
    cells["tier0_fidelity"]=(cells.source_fidelity_eligible & cells.transport_eligible & cells.head_quality_pass & cells.reconstruction_retention_pass)
    cells["tier1_sparse_attribution"]=(cells.tier0_fidelity & cells.tier1_excess_attribution_ci_low.gt(0) & cells.tier1_excess_attribution_q.lt(.05))
    cells["tier2_selective_steering"]=(cells.tier0_fidelity & cells.excess_selectivity_ci_low.gt(0) & cells.wbi_improvement_ci_low.gt(0)
        & cells.wrong_atom_margin_ci_low.gt(0) & cells.excess_selectivity_q.lt(.05) & cells.wbi_improvement_q.lt(.05) & cells.wrong_atom_margin_q.lt(.05))
    cells["tier3_behavior_changing"]=(cells.tier0_fidelity & cells.behavior_excess_ci_low.gt(0) & cells.behavior_excess_q.lt(.05))
    cells["tier4_waveform_causal"]=False
    out=BASE/"summary"; out.mkdir(parents=True,exist_ok=True); cells.to_csv(out/"external_steering_cells.csv",index=False)
    profile=cells.groupby(["model","cohort","protocol","target"],as_index=False).agg(
        seeds=("seed","nunique"),tier0_pass=("tier0_fidelity","sum"),tier1_pass=("tier1_sparse_attribution","sum"),
        tier2_pass=("tier2_selective_steering","sum"),tier3_pass=("tier3_behavior_changing","sum"),
        ste_mean=("ste","mean"),otd_mean=("otd_mean","mean"),wbi_median=("wbi","median"),
        raw_head_auroc=("raw_head_test_auroc","mean"),sae_retention=("sae_readout_retention","mean"))
    profile["robustness"]=np.select([profile.tier2_pass.eq(3),profile.tier2_pass.eq(2)],["robust_3_of_3","suggestive_2_of_3"],default="unstable_or_null")
    profile.to_csv(out/"external_steering_target_profile.csv",index=False)
    frozen=cells[cells.protocol.eq("frozen_atom")]; primary=frozen[frozen.tier0_fidelity]
    local=cells[cells.protocol.eq("local_atom")]
    adapted=cells[cells.protocol.eq("cohort_adapted_atom")]
    lines=["# External SAE Steering Audit","",
        "- All six models and all feasible native tasks are computed; gates determine interpretation, not execution.",
        "- Frozen-Atom is primary; Local-Atom re-ranks the fixed source SAE dictionary on each cohort and is a cohort-local ranking sensitivity.",
        "- Cohort-adapted SAE retraining is evaluated separately and must not be inferred from Local-Atom results.",
        "- Chapman/CPSC use record-level bootstrap because native patient identifiers are unavailable.",
        "- Tier 4 waveform causality is not claimed.","",
        f"- Completed seed-level cells: {len(cells)} ({len(frozen)} frozen, {len(local)} source-local, {len(adapted)} cohort-adapted).",
        f"- Frozen-Atom tier-0 eligible cells: {len(primary)}/{len(frozen)}.",
        f"- Eligible Tier 1/Tier 2/Tier 3: {int(primary.tier1_sparse_attribution.sum())}/{int(primary.tier2_selective_steering.sum())}/{int(primary.tier3_behavior_changing.sum())} of {len(primary)}.","",
        "## Target profile","",single.markdown_table(profile),"",
        "A gate-failing row is a fidelity/transport/readout limitation, not evidence of absent physiological information."]
    (out/"external_steering_report.md").write_text("\n".join(lines)+"\n")
    print(profile.to_string(index=False))


if __name__=="__main__": main()
