#!/usr/bin/env python
"""Build the authoritative 6-model x 5-cohort execution matrix."""
from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/external_benchmark_v1"
MODELS={
 "CSFM":"csfm_cu118_commons","CARDIAC-FM":"cardiac_fm_cu118_commons","ECG-FM":"ecg_fm_cu118_commons",
 "ECG-JEPA":"ecg_jepa_cu118_commons","HuBERT-ECG":"hubert_ecg_cu118_commons","ST-MEM":"st_mem_cu118_commons"}
COHORTS={"ptbxl":21837,"chapman_f":10208,"cpsc_f":6876,"ningbo_f":34631,"mimic_f":100000}
TASKS_PER_COHORT={"chapman_f":4,"cpsc_f":2,"ningbo_f":4,"mimic_f":5}
LAYER_SAMPLE_RECORDS=4096


def markdown(frame: pd.DataFrame) -> str:
    columns=list(frame.columns)
    lines=["| "+" | ".join(columns)+" |","|"+"|".join(["---"]*len(columns))+"|"]
    for values in frame.itertuples(index=False,name=None):
        lines.append("| "+" | ".join(str(value) for value in values)+" |")
    return "\n".join(lines)


def index_records(suffix:str,cohort:str,kind:str) -> int:
    path=ROOT/f"results/activations_external_full_v1/{kind}/{suffix}/{cohort}/index_report.json"
    if not path.exists(): return 0
    return int(json.loads(path.read_text()).get("n_records",0))


def main() -> None:
    fidelity=pd.read_csv(ROOT/"results/sae_reconciliation/matched_scale_v1/matched_scale_model_profile.csv").set_index("model")
    transport=pd.read_csv(ROOT/"results/multicohort/pooled_sae_transport/pooled_transport_model_cohort_gate.csv")
    rows=[]
    for model,suffix in MODELS.items():
        for cohort,expected in COHORTS.items():
            if cohort=="ptbxl":
                pooled=expected; layers=expected; head=True
                layer_probe=True; strict_encoded=None
                frozen_count=local_count=adapted_count=skipped_count=0
                adapted_checkpoints=0
                pooled_leace_complete=True; pooled_leace_causal=None; pooled_leace_low_coupling=None
                closure_complete=True; closure_stable_ratios=None
            else:
                pooled=index_records(suffix,cohort,"pooled"); layers=index_records(suffix,cohort,"layer_atlas")
                head=(OUT/suffix/cohort/"frozen_heads.joblib").exists()
                probe_report=OUT/"layer_probe"/suffix/cohort/"probe_report.json"
                layer_probe=probe_report.exists()
                strict_encoded=int(json.loads(probe_report.read_text()).get("strict_encoded_count",0)) if layer_probe else None
                steering=OUT/suffix/cohort/"steering"
                frozen_count=len(list(steering.glob("frozen_atom/seed*/*/result.json")))
                local_count=len(list(steering.glob("local_atom/seed*/*/result.json")))
                adapted_count=len(list(steering.glob("cohort_adapted_atom/seed*/*/result.json")))
                skipped_count=len(list(steering.glob("skipped/seed*/*.json")))
                adapted_checkpoints=len(list((OUT/suffix/cohort/"cohort_adapted_sae").glob("seed*/*.metrics.json")))
                leace_summary=OUT/suffix/cohort/"pooled_leace/summary/pooled_leace_summary.json"
                pooled_leace_complete=leace_summary.exists()
                if pooled_leace_complete:
                    leace=json.loads(leace_summary.read_text())
                    pooled_leace_causal=int(leace.get("representation_causal",0))
                    pooled_leace_low_coupling=int(leace.get("low_coupling_causal",0))
                else:
                    pooled_leace_causal=pooled_leace_low_coupling=None
                closure_summary=OUT/suffix/cohort/"closure/closure_summary.json"
                closure_complete=closure_summary.exists()
                closure_stable_ratios=int(json.loads(closure_summary.read_text()).get("stable_ratios",0)) if closure_complete else None
            t=transport[(transport.model==model)&(transport.cohort==cohort)]
            transport_pass=bool(t.iloc[0].primary_transport_eligible) if len(t) else None
            if cohort=="ptbxl":
                status="complete"
            else:
                expected_cells=TASKS_PER_COHORT[cohort]*3
                complete=(
                    pooled>=expected and head and pooled_leace_complete and closure_complete
                    and layers>=min(LAYER_SAMPLE_RECORDS,expected) and layer_probe
                    and frozen_count+skipped_count>=expected_cells
                    and local_count+skipped_count>=expected_cells
                    and adapted_checkpoints>=3
                    and adapted_count+skipped_count>=expected_cells
                )
                status=("complete" if complete else "adapted_steering" if adapted_count else
                        "source_steering" if frozen_count or local_count else "adapted_sae" if adapted_checkpoints else
                        "layer_probe" if layer_probe else "causal_closure" if pooled_leace_complete and closure_complete else
                        "head_complete" if head else
                        "activations_complete" if pooled>=expected else "planned_or_running")
            rows.append({
              "model":model,"model_suffix":suffix,"cohort":cohort,"expected_records":expected,
              "pooled_records_indexed":pooled,"pooled_complete":pooled>=expected,"layer_records_indexed":layers,
              "layer_sample_target":min(LAYER_SAMPLE_RECORDS,expected),"layer_probe_complete":layer_probe,
              "strict_encoded_measurements":strict_encoded,
              "head_complete":head,"pooled_leace_complete":pooled_leace_complete,
              "pooled_leace_causal_cells":pooled_leace_causal,
              "pooled_leace_low_coupling_causal_cells":pooled_leace_low_coupling,
              "closure_complete":closure_complete,"closure_stable_ratios":closure_stable_ratios,
              "source_sae_fidelity":bool(fidelity.loc[model,"matched_scale_primary_eligible"]),
              "transport_smoke_pass":transport_pass,"frozen_atom_results":frozen_count,
              "source_local_atom_results":local_count,"cohort_adapted_sae_checkpoints":adapted_checkpoints,
              "cohort_adapted_atom_results":adapted_count,"task_gate_skips":skipped_count,
              "status":status,
            })
    frame=pd.DataFrame(rows); OUT.mkdir(parents=True,exist_ok=True); frame.to_csv(OUT/"execution_matrix_30pairs.csv",index=False)
    lines=["# 6-Model x 5-Cohort Execution Matrix","",markdown(frame),"",
           "`status` reports execution progress. Scientific eligibility remains controlled by separate fidelity, transport, and head-quality gates."]
    (OUT/"execution_matrix_30pairs.md").write_text("\n".join(lines)+"\n")
    print(frame.groupby(["cohort","status"]).size().to_string())


if __name__=="__main__": main()
