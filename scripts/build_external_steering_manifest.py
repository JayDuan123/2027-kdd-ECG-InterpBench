#!/usr/bin/env python
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
pairs=pd.read_csv(ROOT/"results/external_benchmark_v1/head_pair_manifest.csv")
tasks={"chapman_f":["af_rhythm_native","bbb_conduction_native","qt_interval_native","st_t_abnormal_native"],
       "cpsc_f":["af_rhythm_native","bbb_conduction_native"],
       "ningbo_f":["af_rhythm_native","bbb_conduction_native","qt_interval_native","st_t_abnormal_native"],
       "mimic_f":["af_rhythm_icd","bbb_conduction_icd","qt_interval_icd","mi_ischemia_icd","hypertrophy_icd"]}
rows=[]
for pair in pairs.itertuples(index=False):
    for target in tasks[pair.cohort]:
        for seed in (4311,4312,4313):
            rows.append({"task_index":len(rows),"model_suffix":pair.model_suffix,"cohort":pair.cohort,"target":target,"seed":seed})
out=ROOT/"results/external_benchmark_v1/steering_manifest.csv"
pd.DataFrame(rows).to_csv(out,index=False); print(out,len(rows))
