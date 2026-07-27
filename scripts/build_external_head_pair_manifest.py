#!/usr/bin/env python
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "results/external_benchmark_v1/head_pair_manifest.csv"
frames = [pd.read_csv(ROOT / "results/activations_external_full_v1/plan_chapman_cpsc/plan_summary.csv")[["model", "cohort"]]]
ningbo = ROOT / "results/activations_external_full_v1/plan_ningbo/plan_summary.csv"
if ningbo.exists():
    frames.append(pd.read_csv(ningbo)[["model", "cohort"]])
models = ("csfm", "cardiac_fm", "ecg_fm", "ecg_jepa", "hubert_ecg", "st_mem")
frames.append(pd.DataFrame({"model": models, "cohort": ["mimic"] * len(models)}))
d = pd.concat(frames, ignore_index=True).drop_duplicates().copy()
suffix = {
    "csfm": "csfm_cu118_commons", "cardiac_fm": "cardiac_fm_cu118_commons",
    "ecg_fm": "ecg_fm_cu118_commons", "ecg_jepa": "ecg_jepa_cu118_commons",
    "hubert_ecg": "hubert_ecg_cu118_commons", "st_mem": "st_mem_cu118_commons",
}
d["model_suffix"] = d.model.map(suffix)
d["cohort"] = d.cohort + "_f"
d.insert(0, "task_index", range(len(d)))
out.parent.mkdir(parents=True, exist_ok=True); d.to_csv(out, index=False)
print(out, len(d))
