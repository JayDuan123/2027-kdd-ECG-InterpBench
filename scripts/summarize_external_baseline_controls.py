#!/usr/bin/env python
"""Combine baseline-control workers and apply pre-registered FDR families."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "benchmark_extension_v1" / "baseline_controls"
from scripts.benchmark_extension_common import bh  # noqa: E402

METRICS = ("ste", "otd_mean", "selectivity_margin", "wbi", "behavior_effect")


def main() -> None:
    method_paths = sorted((BASE / "workers").glob("*/method_seed_cells.csv"))
    contrast_paths = sorted((BASE / "workers").glob("*/paired_method_contrasts.csv"))
    if len(method_paths) != 3 or len(contrast_paths) != 3:
        raise RuntimeError(f"Expected three complete cohort workers, found {len(method_paths)} and {len(contrast_paths)}")
    methods = pd.concat([pd.read_csv(path) for path in method_paths], ignore_index=True)
    contrasts = pd.concat([pd.read_csv(path) for path in contrast_paths], ignore_index=True)
    for metric in METRICS:
        p_col = f"delta_{metric}_p_one_sided"
        contrasts[f"delta_{metric}_q"] = bh(contrasts[p_col].to_numpy())
    BASE.mkdir(parents=True, exist_ok=True)
    methods.to_csv(BASE / "baseline_method_seed_cells.csv", index=False)
    contrasts.to_csv(BASE / "baseline_paired_contrasts.csv", index=False)
    methods["target_key"] = methods.cohort.astype(str) + "|" + methods.target.astype(str)
    contrasts["target_key"] = contrasts.cohort.astype(str) + "|" + contrasts.target.astype(str)
    method_profile = methods.groupby(["method", "panel_role"], as_index=False).agg(
        seed_cells=("target", "size"), target_pairs=("target_key", "nunique"),
        ste_mean=("ste", "mean"), otd_mean=("otd_mean", "mean"),
        selectivity_margin_mean=("selectivity_margin", "mean"),
        wbi_mean=("wbi", "mean"), behavior_effect_mean=("behavior_effect", "mean"),
    )
    method_profile.to_csv(BASE / "baseline_method_profile.csv", index=False)
    contrast_profile = contrasts.groupby(["contrast", "panel_role"], as_index=False).agg(
        seed_cells=("target", "size"), target_pairs=("target_key", "nunique"),
        delta_ste_mean=("delta_ste", "mean"), delta_otd_mean=("delta_otd_mean", "mean"),
        delta_selectivity_mean=("delta_selectivity_margin", "mean"),
        delta_wbi_mean=("delta_wbi", "mean"),
        delta_behavior_mean=("delta_behavior_effect", "mean"),
        ste_q05_cells=("delta_ste_q", lambda x: int((x < .05).sum())),
        selectivity_q05_cells=("delta_selectivity_margin_q", lambda x: int((x < .05).sum())),
        wbi_q05_cells=("delta_wbi_q", lambda x: int((x < .05).sum())),
        behavior_q05_cells=("delta_behavior_effect_q", lambda x: int((x < .05).sum())),
    )
    contrast_profile.to_csv(BASE / "baseline_contrast_profile.csv", index=False)
    metadata = {
        "schema_version": 1, "method_rows": len(methods), "contrast_rows": len(contrasts),
        "cohorts": sorted(methods.cohort.unique().tolist()),
        "targets": int(methods[["cohort", "target"]].drop_duplicates().shape[0]),
        "fdr_family": "all panel x seed contrasts, separately by metric",
        "all_complete": True,
    }
    (BASE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(method_profile.to_string(index=False))
    print(contrast_profile.to_string(index=False))


if __name__ == "__main__":
    main()
