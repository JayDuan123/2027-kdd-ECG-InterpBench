#!/usr/bin/env python
"""Audit all dose/direction target workers and apply global FDR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_external_dose_direction import ALPHAS, METRICS, OUT, TOP_K, bh, freeze_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = freeze_panel()
    frames = []
    for panel_index in range(len(panel)):
        worker = args.out / "workers" / f"panel_{panel_index:02d}"
        metadata = json.loads((worker / "metadata.json").read_text())
        if metadata["panel_index"] != panel_index or metadata["panel_targets"] != 1:
            raise RuntimeError(f"Invalid dose worker metadata: {worker}")
        frame = pd.read_csv(worker / "dose_direction_seed_cells.csv")
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True)
    keys = ["model_suffix", "cohort", "target", "seed", "top_k", "mode", "alpha"]
    if frame.duplicated(keys).any():
        raise RuntimeError("Duplicate dose/direction cells across target workers")
    expected = len(panel) * 3 * len(TOP_K) * (len(ALPHAS) + 1)
    if len(frame) != expected:
        raise RuntimeError(f"Dose/direction matrix incomplete: {len(frame)} != {expected}")

    for metric in METRICS:
        frame[f"{metric}_q"] = bh(frame[f"{metric}_p_one_sided"].to_numpy())
    args.out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(args.out / "frozen_target_panel.csv", index=False)
    frame.to_csv(args.out / "dose_direction_seed_cells.csv", index=False)
    profile = frame.groupby(
        ["model", "cohort", "target", "family", "panel_role", "top_k", "mode", "alpha", "direction"],
        dropna=False, as_index=False,
    ).agg(
        seeds=("seed", "nunique"),
        signed_target_change_mean=("signed_target_change", "mean"),
        ste_mean=("ste", "mean"),
        otd_mean=("otd_mean", "mean"),
        wbi_mean=("wbi", "mean"),
        excess_selectivity_mean=("excess_selectivity", "mean"),
        behavior_excess_mean=("behavior_excess", "mean"),
        selectivity_q05_seeds=("excess_selectivity_q", lambda x: int((x < 0.05).sum())),
        behavior_q05_seeds=("behavior_excess_q", lambda x: int((x < 0.05).sum())),
    )
    profile.to_csv(args.out / "dose_direction_profile.csv", index=False)

    monotonic_rows = []
    centroid = profile[profile["mode"].eq("centroid_scale")]
    for keys_value, group in centroid.groupby(["model", "cohort", "target", "panel_role", "top_k"]):
        positive = group[group.alpha.ge(0)].sort_values("alpha")
        negative = group[group.alpha.le(0)].sort_values("alpha")
        at_pos = group[group.alpha.eq(1.0)].signed_target_change_mean
        at_neg = group[group.alpha.eq(-1.0)].signed_target_change_mean
        monotonic_rows.append(
            dict(
                zip(("model", "cohort", "target", "panel_role", "top_k"), keys_value),
                positive_dose_signed_rho=float(spearmanr(positive.alpha, positive.signed_target_change_mean).statistic),
                positive_dose_behavior_rho=float(spearmanr(positive.alpha, positive.behavior_excess_mean).statistic),
                alpha_plus_minus_one_sign_reversal=bool(
                    len(at_pos) == 1 and len(at_neg) == 1
                    and float(at_pos.iloc[0]) * float(at_neg.iloc[0]) < 0
                ),
                positive_doses=len(positive),
                negative_doses=len(negative),
            )
        )
    pd.DataFrame(monotonic_rows).to_csv(args.out / "dose_monotonicity_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "panel_targets": len(panel),
        "top_k": list(TOP_K),
        "centroid_alphas": list(ALPHAS),
        "zero_ablation": True,
        "bootstrap_samples": int(frame.bootstrap_samples.iloc[0]),
        "workers": len(panel),
        "fdr_family": "all panel x seed x top-k x dose/mode cells, separately by metric",
        "all_complete": True,
        "interpretation_note": "Frozen linear readouts make logit deltas algebraically linear in alpha; behavior/selectivity tradeoffs are primary.",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(profile.groupby(["panel_role", "mode"]).size().to_string())


if __name__ == "__main__":
    main()
