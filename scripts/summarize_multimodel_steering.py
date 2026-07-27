#!/usr/bin/env python
"""Cross-model hierarchical steering summary with per-model FDR panels."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"
spec = importlib.util.spec_from_file_location("single_summary", ROOT / "scripts/summarize_steering_benchmark.py")
single = importlib.util.module_from_spec(spec); spec.loader.exec_module(single)


def parse_args():
    p = argparse.ArgumentParser(); p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--bootstrap", type=int, default=2000); p.add_argument("--seed", type=int, default=20260712)
    p.add_argument("--fdr-panel", choices=("model", "model_family"), default="model")
    return p.parse_args()


def main() -> None:
    a = parse_args(); registry = pd.read_csv(a.base / "target_registry.csv").set_index("target")
    gate_path = a.base / "model_target_gate.csv"
    gate = pd.read_csv(gate_path).set_index(["model", "target"]) if gate_path.exists() else None
    operating = pd.read_csv(a.base / "selected_operating_points.csv").set_index("model")
    entries = []
    for result_path in sorted((a.base / "models").glob("*/tasks/seed*/*/result.json")):
        result = json.loads(result_path.read_text()); npz_path = result_path.with_name("records.npz")
        if not npz_path.exists(): continue
        with np.load(npz_path, allow_pickle=False) as loaded: data = {k: loaded[k] for k in loaded.files}
        entries.append((result, data))
    if not entries: raise RuntimeError("No completed multimodel tasks")
    rows = []
    for result, data in entries:
        safe_model = result["model"].lower().replace("-", "_")
        head_metrics = json.loads((a.base / "models" / safe_model / "frozen_heads.metrics.json").read_text())
        wrong = [od["top5_delta"] for other, od in entries if other["model"] == result["model"]
                 and int(other["seed"]) == int(result["seed"]) and other["target"] != result["target"]
                 and registry.loc[other["target"], "analysis_role"] != "nuisance_control"]
        wrong_cross = [od["top5_delta"] for other, od in entries if other["model"] == result["model"]
                       and int(other["seed"]) == int(result["seed"]) and other["target"] != result["target"]
                       and other["family"] != result["family"]
                       and registry.loc[other["target"], "analysis_role"] != "nuisance_control"]
        point = single.one_stats(data, result, wrong)
        point["max_cross_family_wrong_ste"] = single.one_stats(data, result, wrong_cross)["max_any_wrong_ste"]
        rng = np.random.default_rng(a.seed + int(result["seed"]) + sum(map(ord, result["model"] + result["target"])))
        samples = single.vectorized_bootstrap(data, result, wrong, a.bootstrap, rng)
        target_metric = result["metrics"]["top5"][result["target"]]
        if result["target_type"] == "binary":
            raw_readout = float(head_metrics[result["target"]]["test_auroc"]); recon_readout = float(target_metric["baseline_auroc"])
            readout_warning = raw_readout < 0.70
        else:
            raw_readout = float(head_metrics[result["target"]]["test_r2"]); recon_readout = float(target_metric["baseline_r2"])
            readout_warning = raw_readout < 0.20
        retention = recon_readout / raw_readout if raw_readout > 1e-8 else float("nan")
        gate_row = gate.loc[(result["model"], result["target"])] if gate is not None else None
        row = {"model": result["model"], "target": result["target"], "seed": result["seed"],
               "target_type": result["target_type"], "family": result["family"],
               "analysis_role": registry.loc[result["target"], "analysis_role"],
               "raw_readout_test_metric": raw_readout, "sae_recon_readout_metric": recon_readout,
               "sae_readout_retention": retention, "readout_quality_warning": bool(readout_warning or retention < .95),
               "bootstrap_samples": a.bootstrap, "test_patients": int(len(np.unique(data["patient_ids"].astype(str)))),
               "headline_eligible": bool(gate_row.headline_eligible) if gate_row is not None else True,
               "analysis_tier": str(gate_row.analysis_tier) if gate_row is not None else "headline_cross_model",
               "recon_R2_seed4311": operating.loc[result["model"], "recon_R2"],
               "N": operating.loc[result["model"], "N"], "k": operating.loc[result["model"], "k"], **point}
        for metric in ("tier1_excess_attribution", "excess_selectivity", "wbi_improvement", "wrong_atom_margin", "behavior_excess"):
            values = np.asarray(samples[metric]); row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(values, [0.025, 0.975])
            row[f"{metric}_p_one_sided"] = (1 + float((values <= 0).sum())) / (len(values) + 1)
        rows.append(row)
    cells = pd.DataFrame(rows)
    metrics = ("tier1_excess_attribution", "excess_selectivity", "wbi_improvement", "wrong_atom_margin", "behavior_excess")
    for metric in metrics:
        cells[f"{metric}_q"] = np.nan
        panel = ["model", "family"] if a.fdr_panel == "model_family" else ["model"]
        for _, idx in cells.groupby(panel).groups.items():
            cells.loc[idx, f"{metric}_q"] = single.bh(cells.loc[idx, f"{metric}_p_one_sided"].to_numpy())
    cells["tier0_fidelity"] = cells.model.map(operating.in_band).astype(bool)
    cells["tier1_sparse_attribution"] = cells.tier0_fidelity & (cells.tier1_excess_attribution_ci_low > 0) & (cells.tier1_excess_attribution_q < .05)
    cells["tier2_selective_steering"] = (cells.tier0_fidelity & (cells.excess_selectivity_ci_low > 0) & (cells.wbi_improvement_ci_low > 0)
        & (cells.wrong_atom_margin_ci_low > 0) & (cells.excess_selectivity_q < .05) & (cells.wbi_improvement_q < .05) & (cells.wrong_atom_margin_q < .05))
    cells["tier3_behavior_changing"] = cells.tier0_fidelity & (cells.behavior_excess_ci_low > 0) & (cells.behavior_excess_q < .05)
    cells["tier4_waveform_causal"] = False
    out = a.base / "summary"; out.mkdir(parents=True, exist_ok=True); cells.to_csv(out / "multimodel_steering_cells.csv", index=False)
    profile = cells.groupby(["model", "target", "analysis_role", "analysis_tier", "headline_eligible"], as_index=False).agg(
        seeds=("seed", "nunique"), tier1_pass=("tier1_sparse_attribution", "sum"), tier2_pass=("tier2_selective_steering", "sum"),
        tier3_pass=("tier3_behavior_changing", "sum"), ste_mean=("ste", "mean"), otd_mean=("otd_mean", "mean"),
        max_wrong_mean=("max_any_wrong_ste", "mean"), wbi_median=("wbi", "median"),
        raw_readout_metric=("raw_readout_test_metric", "mean"), sae_retention_mean=("sae_readout_retention", "mean"),
        quality_warning_seeds=("readout_quality_warning", "sum"))
    profile["tier2_robustness"] = np.select([profile.tier2_pass == 3, profile.tier2_pass == 2], ["robust_3_of_3", "suggestive_2_of_3"], default="unstable_or_null")
    profile.to_csv(out / "multimodel_target_profile.csv", index=False)
    main = cells[cells.analysis_role.eq("main") & cells.headline_eligible]
    model_summary = main.groupby("model", as_index=False).agg(cells=("target", "size"), targets=("target", "nunique"),
        tier1_pass=("tier1_sparse_attribution", "sum"), tier2_pass=("tier2_selective_steering", "sum"), tier3_pass=("tier3_behavior_changing", "sum"))
    robust = profile[(profile.analysis_role == "main") & (profile.tier2_pass == 3)].groupby("model").size()
    model_summary["robust_tier2_targets"] = model_summary.model.map(robust).fillna(0).astype(int)
    qualified = main[~main.readout_quality_warning]
    qsummary = qualified.groupby("model").agg(quality_qualified_cells=("target", "size"),
        qualified_tier1_pass=("tier1_sparse_attribution", "sum"), qualified_tier2_pass=("tier2_selective_steering", "sum"),
        qualified_tier3_pass=("tier3_behavior_changing", "sum"))
    model_summary = model_summary.merge(qsummary, on="model", how="left")
    model_summary.to_csv(out / "multimodel_model_profile.csv", index=False)
    lines = ["# Multimodel Hierarchical SAE Steering Audit", "", "## Frozen denominator", "",
             f"- Models: {main.model.nunique()}; main targets: {main.target.nunique()}; main seed-level cells: {len(main)}.",
             f"- Tier 1: {int(main.tier1_sparse_attribution.sum())}/{len(main)}.",
             f"- Tier 2: {int(main.tier2_selective_steering.sum())}/{len(main)}.",
             f"- Tier 3: {int(main.tier3_behavior_changing.sum())}/{len(main)}.", "",
             "## Readout-quality sensitivity", "",
             f"- Quality-qualified cells: {len(qualified)}/{len(main)}.",
             f"- Qualified Tier 1/Tier 2/Tier 3: {int(qualified.tier1_sparse_attribution.sum())}/{int(qualified.tier2_selective_steering.sum())}/{int(qualified.tier3_behavior_changing.sum())} of {len(qualified)}.", "",
             "## Model profiles (not a leaderboard)", "", single.markdown_table(model_summary), "",
             "Tier 2 remains a frozen-readout code-space intervention. Tier 4 waveform causality is not claimed."]
    (out / "multimodel_steering_report.md").write_text("\n".join(lines) + "\n")
    print(model_summary.to_string(index=False))


if __name__ == "__main__": main()
