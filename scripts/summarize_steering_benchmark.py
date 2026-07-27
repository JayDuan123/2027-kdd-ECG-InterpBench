#!/usr/bin/env python
"""Patient-bootstrap Tier 1/2/3 summaries for the steering benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_v1"
WBI_EPS = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260711)
    return p.parse_args()


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float); out = np.full(len(p), np.nan); valid = np.isfinite(p)
    pv = p[valid]; order = np.argsort(pv); ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(len(pv)); tmp[order] = np.minimum(ranked, 1.0); out[np.where(valid)[0]] = tmp
    return out


def markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for values in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def focus_mask(labels: np.ndarray, kind: str, threshold: float) -> np.ndarray:
    valid = np.isfinite(labels)
    return valid & (labels == 1 if kind == "binary" else labels >= threshold)


def one_stats(data: dict[str, np.ndarray], result: dict, wrong_deltas: list[np.ndarray] | None = None,
              row_idx: np.ndarray | None = None) -> dict[str, float]:
    names = data["target_names"].astype(str).tolist(); kinds = data["target_types"].astype(str).tolist()
    target = result["target"]; j = names.index(target)
    idx = np.arange(len(data["patient_ids"])) if row_idx is None else row_idx
    labels = data["labels"][idx]; base = data["baseline_logits"][idx]
    delta = data["top5_delta"][idx]; random_delta = data["random_top5_delta"][idx]
    thresholds = data["thresholds_95spec"]
    focus_thresholds = result["focus_thresholds_train"]

    def normalized_effect(d: np.ndarray, head: int) -> float:
        mask = focus_mask(labels[:, head], kinds[head], float(focus_thresholds[names[head]]))
        denom = float(np.nanstd(base[np.isfinite(labels[:, head]), head]))
        return abs(float(np.nanmean(d[mask, head]))) / max(denom, 1e-8)

    ste = normalized_effect(delta, j)
    off = [normalized_effect(delta, h) for h in range(len(names)) if h != j]
    otd, otd_max = float(np.mean(off)), float(np.max(off))
    margin = ste - otd; wbi = otd / (ste + WBI_EPS)
    random_ste, random_margin, random_wbi = [], [], []
    for r in range(random_delta.shape[1]):
        rd = random_delta[:, r]
        rs = normalized_effect(rd, j)
        ro = float(np.mean([normalized_effect(rd, h) for h in range(len(names)) if h != j]))
        random_ste.append(rs); random_margin.append(rs - ro); random_wbi.append(ro / (rs + WBI_EPS))
    wrong_effects = []
    for wrong in wrong_deltas or []:
        wd = wrong if row_idx is None else wrong[row_idx]
        wrong_effects.append(normalized_effect(wd, j))
    max_wrong = float(np.max(wrong_effects)) if wrong_effects else float("nan")

    y = labels[:, j]; valid = np.isfinite(y); clean = base[:, j]; edit = clean + delta[:, j]
    random_edit = clean[:, None] + random_delta[:, :, j]
    if kinds[j] == "binary":
        pos = valid & (y == 1); threshold = float(thresholds[j])
        clean_sens = float((clean[pos] >= threshold).mean())
        behavior = clean_sens - float((edit[pos] >= threshold).mean())
        random_behavior = clean_sens - (random_edit[pos] >= threshold).mean(axis=0)
    else:
        mu = float(data["continuous_target_means"][j]); sd = float(data["continuous_target_stds"][j])
        ys = (y[valid] - mu) / sd; cleanv = clean[valid]; editv = edit[valid]; rv = random_edit[valid]
        sst = float(((ys - ys.mean()) ** 2).sum())
        clean_sse = float(((ys - cleanv) ** 2).sum())
        behavior = (float(((ys - editv) ** 2).sum()) - clean_sse) / max(sst, 1e-8)
        random_behavior = (((ys[:, None] - rv) ** 2).sum(axis=0) - clean_sse) / max(sst, 1e-8)
    return {
        "ste": ste, "otd_mean": otd, "otd_max": otd_max, "selectivity_margin": margin, "wbi": wbi,
        "random_ste_mean": float(np.mean(random_ste)), "random_margin_mean": float(np.mean(random_margin)),
        "random_wbi_mean": float(np.mean(random_wbi)), "tier1_excess_attribution": ste - float(np.mean(random_ste)),
        "excess_selectivity": margin - float(np.mean(random_margin)),
        "wbi_improvement": float(np.mean(random_wbi)) - wbi,
        "max_any_wrong_ste": max_wrong,
        "wrong_atom_margin": ste - max_wrong if np.isfinite(max_wrong) else float("nan"),
        "behavior_effect": behavior, "random_behavior_mean": float(np.mean(random_behavior)),
        "behavior_excess": behavior - float(np.mean(random_behavior)),
    }


def bootstrap_indices(patient_ids: np.ndarray, n: int, rng: np.random.Generator):
    unique, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    groups = [np.where(inverse == i)[0] for i in range(len(unique))]
    for _ in range(n):
        sampled = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[i] for i in sampled])


def vectorized_bootstrap(data: dict[str, np.ndarray], result: dict, wrong_deltas: list[np.ndarray],
                         n_boot: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Exact patient bootstrap from per-patient sufficient statistics."""
    names = data["target_names"].astype(str).tolist(); kinds = data["target_types"].astype(str).tolist()
    target = result["target"]; target_j = names.index(target); n_records = len(data["patient_ids"])
    _, inverse = np.unique(data["patient_ids"].astype(str), return_inverse=True)
    n_patients = int(inverse.max()) + 1
    weights = rng.multinomial(n_patients, np.full(n_patients, 1.0 / n_patients), size=n_boot).astype(np.float64)
    labels = data["labels"]; base = data["baseline_logits"]; delta = data["top5_delta"]
    random_delta = data["random_top5_delta"]; focus_thresholds = result["focus_thresholds_train"]

    def patient_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        shape = (n_patients,) + values.shape[1:]
        out = np.zeros(shape, dtype=np.float64)
        np.add.at(out, inverse[mask], values[mask])
        return out

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sums = weights @ patient_sum(values, mask)
        counts = weights @ patient_sum(np.ones((n_records, 1)), mask)
        return sums / np.maximum(counts, 1.0)

    effects = []; random_effects = []
    for j, name in enumerate(names):
        valid = np.isfinite(labels[:, j])
        focus = focus_mask(labels[:, j], kinds[j], float(focus_thresholds[name]))
        count = weights @ patient_sum(np.ones((n_records, 1)), valid)
        bsum = weights @ patient_sum(base[:, j:j + 1], valid)
        bsq = weights @ patient_sum(base[:, j:j + 1] ** 2, valid)
        variance = np.maximum(bsq / np.maximum(count, 1.0) - (bsum / np.maximum(count, 1.0)) ** 2, 1e-12)
        sd = np.sqrt(variance)
        effects.append(np.abs(weighted_mean(delta[:, j:j + 1], focus)[:, 0]) / sd[:, 0])
        random_effects.append(np.abs(weighted_mean(random_delta[:, :, j], focus)) / sd)
    effects = np.column_stack(effects)
    random_effects = np.stack(random_effects, axis=2)  # bootstrap x random x head
    ste = effects[:, target_j]; off = np.mean(np.delete(effects, target_j, axis=1), axis=1)
    random_ste = random_effects[:, :, target_j]
    random_off = np.mean(np.delete(random_effects, target_j, axis=2), axis=2)
    margin = ste - off; random_margin = random_ste - random_off
    wbi = off / (ste + WBI_EPS); random_wbi = random_off / (random_ste + WBI_EPS)

    target_valid = np.isfinite(labels[:, target_j])
    target_focus = focus_mask(labels[:, target_j], kinds[target_j], float(focus_thresholds[target]))
    target_count = weights @ patient_sum(np.ones((n_records, 1)), target_focus)
    bsum = weights @ patient_sum(base[:, target_j:target_j + 1], target_valid)
    bsq = weights @ patient_sum(base[:, target_j:target_j + 1] ** 2, target_valid)
    bcount = weights @ patient_sum(np.ones((n_records, 1)), target_valid)
    target_sd = np.sqrt(np.maximum(bsq / np.maximum(bcount, 1.0) - (bsum / np.maximum(bcount, 1.0)) ** 2, 1e-12))[:, 0]
    wrong_effects = []
    for wrong in wrong_deltas:
        wrong_mean = weighted_mean(wrong[:, target_j:target_j + 1], target_focus)[:, 0]
        wrong_effects.append(np.abs(wrong_mean) / target_sd)
    max_wrong = np.max(np.column_stack(wrong_effects), axis=1) if wrong_effects else np.full(n_boot, np.nan)

    y = labels[:, target_j]; clean = base[:, target_j]; edit = clean + delta[:, target_j]
    random_edit = clean[:, None] + random_delta[:, :, target_j]
    if kinds[target_j] == "binary":
        threshold = float(data["thresholds_95spec"][target_j]); pos = target_focus
        behavior_values = ((clean >= threshold).astype(float) - (edit >= threshold).astype(float))[:, None]
        random_values = (clean[:, None] >= threshold).astype(float) - (random_edit >= threshold).astype(float)
        behavior = weighted_mean(behavior_values, pos)[:, 0]
        random_behavior = weighted_mean(random_values, pos)
    else:
        mu = float(data["continuous_target_means"][target_j]); sd = float(data["continuous_target_stds"][target_j])
        ys = (y - mu) / sd; valid = target_valid
        err_clean = (ys - clean) ** 2
        behavior_values = ((ys - edit) ** 2 - err_clean)[:, None]
        random_values = (ys[:, None] - random_edit) ** 2 - err_clean[:, None]
        numerator = weights @ patient_sum(behavior_values, valid)
        random_numerator = weights @ patient_sum(random_values, valid)
        count = weights @ patient_sum(np.ones((n_records, 1)), valid)
        ysum = weights @ patient_sum(ys[:, None], valid)
        ysq = weights @ patient_sum((ys ** 2)[:, None], valid)
        sst = np.maximum(ysq - ysum ** 2 / np.maximum(count, 1.0), 1e-8)
        behavior = numerator[:, 0] / sst[:, 0]
        random_behavior = random_numerator / sst
    return {
        "tier1_excess_attribution": ste - random_ste.mean(axis=1),
        "excess_selectivity": margin - random_margin.mean(axis=1),
        "wbi_improvement": random_wbi.mean(axis=1) - wbi,
        "wrong_atom_margin": ste - max_wrong,
        "behavior_excess": behavior - random_behavior.mean(axis=1),
    }


def main() -> None:
    a = parse_args(); registry = pd.read_csv(a.base / "target_registry.csv").set_index("target")
    entries = []
    for result_path in sorted((a.base / "tasks").glob("seed*/*/result.json")):
        result = json.loads(result_path.read_text()); npz_path = result_path.with_name("records.npz")
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as loaded:
            data = {k: loaded[k] for k in loaded.files}
        entries.append((result, data))
    rows = []
    for result, data in entries:
        wrong = [other_data["top5_delta"] for other_result, other_data in entries
                 if int(other_result["seed"]) == int(result["seed"])
                 and other_result["target"] != result["target"]
                 and registry.loc[other_result["target"], "analysis_role"] != "nuisance_control"]
        wrong_cross = [other_data["top5_delta"] for other_result, other_data in entries
                       if int(other_result["seed"]) == int(result["seed"])
                       and other_result["target"] != result["target"]
                       and other_result["family"] != result["family"]
                       and registry.loc[other_result["target"], "analysis_role"] != "nuisance_control"]
        point = one_stats(data, result, wrong)
        point["max_cross_family_wrong_ste"] = one_stats(data, result, wrong_cross)["max_any_wrong_ste"]
        rng = np.random.default_rng(a.seed + int(result["seed"]) + sum(map(ord, result["target"])))
        samples = vectorized_bootstrap(data, result, wrong, a.bootstrap, rng)
        row = {"target": result["target"], "seed": result["seed"], "target_type": result["target_type"],
               "family": result["family"], "analysis_role": registry.loc[result["target"], "analysis_role"], **point}
        for metric in ("tier1_excess_attribution", "excess_selectivity", "wbi_improvement", "wrong_atom_margin", "behavior_excess"):
            values = np.asarray(samples[metric])
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.quantile(values, [0.025, 0.975])
            row[f"{metric}_p_one_sided"] = (1.0 + float((values <= 0).sum())) / (len(values) + 1.0)
        rows.append(row)
    if not entries:
        raise RuntimeError("No completed steering tasks found")
    cells = pd.DataFrame(rows)
    for metric in ("tier1_excess_attribution", "excess_selectivity", "wbi_improvement", "wrong_atom_margin", "behavior_excess"):
        cells[f"{metric}_q"] = bh(cells[f"{metric}_p_one_sided"].to_numpy())
    cells["tier0_fidelity"] = True
    cells["tier1_sparse_attribution"] = (cells.tier1_excess_attribution_ci_low > 0) & (cells.tier1_excess_attribution_q < 0.05)
    cells["tier2_selective_steering"] = ((cells.excess_selectivity_ci_low > 0) & (cells.wbi_improvement_ci_low > 0)
                                             & (cells.wrong_atom_margin_ci_low > 0)
                                             & (cells.excess_selectivity_q < 0.05) & (cells.wbi_improvement_q < 0.05)
                                             & (cells.wrong_atom_margin_q < 0.05))
    cells["tier3_behavior_changing"] = (cells.behavior_excess_ci_low > 0) & (cells.behavior_excess_q < 0.05)
    cells["tier4_waveform_causal"] = False
    summary = a.base / "summary"; summary.mkdir(parents=True, exist_ok=True)
    cells.to_csv(summary / "steering_cells.csv", index=False)

    grouped = cells.groupby(["target", "target_type", "family", "analysis_role"], as_index=False).agg(
        seeds=("seed", "nunique"), ste_mean=("ste", "mean"), ste_sd=("ste", "std"),
        excess_selectivity_mean=("excess_selectivity", "mean"), wbi_median=("wbi", "median"),
        tier1_seed_pass=("tier1_sparse_attribution", "sum"), tier2_seed_pass=("tier2_selective_steering", "sum"),
        tier3_seed_pass=("tier3_behavior_changing", "sum"))
    grouped["robustness"] = np.select([grouped.tier2_seed_pass == 3, grouped.tier2_seed_pass == 2],
                                      ["robust_3_of_3", "suggestive_2_of_3"], default="unstable_or_null")
    grouped.to_csv(summary / "steering_by_target.csv", index=False)

    novelty = cells[cells.analysis_role.eq("main")]
    lines = ["# Hierarchical SAE Steering Benchmark", "",
             "## Frozen protocol", "",
             "- Primary intervention: train-centroid clamp of the five highest train-only IG SAE atoms.",
             f"- WBI uses epsilon={WBI_EPS} in standardized-effect units; raw STE and OTD remain separately reported.",
             "- Tier 2 requires patient-bootstrap evidence for excess selectivity and WBI improvement over matched random five-atom clamps, plus superiority to the strongest other non-nuisance target's wrong-atom intervention; cross-family maxima are also reported.",
             "- Binary Tier 3 uses sensitivity at a validation-frozen 95% specificity threshold; continuous Tier 3 uses held-out R2 damage.",
             "- Tier 4 waveform-level causality is not claimed.", "",
             "## Main counts", "",
             f"- Novelty cells: {len(novelty)} ({novelty.target.nunique()} targets x {novelty.seed.nunique()} seeds).",
             f"- Tier 1 pass: {int(novelty.tier1_sparse_attribution.sum())}/{len(novelty)}.",
             f"- Tier 2 pass: {int(novelty.tier2_selective_steering.sum())}/{len(novelty)}.",
             f"- Tier 3 pass: {int(novelty.tier3_behavior_changing.sum())}/{len(novelty)}.", "",
             "## Target profile", "",
             markdown_table(grouped), "",
             "## Claim boundary", "",
             "A Tier 2 pass supports selective manipulation of frozen readouts in SAE code space. It does not demonstrate a physiologically valid edited ECG waveform."]
    (summary / "steering_benchmark_report.md").write_text("\n".join(lines) + "\n")
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()
