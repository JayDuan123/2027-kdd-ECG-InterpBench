#!/usr/bin/env python
"""Shared statistics and artifact helpers for benchmark_extension_v1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


WBI_EPS = 0.05


def bh(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return out
    p = values[valid]
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    out[np.where(valid)[0]] = restored
    return out


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: loaded[key] for key in loaded.files}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def focus_mask(labels: np.ndarray, kind: str, threshold: float) -> np.ndarray:
    valid = np.isfinite(labels)
    return valid & (labels == 1 if kind == "binary" else labels >= threshold)


def group_bootstrap_weights(
    group_ids: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Return multinomial group weights and row-to-group indices."""
    _, inverse = np.unique(group_ids.astype(str), return_inverse=True)
    n_groups = int(inverse.max()) + 1
    weights = rng.multinomial(
        n_groups, np.full(n_groups, 1.0 / n_groups), size=n_bootstrap
    ).astype(np.float64)
    return weights, inverse


def bootstrap_steering_metrics(
    data: dict[str, np.ndarray],
    result: dict,
    weights: np.ndarray,
    inverse: np.ndarray,
    wrong_deltas: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Exact grouped bootstrap metrics using caller-provided paired weights."""
    names = data["target_names"].astype(str).tolist()
    kinds = data["target_types"].astype(str).tolist()
    target_j = names.index(result["target"])
    labels = np.asarray(data["labels"], dtype=np.float64)
    base = np.asarray(data["baseline_logits"], dtype=np.float64)
    delta = np.asarray(data["top5_delta"], dtype=np.float64)
    random_delta = np.asarray(data["random_top5_delta"], dtype=np.float64)
    n_records = len(inverse)
    n_groups = weights.shape[1]

    def group_sum(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        out = np.zeros((n_groups,) + values.shape[1:], dtype=np.float64)
        np.add.at(out, inverse[mask], values[mask])
        return out

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sums = weights @ group_sum(values, mask)
        counts = weights @ group_sum(np.ones((n_records, 1)), mask)
        return sums / np.maximum(counts, 1.0)

    effects = []
    random_effects = []
    focus_thresholds = result["focus_thresholds_train"]
    for head, name in enumerate(names):
        valid = np.isfinite(labels[:, head])
        focus = focus_mask(labels[:, head], kinds[head], float(focus_thresholds[name]))
        count = weights @ group_sum(np.ones((n_records, 1)), valid)
        baseline = base[:, head : head + 1]
        baseline_sum = weights @ group_sum(baseline, valid)
        baseline_sq_sum = weights @ group_sum(baseline**2, valid)
        variance = np.maximum(
            baseline_sq_sum / np.maximum(count, 1.0)
            - (baseline_sum / np.maximum(count, 1.0)) ** 2,
            1e-12,
        )
        sd = np.sqrt(variance)
        effects.append(np.abs(weighted_mean(delta[:, head : head + 1], focus)[:, 0]) / sd[:, 0])
        random_effects.append(np.abs(weighted_mean(random_delta[:, :, head], focus)) / sd)

    effects = np.column_stack(effects)
    random_effects = np.stack(random_effects, axis=2)
    ste = effects[:, target_j]
    otd = np.mean(np.delete(effects, target_j, axis=1), axis=1)
    random_ste = random_effects[:, :, target_j]
    random_otd = np.mean(np.delete(random_effects, target_j, axis=2), axis=2)
    margin = ste - otd
    random_margin = random_ste - random_otd
    wbi = otd / (ste + WBI_EPS)
    random_wbi = random_otd / (random_ste + WBI_EPS)

    target_valid = np.isfinite(labels[:, target_j])
    target_focus = focus_mask(
        labels[:, target_j], kinds[target_j], float(focus_thresholds[result["target"]])
    )
    target_baseline = base[:, target_j : target_j + 1]
    base_count = weights @ group_sum(np.ones((n_records, 1)), target_valid)
    base_sum = weights @ group_sum(target_baseline, target_valid)
    base_sq_sum = weights @ group_sum(target_baseline**2, target_valid)
    target_sd = np.sqrt(
        np.maximum(
            base_sq_sum / np.maximum(base_count, 1.0)
            - (base_sum / np.maximum(base_count, 1.0)) ** 2,
            1e-12,
        )
    )[:, 0]
    wrong_effects = []
    for wrong in wrong_deltas or []:
        wrong_mean = weighted_mean(
            np.asarray(wrong, dtype=np.float64)[:, target_j : target_j + 1], target_focus
        )[:, 0]
        wrong_effects.append(np.abs(wrong_mean) / target_sd)
    max_wrong = (
        np.max(np.column_stack(wrong_effects), axis=1)
        if wrong_effects
        else np.full(len(weights), np.nan)
    )

    y = labels[:, target_j]
    clean = base[:, target_j]
    edited = clean + delta[:, target_j]
    random_edited = clean[:, None] + random_delta[:, :, target_j]
    if kinds[target_j] == "binary":
        threshold = float(data["thresholds_95spec"][target_j])
        behavior_values = (
            (clean >= threshold).astype(float) - (edited >= threshold).astype(float)
        )[:, None]
        random_values = (
            (clean[:, None] >= threshold).astype(float)
            - (random_edited >= threshold).astype(float)
        )
        behavior = weighted_mean(behavior_values, target_focus)[:, 0]
        random_behavior = weighted_mean(random_values, target_focus)
    else:
        mean = float(data["continuous_target_means"][target_j])
        sd = float(data["continuous_target_stds"][target_j])
        standardized = (y - mean) / sd
        clean_error = (standardized - clean) ** 2
        behavior_values = ((standardized - edited) ** 2 - clean_error)[:, None]
        random_values = (standardized[:, None] - random_edited) ** 2 - clean_error[:, None]
        numerator = weights @ group_sum(behavior_values, target_valid)
        random_numerator = weights @ group_sum(random_values, target_valid)
        count = weights @ group_sum(np.ones((n_records, 1)), target_valid)
        y_sum = weights @ group_sum(standardized[:, None], target_valid)
        y_sq_sum = weights @ group_sum((standardized**2)[:, None], target_valid)
        sst = np.maximum(y_sq_sum - y_sum**2 / np.maximum(count, 1.0), 1e-8)
        behavior = numerator[:, 0] / sst[:, 0]
        random_behavior = random_numerator / sst

    return {
        "ste": ste,
        "otd_mean": otd,
        "selectivity_margin": margin,
        "wbi": wbi,
        "random_ste_mean": random_ste.mean(axis=1),
        "random_otd_mean": random_otd.mean(axis=1),
        "random_selectivity_margin_mean": random_margin.mean(axis=1),
        "random_wbi_mean": random_wbi.mean(axis=1),
        "tier1_excess_attribution": ste - random_ste.mean(axis=1),
        "excess_selectivity": margin - random_margin.mean(axis=1),
        "wbi_improvement": random_wbi.mean(axis=1) - wbi,
        "wrong_atom_margin": ste - max_wrong,
        "behavior_effect": behavior,
        "random_behavior_mean": random_behavior.mean(axis=1),
        "behavior_excess": behavior - random_behavior.mean(axis=1),
    }


def interval_and_p(samples: np.ndarray, improvement_sign: int = 1) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan, "p_two_sided": np.nan}
    low, high = np.quantile(finite, [0.025, 0.975])
    oriented = finite * improvement_sign
    p_one = (1.0 + float((oriented <= 0).sum())) / (len(oriented) + 1.0)
    left = (1.0 + float((finite <= 0).sum())) / (len(finite) + 1.0)
    right = (1.0 + float((finite >= 0).sum())) / (len(finite) + 1.0)
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "p_one_sided": float(p_one),
        "p_two_sided": float(min(1.0, 2.0 * min(left, right))),
    }
