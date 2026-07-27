#!/usr/bin/env python
"""Compare matched-scale FM profiles under fixed k/d and fixed k/N."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_multiscale_sae_inference import MODEL_ORDER, bh_adjust


EXPANSIONS = (1, 4, 8, 16, 32)
SEEDS = (4311, 4312, 4313)
METRICS = {
    "reconstruction": "{split}_recon_R2",
    "semantic_alignment": "{split}_semantic_alignment",
    "concept_coverage": "{split}_concept_coverage_020",
    "dead_fraction": "{split}_dead_fraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument(
        "--sensitivity-root",
        type=Path,
        default=ROOT / "results/multiscale_sae_fixed_k_over_n_middepth_v1",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--relative-depth", type=float, default=0.5)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    return parser.parse_args()


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def normalized_log_auc(values: np.ndarray) -> np.ndarray:
    x = np.log(np.asarray(EXPANSIONS, dtype=np.float64))
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return trapezoid(values, x=x, axis=0) / (x[-1] - x[0])


def paired_bootstrap(
    sensitivity: np.ndarray, primary: np.ndarray, draws: np.ndarray
) -> tuple[float, float, float, float]:
    difference = sensitivity - primary
    boot = difference[draws].mean(axis=1)
    observed = float(difference.mean())
    low, high = np.quantile(boot, [0.025, 0.975])
    p_value = float(
        min(
            1.0,
            2.0
            * min(
                (1.0 + np.sum(boot <= 0)) / (len(boot) + 1.0),
                (1.0 + np.sum(boot >= 0)) / (len(boot) + 1.0),
            ),
        )
    )
    return observed, float(low), float(high), p_value


def validate_frame(frame: pd.DataFrame, arm: str, depth: float, label: str) -> None:
    expected = {
        (model, depth, expansion, seed)
        for model in MODEL_ORDER
        for expansion in EXPANSIONS
        for seed in SEEDS
    }
    observed = {
        (str(row.model), float(row.relative_depth), int(row.expansion_E), int(row.seed))
        for row in frame.itertuples(index=False)
    }
    if len(frame) != len(expected) or observed != expected:
        raise RuntimeError(
            f"{label} is not the complete common mid-depth grid: "
            f"rows={len(frame)}, unique={len(observed)}, expected={len(expected)}"
        )
    if set(frame.sparsity_arm) != {arm}:
        raise RuntimeError(f"{label} sparsity arm mismatch: {set(frame.sparsity_arm)}")


def main() -> None:
    args = parse_args()
    primary_audit = json.loads((args.primary_root / "audit.json").read_text())
    sensitivity_audit = json.loads((args.sensitivity_root / "audit.json").read_text())
    if not primary_audit.get("audit_pass"):
        raise RuntimeError(f"primary audit incomplete: {primary_audit}")
    if not sensitivity_audit.get("audit_pass"):
        raise RuntimeError(f"sensitivity audit incomplete: {sensitivity_audit}")

    primary_all = pd.read_csv(args.primary_root / "cell_metrics.csv")
    primary = primary_all[np.isclose(primary_all.relative_depth, args.relative_depth)].copy()
    sensitivity = pd.read_csv(args.sensitivity_root / "cell_metrics.csv")
    validate_frame(primary, "fixed_k_over_d", args.relative_depth, "primary slice")
    validate_frame(sensitivity, "fixed_k_over_n", args.relative_depth, "sensitivity")
    primary = primary.set_index(["model", "expansion_E", "seed"]).sort_index()
    sensitivity = sensitivity.set_index(["model", "expansion_E", "seed"]).sort_index()
    if not primary.index.equals(sensitivity.index):
        raise RuntimeError("primary and sensitivity cells are not paired")

    rng = np.random.default_rng(args.bootstrap_seed)
    draws = rng.integers(0, len(SEEDS), size=(args.bootstrap_samples, len(SEEDS)))
    scale_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    difference_rows: list[dict[str, Any]] = []
    auc_rows: list[dict[str, Any]] = []
    auc_difference_rows: list[dict[str, Any]] = []

    for metric_name, template in METRICS.items():
        column = template.format(split=args.split)
        higher_is_better = metric_name != "dead_fraction"
        difference_start = len(difference_rows)
        for expansion in EXPANSIONS:
            arm_means: dict[str, dict[str, float]] = {"fixed_k_over_d": {}, "fixed_k_over_n": {}}
            for model in MODEL_ORDER:
                key = (model, expansion)
                primary_values = primary.loc[key][column].reindex(SEEDS).to_numpy(dtype=float)
                sensitivity_values = sensitivity.loc[key][column].reindex(SEEDS).to_numpy(dtype=float)
                for arm, values in (
                    ("fixed_k_over_d", primary_values),
                    ("fixed_k_over_n", sensitivity_values),
                ):
                    mean = float(values.mean())
                    arm_means[arm][model] = mean
                    scale_rows.append(
                        {
                            "split": args.split,
                            "metric": metric_name,
                            "sparsity_arm": arm,
                            "model": model,
                            "relative_depth": args.relative_depth,
                            "expansion_E": expansion,
                            "mean": mean,
                            "sd_across_seeds": float(np.std(values, ddof=1)),
                            "seeds": len(SEEDS),
                        }
                    )
                observed, low, high, p_value = paired_bootstrap(
                    sensitivity_values, primary_values, draws
                )
                difference_rows.append(
                    {
                        "split": args.split,
                        "metric": metric_name,
                        "model": model,
                        "relative_depth": args.relative_depth,
                        "expansion_E": expansion,
                        "fixed_k_over_N_minus_fixed_k_over_d": observed,
                        "ci_low": low,
                        "ci_high": high,
                        "p_two_sided": p_value,
                        "q_bh": float("nan"),
                        "bootstrap_unit": "paired SAE seed",
                        "bootstrap_samples": args.bootstrap_samples,
                    }
                )
            orders = {}
            ranks = {}
            for arm in ("fixed_k_over_d", "fixed_k_over_n"):
                order = sorted(
                    MODEL_ORDER,
                    key=lambda model: arm_means[arm][model],
                    reverse=higher_is_better,
                )
                orders[arm] = order
                ranks[arm] = [order.index(model) + 1 for model in MODEL_ORDER]
            tau, p_value = kendalltau(ranks["fixed_k_over_d"], ranks["fixed_k_over_n"])
            rank_rows.append(
                {
                    "split": args.split,
                    "metric": metric_name,
                    "relative_depth": args.relative_depth,
                    "expansion_E": expansion,
                    "kendall_tau": float(tau),
                    "p_value": float(p_value),
                    "fixed_k_over_d_order": ">".join(orders["fixed_k_over_d"]),
                    "fixed_k_over_N_order": ">".join(orders["fixed_k_over_n"]),
                }
            )
        metric_indices = list(range(difference_start, len(difference_rows)))
        q_values = bh_adjust(
            np.asarray([difference_rows[index]["p_two_sided"] for index in metric_indices])
        )
        for index, q_value in zip(metric_indices, q_values):
            difference_rows[index]["q_bh"] = float(q_value)

        auc_start = len(auc_difference_rows)
        for model in MODEL_ORDER:
            primary_matrix = np.stack(
                [
                    primary.loc[(model, expansion)][column].reindex(SEEDS).to_numpy(dtype=float)
                    for expansion in EXPANSIONS
                ]
            )
            sensitivity_matrix = np.stack(
                [
                    sensitivity.loc[(model, expansion)][column].reindex(SEEDS).to_numpy(dtype=float)
                    for expansion in EXPANSIONS
                ]
            )
            primary_auc = np.asarray(normalized_log_auc(primary_matrix), dtype=float)
            sensitivity_auc = np.asarray(normalized_log_auc(sensitivity_matrix), dtype=float)
            for arm, values in (
                ("fixed_k_over_d", primary_auc),
                ("fixed_k_over_n", sensitivity_auc),
            ):
                auc_rows.append(
                    {
                        "split": args.split,
                        "metric": metric_name,
                        "sparsity_arm": arm,
                        "model": model,
                        "relative_depth": args.relative_depth,
                        "observed_common_scale_auc": float(values.mean()),
                        "sd_across_seeds": float(np.std(values, ddof=1)),
                        "seeds": len(SEEDS),
                    }
                )
            observed, low, high, p_value = paired_bootstrap(
                sensitivity_auc, primary_auc, draws
            )
            auc_difference_rows.append(
                {
                    "split": args.split,
                    "metric": metric_name,
                    "model": model,
                    "relative_depth": args.relative_depth,
                    "fixed_k_over_N_minus_fixed_k_over_d_auc": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "p_two_sided": p_value,
                    "q_bh": float("nan"),
                    "bootstrap_unit": "paired SAE seed",
                    "bootstrap_samples": args.bootstrap_samples,
                }
            )
        auc_indices = list(range(auc_start, len(auc_difference_rows)))
        q_values = bh_adjust(
            np.asarray([auc_difference_rows[index]["p_two_sided"] for index in auc_indices])
        )
        for index, q_value in zip(auc_indices, q_values):
            auc_difference_rows[index]["q_bh"] = float(q_value)

    out = args.sensitivity_root
    atomic_csv(out / f"{args.split}_sparsity_scale_profiles.csv", scale_rows)
    atomic_csv(out / f"{args.split}_sparsity_rank_agreement.csv", rank_rows)
    atomic_csv(out / f"{args.split}_sparsity_cell_differences.csv", difference_rows)
    atomic_csv(out / f"{args.split}_sparsity_auc_profiles.csv", auc_rows)
    atomic_csv(out / f"{args.split}_sparsity_auc_differences.csv", auc_difference_rows)
    e8_rows = [row for row in difference_rows if int(row["expansion_E"]) == 8]
    metadata = {
        "status": "complete",
        "split": args.split,
        "relative_depth": args.relative_depth,
        "models": MODEL_ORDER,
        "expansions": list(EXPANSIONS),
        "seeds": list(SEEDS),
        "primary_arm": "fixed_k_over_d=1/8",
        "sensitivity_arm": "fixed_k_over_N=1/64",
        "matched_cells_per_arm": len(primary),
        "rank_agreement_rows": len(rank_rows),
        "cell_difference_rows": len(difference_rows),
        "auc_difference_rows": len(auc_difference_rows),
        "e8_anchor_max_abs_difference": float(
            max(abs(row["fixed_k_over_N_minus_fixed_k_over_d"]) for row in e8_rows)
        ),
        "comparison_rule": "both arms compare all six FMs at the same E and relative depth; no per-model scale selection",
        "patient_level_inference": False,
        "claim_boundary": "mid-depth sparsity-parameterization sensitivity with paired fixed SAE seeds",
    }
    atomic = out / f"{args.split}_sparsity_sensitivity_audit.json"
    tmp = atomic.with_suffix(atomic.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, atomic)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
