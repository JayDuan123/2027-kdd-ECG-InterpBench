#!/usr/bin/env python
"""Post-hoc SAE recovery curves for the CSFM robustness sweep.

This script recomputes concept-ranked SAE activation recovery from saved SAE
checkpoints. It is CPU-safe but can take a few minutes because it re-encodes
train/test activations for each completed robustness cell.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from benchmark_v1.sae_extension.csfm_environment import CSFMSAEEnvironment
from benchmark_v1.sae_extension.metrics import decomposability, select_concept_features
from benchmark_v1.sae_extension.topk_sae import TopKSAE


K_GRID = (1, 5, 10, 20, 50, 100, 200, 512)


def load_sae(checkpoint: Path, device: str = "cpu") -> TopKSAE:
    saved = torch.load(checkpoint, map_location=device)
    meta = saved["meta"]
    d = int(meta["d"])
    n_features = int(meta["n_features"])
    k = int(meta["k"])
    sae = TopKSAE(d=d, n_features=n_features, k=k).to(device)
    sae.load_state_dict(saved["sae"])
    sae.eval()
    return sae


def row_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def sd(values: list[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def median(values: list[float]) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else float("nan")


def q(values: list[float], quantile: float) -> float:
    values = [v for v in values if np.isfinite(v)]
    return float(np.quantile(values, quantile)) if values else float("nan")


def fmt_pm(mu: float, sigma: float) -> str:
    if not np.isfinite(mu):
        return "nan"
    if not np.isfinite(sigma):
        return f"{mu:.3f}"
    return f"{mu:.3f} +/- {sigma:.3f}"


def fmt_iqr(values: list[float]) -> str:
    med = median(values)
    if not np.isfinite(med):
        return "nan"
    return f"{med:.2f} [{q(values, 0.25):.2f}, {q(values, 0.75):.2f}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/sae_extension/csfm_sae_main_robustness")
    parser.add_argument("--out-prefix", default="results/sae_extension/csfm_sae_recovery_summary")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.root)
    env = CSFMSAEEnvironment("results/sae_artifacts", device=args.device, max_test_shards=0)
    per_seed_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for result_path in sorted(root.glob("*/*/*/sae_layer_per_cell.csv")):
        with result_path.open(newline="") as f:
            result = next(csv.DictReader(f))
        run = result_path.relative_to(root).parts[0]
        e_dir = result_path.relative_to(root).parts[1]
        seed_dir = result_path.relative_to(root).parts[2]
        model = result["model"]
        concept = result["concept"]
        task = result["task"]
        layer = int(float(result["layer"]))
        E = int(float(result["E"]))
        seed = int(float(result["sae_seed"]))
        checkpoint = result_path.parent / "checkpoints" / f"E{E}_k0{int(float(result['k0']))}_seed{seed}.pt"
        if not checkpoint.exists():
            skipped_rows.append(
                {
                    "run": run,
                    "E_dir": e_dir,
                    "seed_dir": seed_dir,
                    "concept": concept,
                    "task": task,
                    "E": E,
                    "seed": seed,
                    "reason": f"missing checkpoint: {checkpoint}",
                }
            )
            continue

        if hasattr(env, "set_active_task"):
            env.set_active_task(task)
        train_acts = env.load_activations(model, layer, "train").to(args.device)
        test_acts = env.load_activations(model, layer, "test").to(args.device)
        sae = load_sae(checkpoint, device=args.device)
        with torch.no_grad():
            z_train = sae.encode(sae.normalise(train_acts)).detach().cpu().numpy()
            z_test = sae.encode(sae.normalise(test_acts)).detach().cpu().numpy()
        measurements_train, names = env.load_measurements("train")
        measurements_test, _ = env.load_measurements("test")
        concept_col = env.concept_column(concept)
        concept_idx = names.index(concept_col)
        ranking = select_concept_features(z_train, measurements_train[:, concept_idx], sae.N)

        recovery = {}
        for k in K_GRID:
            kk = min(k, len(ranking))
            recovery[f"decomp_at_{k}"] = decomposability(
                z_train,
                z_test,
                list(ranking[:kk]),
                measurements_train[:, concept_idx],
                measurements_test[:, concept_idx],
            )
        curve_values = [recovery[f"decomp_at_{k}"] for k in K_GRID if np.isfinite(recovery[f"decomp_at_{k}"])]
        recovery_auc = float(np.mean(curve_values)) if curve_values else float("nan")
        per_seed_rows.append(
            {
                "run": run,
                "E_dir": e_dir,
                "seed_dir": seed_dir,
                "concept": concept,
                "task": task,
                "E": E,
                "seed": seed,
                "N_capacity": int(float(result["N_capacity"])),
                "A_geo_cav": row_float(result, "A_geo_cav"),
                "decomp_full": row_float(result, "decomposability_full_concept_ranked"),
                "target_effect": row_float(result, "target_effect"),
                "offtarget_damage": row_float(result, "offtarget_damage"),
                "wbi": row_float(result, "wbi"),
                "random_wbi_mean": row_float(result, "random_wbi_mean"),
                "recovery_auc_grid_mean": recovery_auc,
                **recovery,
            }
        )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    per_seed_csv = out_prefix.with_name(out_prefix.name + "_per_seed.csv")
    if per_seed_rows:
        with per_seed_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_seed_rows)
    skipped_csv = out_prefix.with_name(out_prefix.name + "_skipped.csv")
    if skipped_rows:
        with skipped_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(skipped_rows[0].keys()))
            writer.writeheader()
            writer.writerows(skipped_rows)

    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in per_seed_rows:
        groups.setdefault((str(row["run"]), int(row["E"])), []).append(row)

    summary_rows = []
    for (run, E), rows in sorted(groups.items()):
        item = {
            "run": run,
            "concept": rows[0]["concept"],
            "task": rows[0]["task"],
            "E": E,
            "n_seeds": len(rows),
            "A_geo_cav_mean": mean([float(r["A_geo_cav"]) for r in rows]),
            "A_geo_cav_sd": sd([float(r["A_geo_cav"]) for r in rows]),
            "decomp_full_mean": mean([float(r["decomp_full"]) for r in rows]),
            "decomp_full_sd": sd([float(r["decomp_full"]) for r in rows]),
            "recovery_auc_mean": mean([float(r["recovery_auc_grid_mean"]) for r in rows]),
            "recovery_auc_sd": sd([float(r["recovery_auc_grid_mean"]) for r in rows]),
            "decomp_at_20_mean": mean([float(r["decomp_at_20"]) for r in rows]),
            "decomp_at_20_sd": sd([float(r["decomp_at_20"]) for r in rows]),
            "decomp_at_50_mean": mean([float(r["decomp_at_50"]) for r in rows]),
            "decomp_at_50_sd": sd([float(r["decomp_at_50"]) for r in rows]),
            "decomp_at_100_mean": mean([float(r["decomp_at_100"]) for r in rows]),
            "decomp_at_100_sd": sd([float(r["decomp_at_100"]) for r in rows]),
            "target_effect_median": median([float(r["target_effect"]) for r in rows]),
            "target_effect_iqr_low": q([float(r["target_effect"]) for r in rows], 0.25),
            "target_effect_iqr_high": q([float(r["target_effect"]) for r in rows], 0.75),
            "wbi_median": median([float(r["wbi"]) for r in rows]),
            "wbi_iqr_low": q([float(r["wbi"]) for r in rows], 0.25),
            "wbi_iqr_high": q([float(r["wbi"]) for r in rows], 0.75),
            "random_wbi_mean": mean([float(r["random_wbi_mean"]) for r in rows]),
            "random_wbi_sd": sd([float(r["random_wbi_mean"]) for r in rows]),
        }
        summary_rows.append(item)

    summary_csv = out_prefix.with_suffix(".csv")
    if summary_rows:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    md = out_prefix.with_suffix(".md")
    lines = ["# CSFM SAE Recovery-Curve Summary", ""]
    for E in sorted({int(row["E"]) for row in summary_rows}):
        lines.append(f"## E={E}")
        lines.append("")
        lines.append("| run | concept -> task | A_geo_cav | full decomp | decomp@20 | decomp@50 | decomp@100 | recovery AUC | WBI median [IQR] | target effect median [IQR] |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in [r for r in summary_rows if int(r["E"]) == E]:
            target_iqr = (
                f"{row['target_effect_median']:.3f} "
                f"[{row['target_effect_iqr_low']:.3f}, {row['target_effect_iqr_high']:.3f}]"
            )
            wbi_iqr = f"{row['wbi_median']:.2f} [{row['wbi_iqr_low']:.2f}, {row['wbi_iqr_high']:.2f}]"
            lines.append(
                "| {run} | {concept} -> {task} | {ageo} | {full} | {k20} | {k50} | {k100} | {auc} | {wbi} | {target} |".format(
                    run=row["run"],
                    concept=row["concept"],
                    task=row["task"],
                    ageo=fmt_pm(row["A_geo_cav_mean"], row["A_geo_cav_sd"]),
                    full=fmt_pm(row["decomp_full_mean"], row["decomp_full_sd"]),
                    k20=fmt_pm(row["decomp_at_20_mean"], row["decomp_at_20_sd"]),
                    k50=fmt_pm(row["decomp_at_50_mean"], row["decomp_at_50_sd"]),
                    k100=fmt_pm(row["decomp_at_100_mean"], row["decomp_at_100_sd"]),
                    auc=fmt_pm(row["recovery_auc_mean"], row["recovery_auc_sd"]),
                    wbi=wbi_iqr,
                    target=target_iqr,
                )
            )
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Fixed-k recovery is more stable than threshold-crossing n90 and should be preferred in main text.")
    lines.append("- n90 remains useful as a qualitative appendix diagnostic for tens-of-features recovery.")
    lines.append("- WBI is reported as median/IQR because target-effect denominators can be small, especially for ST/MI.")
    if skipped_rows:
        lines.append(f"- Skipped {len(skipped_rows)} older runs without saved SAE checkpoints; see `{skipped_csv}`.")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_csv)
    print(md)
    print(per_seed_csv)
    if skipped_rows:
        print(skipped_csv)


if __name__ == "__main__":
    main()
