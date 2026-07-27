#!/usr/bin/env python
"""High-vs-low AUROC audit for CSFM waveform measurement concepts.

Continuous waveform measurements (HR/QT/ST/QRS) are converted to binary labels
using the train-split median for that concept. The audit then follows the same
top-atom protocol:
  * train selects the top SAE atom(s);
  * test reports direction-invariant top1 AUROC and top-k/full SAE AUROC.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_csfm_top_atom_probe_audit import (  # noqa: E402
    directionless_auc,
    encode_sae,
    fit_probe_metric,
    load_sae,
    markdown_table,
    single_feature_train_scores,
)
from scripts.run_sae_monosemanticity_taxonomy import collect_runs, load_concept_sets  # noqa: E402


WAVEFORM_RE = r"^(hr_|rr_|qt|qtc|st_|qrs|qrst)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="CSFM")
    parser.add_argument("--feature-dir", type=Path, default=REPO_ROOT / "results/probe_features/csfm_cu118_commons")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/top_atom_probe",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--representative-gap", type=float, default=0.05)
    parser.add_argument("--concept-regex", default=WAVEFORM_RE)
    parser.add_argument("--max-runs-per-model", type=int, default=1)
    return parser.parse_args()


def median_binarize(y_train_raw: np.ndarray, y_test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    threshold = float(np.nanmedian(y_train_raw))
    y_train = (y_train_raw >= threshold).astype(float)
    y_test = (y_test_raw >= threshold).astype(float)
    return y_train, y_test, threshold


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tax_args = argparse.Namespace(
        combined_results=REPO_ROOT / "results/sae_extension/six_model_sae_audit/l0clamp_reclassified/sae_l0clamp_reclassified_cells.csv",
        selected_transformer=REPO_ROOT
        / "results/sae_extension/six_model_sae_audit/phase0_selected_transformer_operating_points_l0clamp.csv",
        seed_fill_root=REPO_ROOT / "results/sae_extension/six_model_sae_audit/monosem_seed_fill",
        max_runs_per_model=args.max_runs_per_model,
    )
    runs = [run for run in collect_runs(tax_args) if run.model == args.model]
    if not runs:
        raise RuntimeError(f"No SAE runs found for model={args.model}")

    records = pd.read_csv(args.feature_dir / "records.csv")
    train_idx = np.where(records["split"].to_numpy() == "train")[0]
    test_idx = np.where(records["split"].to_numpy() == "test")[0]
    concept_sets = load_concept_sets(
        argparse.Namespace(
            concepts_matrix=REPO_ROOT / "results/manifest/concepts_matrix.csv",
            concept_config=REPO_ROOT / "configs/concepts.csv",
            orthogonal_set=REPO_ROOT / "results/analysis/model_comparison/orthogonal_concepts/locked_orthogonal_set.csv",
            include_metadata_ortho=False,
            metadata=Path("/rhf/allocations/wq8/yd68/data/ptb-xl/1.0.3/ptbxl_database.csv"),
        ),
        records,
    )
    full = concept_sets["FULL"]
    concepts = [c for c in full.columns if pd.Series([c]).str.contains(args.concept_regex, regex=True).iloc[0]]
    rows: list[dict[str, object]] = []

    for run in runs:
        acts = np.load(args.feature_dir / f"layer_{run.layer:02d}_mean.npy", mmap_mode="r")
        x_train_raw = np.asarray(acts[train_idx])
        x_test_raw = np.asarray(acts[test_idx])
        sae = load_sae(run.checkpoint)
        z_train = encode_sae(sae, x_train_raw)
        z_test = encode_sae(sae, x_test_raw)
        print(f"Waveform AUROC audit {run.model} seed={run.seed} layer={run.layer} N={z_train.shape[1]}", flush=True)

        for concept in concepts:
            y_all = full[concept].to_numpy(dtype=float)
            tr_valid = np.isfinite(y_all[train_idx])
            te_valid = np.isfinite(y_all[test_idx])
            if tr_valid.sum() < 20 or te_valid.sum() < 20:
                continue
            y_train_raw = y_all[train_idx][tr_valid]
            y_test_raw = y_all[test_idx][te_valid]
            y_train, y_test, threshold = median_binarize(y_train_raw, y_test_raw)
            if len(np.unique(y_train.astype(int))) < 2 or len(np.unique(y_test.astype(int))) < 2:
                continue
            ztr = z_train[tr_valid]
            zte = z_test[te_valid]
            rank_scores = single_feature_train_scores(ztr, y_train, "binary")
            ranking = np.argsort(-rank_scores)
            top1 = int(ranking[0])
            top1_metric = directionless_auc(y_test, zte[:, top1])
            full_metric = fit_probe_metric(ztr, y_train, zte, y_test, "binary", args.ridge_alpha)
            out: dict[str, object] = {
                "model": run.model,
                "seed": run.seed,
                "layer": run.layer,
                "cell_id": run.cell_id,
                "concept": concept,
                "threshold_train_median": threshold,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "n_features": int(ztr.shape[1]),
                "full_sae_probe_auroc": full_metric,
                "top1_atom_auroc": top1_metric,
                "gap_full_minus_top1": full_metric - top1_metric,
                "top1_feature_idx": top1,
                "top1_train_select_auroc": float(rank_scores[top1]),
            }
            for k in args.topks:
                k_eff = min(int(k), ztr.shape[1])
                selected = ranking[:k_eff]
                metric = fit_probe_metric(ztr[:, selected], y_train, zte[:, selected], y_test, "binary", args.ridge_alpha)
                out[f"top{k}_probe_auroc"] = metric
                out[f"top{k}_k_actual"] = k_eff
                out[f"gap_full_minus_top{k}"] = full_metric - metric
            out["has_representative_atom"] = bool(top1_metric >= full_metric - args.representative_gap)
            out["top10_close_to_full"] = bool(out.get("top10_probe_auroc", np.nan) >= full_metric - args.representative_gap)
            rows.append(out)

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "csfm_waveform_concept_highlow_auroc.csv"
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby("concept", as_index=False)
        .agg(
            full_sae_probe_auroc_mean=("full_sae_probe_auroc", "mean"),
            top1_atom_auroc_mean=("top1_atom_auroc", "mean"),
            top5_probe_auroc_mean=("top5_probe_auroc", "mean"),
            top10_probe_auroc_mean=("top10_probe_auroc", "mean"),
            gap_full_minus_top1_mean=("gap_full_minus_top1", "mean"),
            representative_rate=("has_representative_atom", "mean"),
        )
        .sort_values("full_sae_probe_auroc_mean", ascending=False)
    )
    out_summary = args.out_dir / "csfm_waveform_concept_highlow_auroc_summary.csv"
    summary.to_csv(out_summary, index=False)

    families = {
        "HR/RR": ["hr_atrial", "hr_ventricular", "rr_mean", "rr_iqr"],
        "QT/QTc": ["qt_interval", "qtc_bazett", "qtc_framingham", "qtc_fridericia"],
        "ST": [
            "st_amp_global",
            "st_amp_anterior",
            "st_amp_inferior",
            "st_amp_lateral",
            "st_elev_global",
            "st_elev_anterior",
            "st_elev_inferior",
            "st_elev_lateral",
        ],
        "QRS/Axis": ["qrs_duration", "qrs_area_global", "qrs_axis_front", "qrs_balance_global", "qrst_angle"],
    }
    md = [
        "# CSFM Waveform Measurement High-vs-Low AUROC Audit",
        "",
        "Continuous waveform concepts are binarized at the train-split median, then evaluated with the same train-select/test-evaluate top-atom protocol.",
        "",
        f"- CSV: `{out_csv}`",
        f"- summary: `{out_summary}`",
        "",
        "## Overall Summary",
        "",
        markdown_table(
            summary,
            [
                "concept",
                "full_sae_probe_auroc_mean",
                "top1_atom_auroc_mean",
                "top5_probe_auroc_mean",
                "top10_probe_auroc_mean",
                "gap_full_minus_top1_mean",
                "representative_rate",
            ],
            max_rows=40,
        ),
    ]
    for family, names in families.items():
        sub = summary[summary["concept"].isin(names)].copy()
        if sub.empty:
            continue
        md.extend(["", f"## {family}", ""])
        md.append(
            markdown_table(
                sub,
                [
                    "concept",
                    "full_sae_probe_auroc_mean",
                    "top1_atom_auroc_mean",
                    "top5_probe_auroc_mean",
                    "top10_probe_auroc_mean",
                    "gap_full_minus_top1_mean",
                    "representative_rate",
                ],
            )
        )
    out_md = args.out_dir / "csfm_waveform_concept_highlow_auroc_report.md"
    out_md.write_text("\n".join(md) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
