#!/usr/bin/env python
"""CSFM top-atom vs full-probe audit for SAE concept representation.

This is an observational held-out validation:
  * train split selects the top SAE atom(s) for each concept;
  * test split reports the selected atom/top-k probe metric;
  * full SAE-code and raw-activation probes are reported as readout ceilings.

Binary concepts use direction-invariant AUROC for single atoms. Continuous
concepts use ridge R2 for probes and train single-feature R2 for atom ranking.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_v1.sae_extension.topk_sae import TopKSAE  # noqa: E402
from scripts.run_sae_monosemanticity_taxonomy import collect_runs, load_concept_sets  # noqa: E402


@dataclass(frozen=True)
class ConceptInfo:
    name: str
    concept_set: str
    kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="CSFM")
    parser.add_argument("--feature-dir", type=Path, default=REPO_ROOT / "results/probe_features/csfm_cu118_commons")
    parser.add_argument(
        "--taxonomy-dir",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/sae_monosemanticity",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/top_atom_probe",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--topks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--representative-gap", type=float, default=0.05)
    parser.add_argument("--include-metadata-ortho", action="store_true", default=True)
    parser.add_argument("--max-runs-per-model", type=int, default=1)
    parser.add_argument(
        "--compute-raw-probe",
        action="store_true",
        help="Also fit full raw-activation probes. This is slower and is not needed for the primary SAE-code vs top-atom audit.",
    )
    parser.add_argument(
        "--metadata-step0-csv",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/metadata_controls/metadata_step0_sae_gate_csfm_l5.csv",
        help="Existing CSFM metadata Step0 raw-activation probe metrics to merge into ORTHO_METADATA rows.",
    )
    return parser.parse_args()


def load_sae(checkpoint: Path) -> TopKSAE:
    ckpt = torch.load(checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    sae = TopKSAE(d=int(meta["d"]), n_features=int(meta["n_features"]), k=int(meta["k"]))
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    sae.track_dead_features = False
    return sae


def encode_sae(sae: TopKSAE, acts: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(acts), batch_size):
            batch = torch.as_tensor(acts[start : start + batch_size], dtype=torch.float32)
            chunks.append(sae.encode(sae.normalise(batch)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.nanmean(x_train, axis=0, keepdims=True)
    sd = np.nanstd(x_train, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return np.nan_to_num((x_train - mu) / sd), np.nan_to_num((x_test - mu) / sd)


def infer_kind(y: np.ndarray) -> str:
    finite = y[np.isfinite(y)]
    unique = np.unique(finite)
    if unique.size <= 2 and np.all(np.isin(unique, [0, 1])):
        return "binary"
    return "continuous"


def valid_mask(y: np.ndarray, kind: str) -> np.ndarray:
    mask = np.isfinite(y)
    if kind == "binary":
        mask &= np.isin(y, [0, 1])
    return mask


def directionless_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y.astype(int))) < 2:
        return np.nan
    auc = float(roc_auc_score(y.astype(int), score))
    return max(auc, 1.0 - auc)


def fit_probe_metric(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    alpha: float,
) -> float:
    x_train, x_test = standardize_train_test(x_train, x_test)
    if kind == "binary":
        if len(np.unique(y_train.astype(int))) < 2 or len(np.unique(y_test.astype(int))) < 2:
            return np.nan
        model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, y_train.astype(float))
        score = model.predict(x_test)
        return float(roc_auc_score(y_test.astype(int), score))
    y_mu = float(np.nanmean(y_train))
    y_sd = float(np.nanstd(y_train))
    if y_sd < 1e-8:
        return np.nan
    y_train_z = (y_train - y_mu) / y_sd
    y_test_z = (y_test - y_mu) / y_sd
    model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, y_train_z)
    score = model.predict(x_test)
    return float(r2_score(y_test_z, score))


def single_feature_train_scores(z_train: np.ndarray, y_train: np.ndarray, kind: str) -> np.ndarray:
    scores = np.zeros(z_train.shape[1], dtype=float)
    if kind == "binary":
        for j in range(z_train.shape[1]):
            scores[j] = directionless_auc(y_train, z_train[:, j])
        return np.nan_to_num(scores, nan=0.5)
    for j in range(z_train.shape[1]):
        metric = fit_probe_metric(z_train[:, [j]], y_train, z_train[:, [j]], y_train, kind, alpha=1.0)
        scores[j] = metric
    return np.nan_to_num(scores, nan=-np.inf)


def single_atom_test_metric(z_test_feature: np.ndarray, y_test: np.ndarray, kind: str) -> float:
    if kind == "binary":
        return directionless_auc(y_test, z_test_feature)
    # For continuous concepts, fit-free single-atom effect size is squared Pearson
    # on held-out data, direction-invariant and easy to compare with R2.
    if np.nanstd(z_test_feature) < 1e-8 or np.nanstd(y_test) < 1e-8:
        return np.nan
    corr = np.corrcoef(z_test_feature, y_test)[0, 1]
    return float(corr * corr)


def load_taxonomy_labels(taxonomy_dir: Path) -> pd.DataFrame:
    path = taxonomy_dir / "monosemanticity_feature_labels.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def monosemantic_features_for(
    labels: pd.DataFrame,
    model: str,
    seed: int,
    concept_set: str,
    concept: str,
    ratio: float = 3.0,
) -> list[int]:
    if labels.empty:
        return []
    rows = labels[
        labels["model"].astype(str).eq(model)
        & labels["seed"].astype(int).eq(int(seed))
        & labels["concept_set"].astype(str).eq(concept_set)
        & labels["ratio"].astype(float).eq(float(ratio))
        & labels["label"].astype(str).eq("monosemantic")
        & labels["top_concept"].astype(str).eq(concept)
    ]
    return sorted(rows["feature_idx"].astype(int).tolist())


def feature_label(
    labels: pd.DataFrame,
    model: str,
    seed: int,
    concept_set: str,
    feature_idx: int,
    ratio: float = 3.0,
) -> str:
    if labels.empty:
        return ""
    rows = labels[
        labels["model"].astype(str).eq(model)
        & labels["seed"].astype(int).eq(int(seed))
        & labels["concept_set"].astype(str).eq(concept_set)
        & labels["ratio"].astype(float).eq(float(ratio))
        & labels["feature_idx"].astype(int).eq(int(feature_idx))
    ]
    if rows.empty:
        return ""
    return str(rows.iloc[0]["label"])


def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    table = df.loc[:, cols].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in table.to_numpy(dtype=str)]
    return "\n".join([header, sep, *body])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the taxonomy run collector so we audit the same CSFM recon-band SAE
    # configuration and seed-fill checkpoints.
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
    labels = load_taxonomy_labels(args.taxonomy_dir)
    metadata_raw_metric: dict[str, float] = {}
    if args.metadata_step0_csv.exists():
        step0 = pd.read_csv(args.metadata_step0_csv)
        metadata_raw_metric = {
            str(row["concept"]): float(row["probe_metric"])
            for _, row in step0.iterrows()
            if pd.notna(row.get("probe_metric"))
        }

    rows: list[dict[str, object]] = []
    for run in runs:
        acts = np.load(args.feature_dir / f"layer_{run.layer:02d}_mean.npy", mmap_mode="r")
        x_train_raw = np.asarray(acts[train_idx])
        x_test_raw = np.asarray(acts[test_idx])
        sae = load_sae(run.checkpoint)
        z_train = encode_sae(sae, x_train_raw)
        z_test = encode_sae(sae, x_test_raw)

        concept_sets = load_concept_sets(argparse.Namespace(
            concepts_matrix=REPO_ROOT / "results/manifest/concepts_matrix.csv",
            concept_config=REPO_ROOT / "configs/concepts.csv",
            orthogonal_set=REPO_ROOT / "results/analysis/model_comparison/orthogonal_concepts/locked_orthogonal_set.csv",
            include_metadata_ortho=args.include_metadata_ortho,
            metadata=Path("/rhf/allocations/wq8/yd68/data/ptb-xl/1.0.3/ptbxl_database.csv"),
        ), records)

        print(f"Top-atom audit {run.model} seed={run.seed} layer={run.layer} N={z_train.shape[1]}", flush=True)
        for set_name, values in concept_sets.items():
            for concept in values.columns:
                y_all = values[concept].to_numpy(dtype=float)
                kind = infer_kind(y_all)
                tr_mask = valid_mask(y_all[train_idx], kind)
                te_mask = valid_mask(y_all[test_idx], kind)
                y_train = y_all[train_idx][tr_mask]
                y_test = y_all[test_idx][te_mask]
                if len(y_train) < 20 or len(y_test) < 20:
                    continue
                if kind == "binary" and (len(np.unique(y_train.astype(int))) < 2 or len(np.unique(y_test.astype(int))) < 2):
                    continue

                ztr = z_train[tr_mask]
                zte = z_test[te_mask]
                xtr = x_train_raw[tr_mask]
                xte = x_test_raw[te_mask]

                rank_scores = single_feature_train_scores(ztr, y_train, kind)
                ranking = np.argsort(-rank_scores)
                top1 = int(ranking[0])
                top1_metric = single_atom_test_metric(zte[:, top1], y_test, kind)
                full_sae_metric = fit_probe_metric(ztr, y_train, zte, y_test, kind, args.ridge_alpha)
                if args.compute_raw_probe:
                    full_raw_metric = fit_probe_metric(xtr, y_train, xte, y_test, kind, args.ridge_alpha)
                    raw_source = "computed_raw_probe"
                elif set_name == "ORTHO_METADATA" and concept in metadata_raw_metric:
                    full_raw_metric = metadata_raw_metric[concept]
                    raw_source = "metadata_step0"
                else:
                    full_raw_metric = np.nan
                    raw_source = ""
                metric_name = "auroc" if kind == "binary" else "r2"

                out: dict[str, object] = {
                    "model": run.model,
                    "seed": run.seed,
                    "layer": run.layer,
                    "cell_id": run.cell_id,
                    "checkpoint": str(run.checkpoint),
                    "matched_tier": run.matched_tier,
                    "concept": concept,
                    "concept_set": set_name,
                    "type": kind,
                    "metric": metric_name,
                    "n_train": int(len(y_train)),
                    "n_test": int(len(y_test)),
                    "n_features": int(ztr.shape[1]),
                    "full_sae_probe_metric": full_sae_metric,
                    "full_raw_probe_metric": full_raw_metric,
                    "full_raw_probe_source": raw_source,
                    "top1_atom_metric": top1_metric,
                    "gap_full_sae_minus_top1": full_sae_metric - top1_metric,
                    "gap_full_raw_minus_top1": full_raw_metric - top1_metric if np.isfinite(full_raw_metric) else np.nan,
                    "top1_feature_idx": top1,
                    "top1_train_select_metric": float(rank_scores[top1]),
                    "top1_taxonomy_label_ratio3": feature_label(labels, run.model, run.seed, set_name, top1),
                    "mono_features_for_concept_ratio3": ";".join(
                        map(str, monosemantic_features_for(labels, run.model, run.seed, set_name, concept))
                    ),
                }
                for k in args.topks:
                    k_eff = min(int(k), ztr.shape[1])
                    selected = ranking[:k_eff]
                    metric = fit_probe_metric(ztr[:, selected], y_train, zte[:, selected], y_test, kind, args.ridge_alpha)
                    out[f"top{k}_probe_metric"] = metric
                    out[f"top{k}_k_actual"] = k_eff
                    out[f"gap_full_sae_minus_top{k}"] = full_sae_metric - metric
                out["has_representative_atom"] = bool(top1_metric >= full_sae_metric - args.representative_gap)
                out["top10_close_to_full"] = bool(out.get("top10_probe_metric", np.nan) >= full_sae_metric - args.representative_gap)
                rows.append(out)

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / "csfm_top_atom_vs_probe.csv"
    df.to_csv(out_csv, index=False)

    summary = (
        df.groupby(["concept_set", "type"], as_index=False)
        .agg(
            n_concepts=("concept", "nunique"),
            full_sae_mean=("full_sae_probe_metric", "mean"),
            top1_mean=("top1_atom_metric", "mean"),
            top10_mean=("top10_probe_metric", "mean"),
            gap_top1_mean=("gap_full_sae_minus_top1", "mean"),
            representative_rate=("has_representative_atom", "mean"),
            top10_close_rate=("top10_close_to_full", "mean"),
        )
    )
    out_summary = args.out_dir / "csfm_top_atom_vs_probe_summary.csv"
    summary.to_csv(out_summary, index=False)

    focus = df[df["concept"].isin(["sex", "age", "recording_year", "baseline_drift_present"])].copy()
    clinical_focus_names = [
        "qrs_duration",
        "qrst_angle",
        "r_amp_global",
        "st_amp_global",
        "hr_atrial",
        "rr_mean",
        "p_found",
    ]
    clinical_focus = df[df["concept"].isin(clinical_focus_names)].copy()
    md = [
        "# CSFM Top-Atom vs Full-Probe Audit",
        "",
        "Held-out test audit. The top atom is selected on train only. Binary single-atom AUROC is direction-invariant: max(AUROC, 1-AUROC). Continuous concepts use R2/correlation-style scores.",
        "",
        f"- CSV: `{out_csv}`",
        f"- summary: `{out_summary}`",
        "",
        "## Concept-Set Summary",
        "",
        markdown_table(summary, list(summary.columns)),
        "",
        "## Metadata Focus",
        "",
        markdown_table(
            focus.sort_values(["concept", "seed"]),
            [
                "seed",
                "concept",
                "type",
                "full_sae_probe_metric",
                "full_raw_probe_metric",
                "top1_atom_metric",
                "top5_probe_metric",
                "top10_probe_metric",
                "gap_full_sae_minus_top1",
                "has_representative_atom",
                "top1_feature_idx",
                "top1_taxonomy_label_ratio3",
                "mono_features_for_concept_ratio3",
            ],
        ),
        "",
        "## Clinical Focus",
        "",
        markdown_table(
            clinical_focus.sort_values(["concept", "seed"]),
            [
                "seed",
                "concept",
                "concept_set",
                "type",
                "full_sae_probe_metric",
                "full_raw_probe_metric",
                "top1_atom_metric",
                "top5_probe_metric",
                "top10_probe_metric",
                "gap_full_sae_minus_top1",
                "has_representative_atom",
                "top1_feature_idx",
                "top1_taxonomy_label_ratio3",
            ],
            max_rows=80,
        ),
        "",
    ]
    out_md = args.out_dir / "csfm_top_atom_vs_probe_report.md"
    out_md.write_text("\n".join(md))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
