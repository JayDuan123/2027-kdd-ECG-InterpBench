#!/usr/bin/env python
"""Observational SAE monosemanticity taxonomy.

This implements the SAE Monosemanticity Taxonomy spec:
feature-concept enrichment with shuffle p-values + BH-FDR, followed by
dead / monosemantic / entangled / inactive-informative feature labels under
dominance-ratio sensitivity.

This is observational only. It does not run model forward passes or steering.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_v1.sae_extension.topk_sae import TopKSAE  # noqa: E402


MODEL_FEATURE_DIR = {
    "CSFM": "csfm_cu118_commons",
    "CARDIAC-FM": "cardiac_fm_cu118_commons",
    "ECG-FM": "ecg_fm_cu118_commons",
    "ECG-JEPA": "ecg_jepa_cu118_commons",
    "HuBERT-ECG": "hubert_ecg_cu118_commons",
    "ST-MEM": "st_mem_cu118_commons",
}


@dataclass(frozen=True)
class DictionaryRun:
    model: str
    layer: int
    seed: int
    checkpoint: Path
    source: str
    recon_r2: float | None = None
    n_capacity: int | None = None
    k: int | None = None
    cell_id: str | None = None
    matched_tier: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combined-results",
        type=Path,
        default=REPO_ROOT / "results/sae_extension/six_model_sae_audit/l0clamp_reclassified/sae_l0clamp_reclassified_cells.csv",
        help="Steering combined-results CSV with result_csv paths.",
    )
    parser.add_argument(
        "--selected-transformer",
        type=Path,
        default=REPO_ROOT
        / "results/sae_extension/six_model_sae_audit/phase0_selected_transformer_operating_points_l0clamp.csv",
        help="Selected transformer operating points with source_csv paths.",
    )
    parser.add_argument(
        "--seed-fill-root",
        type=Path,
        default=REPO_ROOT / "results/sae_extension/six_model_sae_audit/monosem_seed_fill",
        help="Optional root containing monosemanticity seed-fill sae_recon_curve.csv files.",
    )
    parser.add_argument("--concepts-matrix", type=Path, default=REPO_ROOT / "results/manifest/concepts_matrix.csv")
    parser.add_argument("--concept-config", type=Path, default=REPO_ROOT / "configs/concepts.csv")
    parser.add_argument(
        "--orthogonal-set",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/orthogonal_concepts/locked_orthogonal_set.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/rhf/allocations/wq8/yd68/data/ptb-xl/1.0.3/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/sae_monosemanticity",
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--max-runs-per-model", type=int, default=1)
    parser.add_argument("--include-metadata-ortho", action="store_true")
    parser.add_argument("--seed", type=int, default=4311)
    parser.add_argument("--dead-threshold", type=float, default=1e-6)
    parser.add_argument("--train-cap", type=int, default=0, help="Optional cap for train rows; 0 means all train rows.")
    return parser.parse_args()


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(len(ranked))
    q = ranked * m / (np.arange(len(ranked)) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    tmp = np.empty_like(q)
    tmp[order] = q
    out[valid] = tmp
    return out


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


def abs_corr_matrix(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    zc = z - np.nanmean(z, axis=0, keepdims=True)
    yc = y - np.nanmean(y, axis=0, keepdims=True)
    z_norm = np.sqrt(np.nansum(zc * zc, axis=0))
    y_norm = np.sqrt(np.nansum(yc * yc, axis=0))
    denom = z_norm[:, None] * y_norm[None, :]
    num = zc.T @ yc
    corr = np.divide(num, denom, out=np.zeros_like(num, dtype=float), where=denom > 1e-12)
    return np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))


def standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    norm = np.sqrt(np.nansum(xc * xc, axis=0, keepdims=True))
    return np.divide(xc, norm, out=np.zeros_like(xc, dtype=float), where=norm > 1e-12)


def enrichment_for_set(
    z_train: np.ndarray,
    concept_values: pd.DataFrame,
    concept_names: list[str],
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = concept_values.loc[:, concept_names].to_numpy(dtype=float)
    valid_cols = np.isfinite(y).mean(axis=0) > 0.8
    concept_names[:] = [name for name, keep in zip(concept_names, valid_cols) if keep]
    y = y[:, valid_cols]
    complete = np.isfinite(y).all(axis=1)
    z = z_train[complete]
    y = y[complete]
    z_std = standardize_columns(z)
    y_std = standardize_columns(y)
    assoc = np.abs(z_std.T @ y_std)
    n_features, n_concepts = assoc.shape
    pvals = np.ones_like(assoc)
    for j in range(n_concepts):
        yj = y_std[:, j]
        null = np.zeros((n_permutations, n_features), dtype=float)
        for b in range(n_permutations):
            null[b] = np.abs(z_std.T @ yj[rng.permutation(len(yj))])
        pvals[:, j] = (np.sum(null >= assoc[:, j][None, :], axis=0) + 1.0) / (n_permutations + 1.0)
    qvals = np.ones_like(pvals)
    for j in range(n_concepts):
        qvals[:, j] = bh_fdr(pvals[:, j])
    enriched = qvals < 0.05
    return assoc, qvals, enriched


def classify_features(
    assoc: np.ndarray,
    enriched: np.ndarray,
    firing_rate: np.ndarray,
    ratio: float,
    dead_threshold: float,
) -> list[dict[str, object]]:
    rows = []
    for i in range(assoc.shape[0]):
        dead = bool(firing_rate[i] <= dead_threshold)
        ranked = np.argsort(-assoc[i])
        top = int(ranked[0]) if ranked.size else -1
        second = int(ranked[1]) if ranked.size > 1 else -1
        top_assoc = float(assoc[i, top]) if top >= 0 else 0.0
        second_assoc = float(assoc[i, second]) if second >= 0 else 0.0
        dominance = float(top_assoc / max(second_assoc, 1e-12)) if top >= 0 else 0.0
        n_enriched = int(enriched[i].sum())
        if dead:
            label = "dead"
        elif n_enriched == 0:
            label = "inactive_informative"
        elif n_enriched == 1 and dominance >= ratio:
            label = "monosemantic"
        else:
            label = "entangled"
        rows.append(
            {
                "feature_idx": i,
                "label": label,
                "dead": dead,
                "firing_rate": float(firing_rate[i]),
                "n_enriched": n_enriched,
                "top_concept_idx": top,
                "second_concept_idx": second,
                "top_assoc": top_assoc,
                "second_assoc": second_assoc,
                "dominance": dominance,
            }
        )
    return rows


def taxonomy_fractions(labels: list[str]) -> dict[str, float | int]:
    n = len(labels)
    counts = {key: labels.count(key) for key in ["monosemantic", "entangled", "dead", "inactive_informative"]}
    return {
        "n_features": n,
        "mono_count": counts["monosemantic"],
        "entangled_count": counts["entangled"],
        "dead_count": counts["dead"],
        "inactive_count": counts["inactive_informative"],
        "mono_frac": counts["monosemantic"] / n if n else np.nan,
        "entangled_frac": counts["entangled"] / n if n else np.nan,
        "dead_frac": counts["dead"] / n if n else np.nan,
        "inactive_frac": counts["inactive_informative"] / n if n else np.nan,
    }


def markdown_table(df: pd.DataFrame, cols: list[str]) -> str:
    table = df.loc[:, cols].copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
        else:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in table.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def infer_checkpoint_from_result_csv(result_csv: str) -> Path | None:
    path = REPO_ROOT / result_csv if not result_csv.startswith("/") else Path(result_csv)
    if not path.exists():
        return None
    directory = path.parent / "checkpoints"
    if not directory.exists():
        return None
    pts = sorted(directory.glob("*.pt"))
    return pts[0] if pts else None


def infer_checkpoint_from_source_csv(source_csv: str) -> Path | None:
    path = REPO_ROOT / source_csv if not source_csv.startswith("/") else Path(source_csv)
    directory = path.parent / "checkpoints"
    if not directory.exists():
        return None
    pts = sorted(directory.glob("*.pt"))
    return pts[0] if pts else None


def collect_runs(args: argparse.Namespace) -> list[DictionaryRun]:
    runs: list[DictionaryRun] = []
    seen: set[tuple[str, int, str]] = set()
    if args.combined_results.exists():
        df = pd.read_csv(args.combined_results)
        for _, row in df.iterrows():
            ckpt = infer_checkpoint_from_result_csv(str(row.get("result_csv", "")))
            if ckpt is None:
                continue
            key = (str(row["model"]), int(row["layer"]), str(ckpt))
            if key in seen:
                continue
            seen.add(key)
            runs.append(
                DictionaryRun(
                    model=str(row["model"]),
                    layer=int(row["layer"]),
                    seed=int(row.get("sae_seed", 4311)),
                    checkpoint=ckpt,
                    source="combined_results",
                    recon_r2=float(row["recon_R2"]) if "recon_R2" in row else None,
                    n_capacity=int(row["N_capacity"]) if "N_capacity" in row else None,
                    k=int(row["k"]) if "k" in row else None,
                    cell_id=f"{row.get('concept')}->{row.get('task')}",
                    matched_tier=str(row.get("matched_tier", "")),
                )
            )
    if args.selected_transformer.exists():
        df = pd.read_csv(args.selected_transformer)
        for _, row in df.iterrows():
            ckpt = infer_checkpoint_from_source_csv(str(row.get("source_csv", "")))
            if ckpt is None:
                continue
            key = (str(row["model"]), int(row["layer"]), str(ckpt))
            if key in seen:
                continue
            seen.add(key)
            runs.append(
                DictionaryRun(
                    model=str(row["model"]),
                    layer=int(row["layer"]),
                    seed=4311,
                    checkpoint=ckpt,
                    source="selected_transformer",
                    recon_r2=float(row["recon_R2"]) if "recon_R2" in row else None,
                    n_capacity=int(row["N_capacity"]) if "N_capacity" in row else None,
                    k=int(row["clamp_n_features"]) if "clamp_n_features" in row else None,
                    cell_id=f"{row.get('concept')}->{row.get('task')}",
                    matched_tier="in_band",
                )
            )
    if args.seed_fill_root.exists():
        for csv_path in sorted(args.seed_fill_root.rglob("sae_recon_curve.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            row = df.iloc[0]
            ckpt = infer_checkpoint_from_source_csv(str(csv_path))
            if ckpt is None:
                continue
            key = (str(row["model"]), int(row["layer"]), str(ckpt))
            if key in seen:
                continue
            seen.add(key)
            runs.append(
                DictionaryRun(
                    model=str(row["model"]),
                    layer=int(row["layer"]),
                    seed=int(row.get("sae_seed", 4311)),
                    checkpoint=ckpt,
                    source="seed_fill",
                    recon_r2=float(row["recon_R2"]) if "recon_R2" in row else None,
                    n_capacity=int(row["N_capacity"]) if "N_capacity" in row else None,
                    k=int(row["k"]) if "k" in row else None,
                    cell_id=f"{row.get('concept')}->{row.get('task')}",
                    matched_tier=str(row.get("matched_tier", "")),
                )
            )
    # Keep up to max_runs_per_model to make the first taxonomy pass tractable and
    # avoid over-weighting models with many task-specific checkpoints.
    by_model: dict[str, list[DictionaryRun]] = {}
    for run in runs:
        by_model.setdefault(run.model, []).append(run)
    selected: list[DictionaryRun] = []
    for model, model_runs in sorted(by_model.items()):
        source_priority = {"combined_results": 0, "selected_transformer": 1, "seed_fill": 2}
        model_runs = sorted(model_runs, key=lambda r: (r.seed, r.layer, source_priority.get(r.source, 9), str(r.checkpoint)))
        if args.max_runs_per_model <= 0:
            selected.extend(model_runs)
            continue
        groups: dict[tuple[str | None, int, int | None, int | None], list[DictionaryRun]] = {}
        for run in model_runs:
            groups.setdefault((run.cell_id, run.layer, run.n_capacity, run.k), []).append(run)
        same_config = sorted(
            groups.values(),
            key=lambda rs: (
                -len({r.seed for r in rs}),
                -sum(str(r.matched_tier) == "in_band" for r in rs),
                min(source_priority.get(r.source, 9) for r in rs),
                str(rs[0].cell_id),
            ),
        )[0]
        by_seed: dict[int, DictionaryRun] = {}
        for run in sorted(same_config, key=lambda r: (source_priority.get(r.source, 9), str(r.checkpoint))):
            by_seed.setdefault(run.seed, run)
        seed_selected = [by_seed[seed] for seed in sorted(by_seed)]
        selected.extend(seed_selected[: max(args.max_runs_per_model, len(seed_selected))])
    return selected


def load_concept_sets(args: argparse.Namespace, records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    concepts = pd.read_csv(args.concepts_matrix)
    concepts = records[["ecg_id"]].merge(concepts, on="ecg_id", how="left", validate="one_to_one")
    config = pd.read_csv(args.concept_config)
    full_names = [c for c in config.loc[config["main"].astype(str).str.lower().eq("yes"), "concept_id"] if c in concepts]
    ortho = pd.read_csv(args.orthogonal_set)
    ortho_names = [c for c in ortho["concept_id"].tolist() if c in concepts]
    sets = {
        "FULL": concepts.loc[:, full_names],
        "ORTHO": concepts.loc[:, ortho_names],
    }
    if args.include_metadata_ortho:
        meta = pd.read_csv(args.metadata)
        meta["recording_year"] = pd.to_datetime(meta["recording_date"], errors="coerce").dt.year.astype("float")
        meta["baseline_drift_present"] = meta["baseline_drift"].fillna("").astype(str).str.strip().ne("").astype(float)
        keep = ["ecg_id", "age", "sex", "recording_year", "baseline_drift_present"]
        aligned = records[["ecg_id"]].merge(meta[keep], on="ecg_id", how="left", validate="one_to_one")
        sets["ORTHO_METADATA"] = aligned.drop(columns=["ecg_id"])
    return sets


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args)
    if not runs:
        raise RuntimeError("No SAE dictionary runs found.")

    taxonomy_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    for run in runs:
        if run.model not in MODEL_FEATURE_DIR:
            continue
        feature_dir = REPO_ROOT / "results/probe_features" / MODEL_FEATURE_DIR[run.model]
        records = pd.read_csv(feature_dir / "records.csv")
        acts_path = feature_dir / f"layer_{run.layer:02d}_mean.npy"
        if not acts_path.exists():
            inventory_rows.append({**run.__dict__, "status": "missing_activation"})
            continue
        train_idx = np.where(records["split"].to_numpy() == "train")[0]
        if args.train_cap > 0 and len(train_idx) > args.train_cap:
            train_idx = np.sort(rng.choice(train_idx, size=args.train_cap, replace=False))
        acts = np.load(acts_path, mmap_mode="r")[train_idx]
        sae = load_sae(run.checkpoint)
        z_train = encode_sae(sae, np.asarray(acts))
        firing_rate = (z_train > 0).mean(axis=0)
        concept_sets = load_concept_sets(args, records.iloc[train_idx].reset_index(drop=True))
        inventory_rows.append(
            {
                **run.__dict__,
                "status": "computed",
                "n_train": int(len(train_idx)),
                "n_features": int(z_train.shape[1]),
                "firing_dead_frac": float(np.mean(firing_rate <= args.dead_threshold)),
            }
        )
        print(f"Taxonomy {run.model} layer={run.layer} N={z_train.shape[1]} checkpoint={run.checkpoint}", flush=True)
        for set_name, values in concept_sets.items():
            concept_names = list(values.columns)
            assoc, qvals, enriched = enrichment_for_set(
                z_train=z_train,
                concept_values=values,
                concept_names=concept_names,
                rng=rng,
                n_permutations=args.n_permutations,
            )
            for ratio in args.ratios:
                classifications = classify_features(assoc, enriched, firing_rate, ratio, args.dead_threshold)
                labels = [c["label"] for c in classifications]
                frac = taxonomy_fractions(labels)
                taxonomy_rows.append(
                    {
                        "model": run.model,
                        "layer": run.layer,
                        "cell_id": run.cell_id,
                        "concept_set": set_name,
                        "ratio": ratio,
                        "seed": run.seed,
                        "checkpoint": str(run.checkpoint),
                        "recon_R2": run.recon_r2,
                        "matched_tier": run.matched_tier,
                        **frac,
                    }
                )
                for c in classifications:
                    top_idx = c["top_concept_idx"]
                    second_idx = c["second_concept_idx"]
                    feature_rows.append(
                        {
                            "model": run.model,
                            "layer": run.layer,
                            "cell_id": run.cell_id,
                            "concept_set": set_name,
                            "ratio": ratio,
                            "seed": run.seed,
                            "matched_tier": run.matched_tier,
                            "feature_idx": c["feature_idx"],
                            "label": c["label"],
                            "firing_rate": c["firing_rate"],
                            "n_enriched": c["n_enriched"],
                            "top_concept": concept_names[top_idx] if top_idx >= 0 and top_idx < len(concept_names) else "",
                            "second_concept": concept_names[second_idx]
                            if second_idx >= 0 and second_idx < len(concept_names)
                            else "",
                            "top_assoc": c["top_assoc"],
                            "second_assoc": c["second_assoc"],
                            "dominance": c["dominance"],
                        }
                    )

    computed_models = {str(row.get("model")) for row in inventory_rows if row.get("status") == "computed"}
    for model in MODEL_FEATURE_DIR:
        if model not in computed_models:
            inventory_rows.append(
                {
                    "model": model,
                    "layer": "",
                    "seed": "",
                    "checkpoint": "",
                    "source": "",
                    "recon_r2": "",
                    "n_capacity": "",
                    "k": "",
                    "cell_id": "",
                    "matched_tier": "",
                    "status": "missing_recon_band_checkpoint",
                    "n_train": "",
                    "n_features": "",
                    "firing_dead_frac": "",
                }
            )

    tax_df = pd.DataFrame(taxonomy_rows)
    feat_df = pd.DataFrame(feature_rows)
    inv_df = pd.DataFrame(inventory_rows)
    tax_path = args.out_dir / "monosemanticity_taxonomy.csv"
    feat_path = args.out_dir / "monosemanticity_feature_labels.csv"
    inv_path = args.out_dir / "monosemanticity_run_inventory.csv"
    tax_df.to_csv(tax_path, index=False)
    feat_df.to_csv(feat_path, index=False)
    inv_df.to_csv(inv_path, index=False)

    if not tax_df.empty:
        summary = (
            tax_df.groupby(["model", "concept_set", "ratio"], as_index=False)
            .agg(
                mono_frac_mean=("mono_frac", "mean"),
                mono_frac_sd=("mono_frac", "std"),
                entangled_frac_mean=("entangled_frac", "mean"),
                dead_frac_mean=("dead_frac", "mean"),
                inactive_frac_mean=("inactive_frac", "mean"),
                n_runs=("checkpoint", "nunique"),
                in_band_runs=("matched_tier", lambda x: int(np.sum(pd.Series(x).astype(str).eq("in_band")))),
            )
            .fillna({"mono_frac_sd": 0.0})
        )
        # Contamination is defined only where FULL and ORTHO both exist.
        pivot = summary.pivot_table(index=["model", "ratio"], columns="concept_set", values="mono_frac_mean")
        contamination = []
        for (model, ratio), row in pivot.iterrows():
            if "ORTHO" in row and "FULL" in row and np.isfinite(row.get("ORTHO", np.nan)) and np.isfinite(row.get("FULL", np.nan)):
                contamination.append(
                    {
                        "model": model,
                        "ratio": ratio,
                        "collinearity_contamination": float(row["ORTHO"] - row["FULL"]),
                    }
                )
        cont = pd.DataFrame(contamination)
        if not cont.empty:
            summary = summary.merge(cont, on=["model", "ratio"], how="left")
        else:
            summary["collinearity_contamination"] = np.nan
    else:
        summary = pd.DataFrame()
    summary_path = args.out_dir / "monosemanticity_summary.csv"
    summary.to_csv(summary_path, index=False)

    report_path = args.out_dir / "monosemanticity_taxonomy_report.md"
    lines = [
        "# SAE Monosemanticity Taxonomy",
        "",
        "Observational taxonomy only: no steering/intervention. Enrichment uses absolute feature-concept correlation on the train split, shuffle p-values, and BH-FDR q<0.05 over SAE features per concept.",
        "",
        f"- ratios: {args.ratios}",
        f"- permutations: {args.n_permutations}",
        f"- runs computed: {len(tax_df[['model','layer','checkpoint']].drop_duplicates()) if not tax_df.empty else 0}",
        "",
        "## Outputs",
        "",
        f"- taxonomy: `{tax_path}`",
        f"- summary: `{summary_path}`",
        f"- feature labels: `{feat_path}`",
        f"- run inventory: `{inv_path}`",
        "",
    ]
    if not summary.empty:
        r3 = summary[summary["ratio"].astype(float).eq(3.0)].copy()
        lines.extend(["## Ratio 3 Summary", ""])
        cols = ["model", "concept_set", "mono_frac_mean", "mono_frac_sd", "entangled_frac_mean", "dead_frac_mean", "inactive_frac_mean", "n_runs", "in_band_runs", "collinearity_contamination"]
        lines.append(markdown_table(r3, [c for c in cols if c in r3.columns]))
        lines.append("")
    missing = inv_df[inv_df["status"].astype(str).ne("computed")] if not inv_df.empty else pd.DataFrame()
    if not missing.empty:
        lines.extend(["## Missing / Non-primary Models", ""])
        lines.append(markdown_table(missing, [c for c in ["model", "status", "source", "recon_r2", "checkpoint"] if c in missing.columns]))
        lines.append("")
    report_path.write_text("\n".join(lines))
    print(f"Wrote {tax_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {feat_path}")
    print(f"Wrote {inv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
