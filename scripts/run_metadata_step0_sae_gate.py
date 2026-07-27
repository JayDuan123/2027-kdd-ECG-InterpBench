#!/usr/bin/env python
"""Step0 metadata probe + SAE dictionary gate for positive-control route A.

This is deliberately not downstream steering. It asks whether PTB-XL metadata
concepts are (1) encoded in CSFM activations and (2) isolated enough in an SAE
dictionary to justify a later positive-control steering experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_v1.sae_extension.metrics import (  # noqa: E402
    geometric_agreement,
    select_concept_features,
    select_target_features,
    wbi,
)
from benchmark_v1.sae_extension.topk_sae import TopKSAE  # noqa: E402


@dataclass(frozen=True)
class MetadataConcept:
    name: str
    kind: str  # "continuous" or "binary"


CONCEPTS = [
    MetadataConcept("age", "continuous"),
    MetadataConcept("sex", "binary"),
    MetadataConcept("recording_year", "continuous"),
    MetadataConcept("baseline_drift_present", "binary"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=REPO_ROOT / "results/probe_features/csfm_cu118_commons")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "results/sae_extension/six_model_sae_audit/csfm_steering_main_l0clamp/cell_3/checkpoints/N8_k01_seed4311.pt",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/rhf/allocations/wq8/yd68/data/ptb-xl/1.0.3/ptbxl_database.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results/analysis/model_comparison/metadata_controls",
    )
    parser.add_argument("--shuffle-iters", type=int, default=20)
    parser.add_argument("--random-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4311)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--gate-feature-count", type=int, default=1)
    parser.add_argument("--probe-train-cap", type=int, default=6000)
    return parser.parse_args()


def load_sae(checkpoint: Path) -> TopKSAE:
    ckpt = torch.load(checkpoint, map_location="cpu")
    meta = ckpt["meta"]
    sae = TopKSAE(d=int(meta["d"]), n_features=int(meta["n_features"]), k=int(meta["k"]))
    sae.load_state_dict(ckpt["sae"])
    sae.eval()
    sae.track_dead_features = False
    return sae


def metadata_matrix(metadata_path: Path, ecg_ids: np.ndarray) -> pd.DataFrame:
    meta = pd.read_csv(metadata_path)
    meta["recording_year"] = pd.to_datetime(meta["recording_date"], errors="coerce").dt.year.astype("float")
    for col in ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems", "extra_beats"]:
        meta[f"{col}_present"] = meta[col].fillna("").astype(str).str.strip().ne("").astype(float)
    keep = ["ecg_id", "patient_id", "age", "sex", "recording_year", "baseline_drift_present"]
    aligned = pd.DataFrame({"ecg_id": ecg_ids}).merge(meta[keep], on="ecg_id", how="left", validate="one_to_one")
    return aligned


def encode_sae(sae: TopKSAE, acts: np.ndarray, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    z_chunks: list[np.ndarray] = []
    norm_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(acts), batch_size):
            a = torch.as_tensor(acts[start : start + batch_size], dtype=torch.float32)
            a_norm = sae.normalise(a)
            z = sae.encode(a_norm)
            z_chunks.append(z.cpu().numpy())
            norm_chunks.append(a_norm.cpu().numpy())
    return np.concatenate(z_chunks, axis=0), np.concatenate(norm_chunks, axis=0)


def decode_sae_norm(sae: TopKSAE, z: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(z), batch_size):
            zt = torch.as_tensor(z[start : start + batch_size], dtype=torch.float32)
            chunks.append(sae.decode(zt).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def valid_mask(values: np.ndarray, kind: str) -> np.ndarray:
    mask = np.isfinite(values)
    if kind == "binary":
        mask &= np.isin(values, [0, 1])
    return mask


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    alpha: float,
) -> dict[str, object]:
    if kind == "continuous":
        y_mu = float(np.nanmean(y_train))
        y_sd = float(np.nanstd(y_train))
        if y_sd < 1e-12:
            raise ValueError("continuous target has near-zero train variance")
        y_train_fit = (y_train - y_mu) / y_sd
        y_test_eval = (y_test - y_mu) / y_sd
        model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, y_train_fit)
        score = model.predict(x_test)
        metric = float(r2_score(y_test_eval, score))
        metric_name = "r2"
        baseline = 0.0
    else:
        if len(set(y_train.astype(int).tolist())) < 2 or len(set(y_test.astype(int).tolist())) < 2:
            raise ValueError("binary target needs two classes in train and test")
        y_train_fit = y_train.astype(float)
        model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, y_train_fit)
        score = model.predict(x_test)
        metric = float(roc_auc_score(y_test.astype(int), score))
        metric_name = "auroc"
        baseline = 0.5
    coef = np.asarray(model.coef_, dtype=float).reshape(-1)
    coef_norm = float(np.linalg.norm(coef))
    if coef_norm > 0:
        coef = coef / coef_norm
    return {
        "model": model,
        "score": score,
        "metric": metric,
        "metric_name": metric_name,
        "baseline_metric": baseline,
        "coef_unit": coef,
    }


def shuffled_probe_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    kind: str,
    alpha: float,
    rng: np.random.Generator,
    n_iter: int,
) -> np.ndarray:
    out = []
    for _ in range(n_iter):
        ys = np.array(y_train, copy=True)
        rng.shuffle(ys)
        try:
            out.append(fit_linear_probe(x_train, ys, x_test, y_test, kind, alpha)["metric"])
        except ValueError:
            continue
    return np.asarray(out, dtype=float)


def cap_train_rows(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(np.arange(n), size=cap, replace=False))


def metric_from_scores(kind: str, y: np.ndarray, scores: np.ndarray, train_mu: float | None = None, train_sd: float | None = None) -> float:
    if kind == "continuous":
        if train_mu is None or train_sd is None or train_sd < 1e-12:
            return float("nan")
        yz = (y - train_mu) / train_sd
        return float(r2_score(yz, scores))
    if len(set(y.astype(int).tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y.astype(int), scores))


def selectivity_gate(
    sae: TopKSAE,
    z_train: np.ndarray,
    z_test: np.ndarray,
    h_test_norm: np.ndarray,
    metadata: dict[str, dict[str, object]],
    target: str,
    selected: np.ndarray,
    rng: np.random.Generator,
    n_random: int,
) -> dict[str, object]:
    centroid = z_train.mean(axis=0)
    selected = np.asarray(selected, dtype=int)

    def patched_scores(feature_idx: np.ndarray) -> dict[str, np.ndarray]:
        zp = np.array(z_test, copy=True)
        if len(feature_idx):
            zp[:, feature_idx] = centroid[feature_idx][None, :]
        h_patch_norm = decode_sae_norm(sae, zp)
        scores = {}
        for name, info in metadata.items():
            scores[name] = info["model"].predict(h_patch_norm)
        return scores

    # Match the SAE steering convention: f=0 is the SAE reconstruction without
    # a feature clamp, not the original dense activation.
    clean_scores = patched_scores(np.asarray([], dtype=int))
    clean_metrics = {}
    for name, info in metadata.items():
        clean_metrics[name] = metric_from_scores(
            info["kind"],
            info["y_test"],
            clean_scores[name],
            info.get("train_mu"),
            info.get("train_sd"),
        )

    concept_scores = patched_scores(selected)
    concept_metrics = {}
    for name, info in metadata.items():
        concept_metrics[name] = metric_from_scores(
            info["kind"],
            info["y_test"],
            concept_scores[name],
            info.get("train_mu"),
            info.get("train_sd"),
        )

    target_effect = clean_metrics[target] - concept_metrics[target]
    off_damages = []
    for name in metadata:
        if name == target:
            continue
        damage = clean_metrics[name] - concept_metrics[name]
        if np.isfinite(damage):
            off_damages.append(max(0.0, float(damage)))
    off_damage = float(np.mean(off_damages)) if off_damages else float("nan")
    concept_wbi = wbi(float(target_effect), float(off_damage))

    all_features = np.arange(z_train.shape[1])
    random_effects = []
    random_off = []
    random_wbis = []
    for _ in range(n_random):
        choice = rng.choice(all_features, size=len(selected), replace=False)
        rscores = patched_scores(choice)
        rmetrics = {}
        for name, info in metadata.items():
            rmetrics[name] = metric_from_scores(
                info["kind"],
                info["y_test"],
                rscores[name],
                info.get("train_mu"),
                info.get("train_sd"),
            )
        reffect = clean_metrics[target] - rmetrics[target]
        roff_values = []
        for name in metadata:
            if name == target:
                continue
            damage = clean_metrics[name] - rmetrics[name]
            if np.isfinite(damage):
                roff_values.append(max(0.0, float(damage)))
        roff = float(np.mean(roff_values)) if roff_values else float("nan")
        random_effects.append(float(reffect))
        random_off.append(roff)
        random_wbis.append(wbi(float(reffect), roff))

    return {
        "target_effect": float(target_effect),
        "offtarget_damage": float(off_damage),
        "wbi": float(concept_wbi),
        "random_target_effect_mean": float(np.nanmean(random_effects)),
        "random_offtarget_damage_mean": float(np.nanmean(random_off)),
        "random_wbi_median": float(np.nanmedian(random_wbis)),
        "random_wbi_iqr_low": float(np.nanpercentile(random_wbis, 25)),
        "random_wbi_iqr_high": float(np.nanpercentile(random_wbis, 75)),
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


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = pd.read_csv(args.feature_dir / "records.csv")
    acts = np.load(args.feature_dir / f"layer_{args.layer:02d}_mean.npy")
    meta = metadata_matrix(args.metadata, records["ecg_id"].to_numpy())
    sae = load_sae(args.checkpoint)
    print(f"Loaded activation {acts.shape}, metadata {meta.shape}, SAE N={sae.N} k={sae.k}", flush=True)

    train_idx = np.where(records["split"].to_numpy() == "train")[0]
    test_idx = np.where(records["split"].to_numpy() == "test")[0]
    z_train, h_train_norm = encode_sae(sae, acts[train_idx])
    z_test, h_test_norm = encode_sae(sae, acts[test_idx])
    w_dec = sae.decoder_directions().detach().cpu().numpy()
    print(f"Encoded SAE train {z_train.shape}, test {z_test.shape}", flush=True)

    metadata_readouts: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for concept in CONCEPTS:
        print(f"Running metadata Step0 concept={concept.name}", flush=True)
        y_all = meta[concept.name].to_numpy(dtype=float)
        train_valid = valid_mask(y_all[train_idx], concept.kind)
        test_valid = valid_mask(y_all[test_idx], concept.kind)
        x_train_full = h_train_norm[train_valid]
        y_train_full = y_all[train_idx][train_valid]
        x_test = h_test_norm[test_valid]
        y_test = y_all[test_idx][test_valid]
        if len(y_train_full) == 0 or len(y_test) == 0:
            rows.append({"concept": concept.name, "kind": concept.kind, "status": "missing_target"})
            continue
        cap_idx = cap_train_rows(len(y_train_full), args.probe_train_cap, rng)
        x_train = x_train_full[cap_idx]
        y_train = y_train_full[cap_idx]

        probe = fit_linear_probe(x_train, y_train, x_test, y_test, concept.kind, args.ridge_alpha)
        shuf = shuffled_probe_metrics(
            x_train, y_train, x_test, y_test, concept.kind, args.ridge_alpha, rng, args.shuffle_iters
        )
        shuffle_mean = float(np.nanmean(shuf)) if len(shuf) else float("nan")
        shuffle_p95 = float(np.nanpercentile(shuf, 95)) if len(shuf) else float("nan")
        metric = float(probe["metric"])
        encoded = bool(metric > shuffle_p95 and metric > (0.01 if concept.kind == "continuous" else 0.55))

        ztr = z_train[train_valid]
        zte = z_test[test_valid]
        cav_rank = select_target_features(probe["coef_unit"], w_dec, w_dec.shape[1])
        concept_rank = select_concept_features(ztr[cap_idx], y_train, w_dec.shape[1])
        top = max(1, min(args.gate_feature_count, w_dec.shape[1]))
        cav_top = cav_rank[:top]
        concept_top = concept_rank[:top]
        ageo_top1 = geometric_agreement(probe["coef_unit"][:, None], w_dec, list(cav_rank[:1]))
        ageo_top2 = geometric_agreement(probe["coef_unit"][:, None], w_dec, list(cav_rank[: min(2, w_dec.shape[1])]))
        ageo_top4 = geometric_agreement(probe["coef_unit"][:, None], w_dec, list(cav_rank[: min(4, w_dec.shape[1])]))

        z_probe = fit_linear_probe(
            ztr[cap_idx][:, concept_top], y_train, zte[:, concept_top], y_test, concept.kind, args.ridge_alpha
        )
        z_full_probe = fit_linear_probe(ztr[cap_idx], y_train, zte, y_test, concept.kind, args.ridge_alpha)

        train_mu = float(np.nanmean(y_train)) if concept.kind == "continuous" else None
        train_sd = float(np.nanstd(y_train)) if concept.kind == "continuous" else None
        metadata_readouts[concept.name] = {
            "kind": concept.kind,
            "model": probe["model"],
            "y_test": y_test,
            "clean_score": probe["score"],
            "train_mu": train_mu,
            "train_sd": train_sd,
            "test_valid": test_valid,
        }

        rows.append(
            {
                "concept": concept.name,
                "kind": concept.kind,
                "status": "ok",
                "n_train": int(len(y_train_full)),
                "n_train_probe": int(len(y_train)),
                "n_test": int(len(y_test)),
                "probe_metric": metric,
                "probe_metric_name": probe["metric_name"],
                "probe_shuffle_mean": shuffle_mean,
                "probe_shuffle_p95": shuffle_p95,
                "encoded_gate_pass": encoded,
                "A_geo_cav_top1": float(ageo_top1),
                "A_geo_cav_top2": float(ageo_top2),
                "A_geo_cav_top4": float(ageo_top4),
                "cav_top_features": json.dumps([int(x) for x in cav_top.tolist()]),
                "concept_top_features": json.dumps([int(x) for x in concept_top.tolist()]),
                "sae_code_metric_top": float(z_probe["metric"]),
                "sae_code_metric_full": float(z_full_probe["metric"]),
                "sae_code_metric_name": z_probe["metric_name"],
            }
        )

    # WBI-like metadata clamp is evaluated only on the common test subset where
    # all four metadata concepts are valid, so off-target comparisons are aligned.
    common_test_valid = np.ones(len(test_idx), dtype=bool)
    for concept in CONCEPTS:
        common_test_valid &= valid_mask(meta[concept.name].to_numpy(dtype=float)[test_idx], concept.kind)

    gate_rows = []
    for row in rows:
        if row.get("status") != "ok":
            gate_rows.append(row)
            continue
        concept_name = str(row["concept"])
        concept = next(c for c in CONCEPTS if c.name == concept_name)
        train_valid = valid_mask(meta[concept.name].to_numpy(dtype=float)[train_idx], concept.kind)
        y_train_full = meta[concept.name].to_numpy(dtype=float)[train_idx][train_valid]
        cap_idx = cap_train_rows(len(y_train_full), args.probe_train_cap, rng)
        y_train = y_train_full[cap_idx]
        rank = select_concept_features(z_train[train_valid][cap_idx], y_train, w_dec.shape[1])
        selected = rank[: max(1, min(args.gate_feature_count, w_dec.shape[1]))]

        # Refit clean metadata readouts on common-valid test rows for aligned WBI.
        aligned_readouts: dict[str, dict[str, object]] = {}
        for other in CONCEPTS:
            y_all = meta[other.name].to_numpy(dtype=float)
            tr_valid = valid_mask(y_all[train_idx], other.kind)
            ytr_full = y_all[train_idx][tr_valid]
            readout_cap = cap_train_rows(len(ytr_full), args.probe_train_cap, rng)
            ytr = ytr_full[readout_cap]
            yte = y_all[test_idx][common_test_valid]
            probe = fit_linear_probe(
                h_train_norm[tr_valid][readout_cap],
                ytr,
                h_test_norm[common_test_valid],
                yte,
                other.kind,
                args.ridge_alpha,
            )
            aligned_readouts[other.name] = {
                "kind": other.kind,
                "model": probe["model"],
                "y_test": yte,
                "clean_score": probe["score"],
                "train_mu": float(np.nanmean(ytr)) if other.kind == "continuous" else None,
                "train_sd": float(np.nanstd(ytr)) if other.kind == "continuous" else None,
            }
        wbi_gate = selectivity_gate(
            sae=sae,
            z_train=z_train,
            z_test=z_test[common_test_valid],
            h_test_norm=h_test_norm[common_test_valid],
            metadata=aligned_readouts,
            target=concept_name,
            selected=selected,
            rng=rng,
            n_random=args.random_iters,
        )
        row.update(wbi_gate)
        row["gate_b_ageo_pass"] = bool(float(row["A_geo_cav_top1"]) >= 0.5)
        row["gate_b_target_effect_pass"] = bool(float(row["target_effect"]) > 1e-4)
        row["gate_b_wbi_clean_pass"] = bool(
            np.isfinite(float(row["wbi"]))
            and np.isfinite(float(row["random_wbi_median"]))
            and float(row["wbi"]) <= float(row["random_wbi_median"]) + 0.25
        )
        row["gate_b_pass"] = bool(
            row["encoded_gate_pass"]
            and row["gate_b_ageo_pass"]
            and row["gate_b_target_effect_pass"]
            and row["gate_b_wbi_clean_pass"]
        )
        gate_rows.append(row)

    out_csv = args.out_dir / "metadata_step0_sae_gate_csfm_l5.csv"
    pd.DataFrame(gate_rows).to_csv(out_csv, index=False)

    out_md = args.out_dir / "metadata_step0_sae_gate_csfm_l5.md"
    df = pd.DataFrame(gate_rows)
    survivors = df.loc[df.get("gate_b_pass", False) == True, "concept"].tolist() if "gate_b_pass" in df else []
    lines = [
        "# Metadata Step0 SAE Gate: CSFM Layer 5",
        "",
        "This is a pre-steering gate for route A. It tests whether metadata concepts are encoded and isolated enough in the existing CSFM SAE dictionary to justify later positive-control steering.",
        "",
        f"- activation: `{args.feature_dir / f'layer_{args.layer:02d}_mean.npy'}`",
        f"- SAE checkpoint: `{args.checkpoint}`",
        f"- metadata: `{args.metadata}`",
        f"- gate feature count: {args.gate_feature_count}",
        "",
        "## Gate Table",
        "",
        markdown_table(
            df,
            [
                "concept",
                "probe_metric_name",
                "probe_metric",
                "probe_shuffle_p95",
                "encoded_gate_pass",
                "A_geo_cav_top1",
                "A_geo_cav_top2",
                "A_geo_cav_top4",
                "sae_code_metric_top",
                "target_effect",
                "offtarget_damage",
                "wbi",
                "random_wbi_median",
                "gate_b_pass",
            ],
        ),
        "",
        "## Interpretation",
        "",
    ]
    if survivors:
        lines.append(f"Step0 survivors for route-A preregistration: `{', '.join(survivors)}`.")
    else:
        lines.append(
            "No metadata concept passed the full Step0 gate. That means route A should not proceed to downstream steering yet; the result supports the stronger negative interpretation that even orthogonal metadata concepts are not cleanly isolated in this SAE operating point."
        )
    lines.append("")
    lines.append("CSV output: `" + str(out_csv) + "`")
    out_md.write_text("\n".join(lines) + "\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
