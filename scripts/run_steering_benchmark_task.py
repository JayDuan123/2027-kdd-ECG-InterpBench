#!/usr/bin/env python
"""Run one CSFM SAE steering cell with all-readout bidirectional controls."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="CSFM")
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--acts", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    p.add_argument("--manifest", type=Path, default=BASE / "manifest.csv")
    p.add_argument("--registry", type=Path, default=BASE / "target_registry.csv")
    p.add_argument("--heads", type=Path, default=BASE / "frozen_heads.joblib")
    p.add_argument("--out-dir", type=Path, default=BASE / "tasks")
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def encode(sae, x: np.ndarray, batch: int, device: str) -> np.ndarray:
    import torch
    chunks = []
    sae.eval()
    with torch.no_grad():
        for lo in range(0, len(x), batch):
            chunks.append(sae.encode(torch.as_tensor(np.asarray(x[lo:lo + batch]), dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(chunks)


def threshold_at_specificity(y: np.ndarray, score: np.ndarray, specificity: float = 0.95) -> float:
    neg = np.asarray(score)[np.asarray(y) == 0]
    if len(neg) == 0:
        return float("nan")
    return float(np.quantile(neg, specificity, method="higher"))


def ece(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob < edges[i + 1] if i + 1 < bins else prob <= edges[i + 1])
        if mask.any():
            out += mask.mean() * abs(float(prob[mask].mean()) - float(y[mask].mean()))
    return float(out)


def main() -> None:
    a = parse_args()
    import torch
    from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    registry = pd.read_csv(a.registry)
    if a.target not in set(registry.target):
        raise ValueError(f"Unknown target {a.target}")
    registry_by_target = registry.set_index("target")
    frame = pd.read_csv(a.manifest)
    split = frame.split.to_numpy(); patients = frame.patient_id.astype(str).to_numpy()
    x = np.asarray(np.load(a.acts, mmap_mode="r"), dtype=np.float32)
    bundle = joblib.load(a.heads); scaler = bundle["scaler"]; heads = bundle["heads"]
    names = list(bundle["targets"])
    labels = {name: np.asarray(heads[name]["labels"], dtype=float) for name in names}
    kinds = {name: heads[name].get("type", "binary") for name in names}
    families = {name: str(registry_by_target.loc[name, "family"]) for name in names}

    out = a.out_dir / f"seed{a.seed}" / a.target
    final = out / "result.json"; records = out / "records.npz"
    out.mkdir(parents=True, exist_ok=True)
    if final.exists() and records.exists():
        try:
            if json.loads(final.read_text()).get("schema_version") == 3:
                print(f"already complete: {out}"); return
        except (OSError, json.JSONDecodeError):
            pass

    tr = np.where(split == "train")[0]; va = np.where(split == "val")[0]; te = np.where(split == "test")[0]
    ck = torch.load(a.checkpoint, map_location=a.device); cfg = ck["config"]
    n_features, sae_k = int(cfg["n_features"]), int(cfg["k"])
    cache = a.out_dir.parent / "shared_cache" / f"seed{a.seed}_N{n_features}_k{sae_k}"
    complete = cache / "complete.json"; lock = cache.with_name(cache.name + ".lock")
    expected = {"model": a.model, "seed": a.seed, "checkpoint": str(a.checkpoint), "targets": names}

    def cache_valid() -> bool:
        if not complete.exists(): return False
        try: return json.loads(complete.read_text()) == expected
        except (OSError, json.JSONDecodeError): return False

    builder = False
    if not cache_valid():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(lock); builder = True
        except FileExistsError:
            for _ in range(600):
                if cache_valid(): break
                time.sleep(1)
            else: raise RuntimeError(f"timed out waiting for shared cache {cache}")
    if builder:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            sae = BatchTopKSAE(x.shape[1], n_features, sae_k).to(a.device)
            sae.load_state_dict(ck["model"]); sae.eval()
            ztr = encode(sae, x[tr], a.batch_size, a.device)
            zva = encode(sae, x[va], a.batch_size, a.device)
            zte_build = encode(sae, x[te], a.batch_size, a.device)
            dec = sae.W_dec.detach().cpu().numpy(); sigma = sae.sigma.detach().cpu().numpy()
            gradient_matrix = []
            for name in names:
                coef = np.asarray(heads[name]["clf"].coef_).reshape(-1)
                gradient_matrix.append(((coef / scaler.scale_) * sigma) @ dec)
            gradient_matrix = np.asarray(gradient_matrix)
            focus_thresholds_build = {}; ranking_matrix = []
            for j, name in enumerate(names):
                y = labels[name][tr]; valid = np.isfinite(y)
                if kinds[name] == "binary": focus = valid & (y == 1); focus_thresholds_build[name] = 1.0
                else:
                    threshold = np.nanquantile(y[valid], .75); focus = valid & (y >= threshold); focus_thresholds_build[name] = float(threshold)
                ig = (ztr[focus] * gradient_matrix[j]).mean(0); ranking_matrix.append(np.argsort(ig)[::-1])
            with torch.no_grad():
                recon_va = sae.decode(torch.as_tensor(zva, device=a.device)).cpu().numpy()
                recon_te = sae.decode(torch.as_tensor(zte_build, device=a.device)).cpu().numpy()
            base_build = np.column_stack([heads[name]["clf"].decision_function(scaler.transform(recon_te)) if kinds[name] == "binary"
                                          else heads[name]["clf"].predict(scaler.transform(recon_te)) for name in names])
            base_va_build = np.column_stack([heads[name]["clf"].decision_function(scaler.transform(recon_va)) if kinds[name] == "binary"
                                             else heads[name]["clf"].predict(scaler.transform(recon_va)) for name in names])
            thresholds_build = np.full(len(names), np.nan)
            for j, name in enumerate(names):
                if kinds[name] == "binary":
                    yv = labels[name][va]; valid = np.isfinite(yv)
                    thresholds_build[j] = threshold_at_specificity(yv[valid].astype(int), base_va_build[valid, j])
            arrays = {"zte": zte_build.astype(np.float32), "gradients": gradient_matrix.astype(np.float64),
                      "rankings": np.asarray(ranking_matrix, dtype=np.int32), "freq": (ztr > 0).mean(0).astype(np.float32),
                      "mag": np.divide(ztr.sum(0), (ztr > 0).sum(0), out=np.zeros(ztr.shape[1]), where=(ztr > 0).sum(0) > 0).astype(np.float32),
                      "centroid": ztr.mean(0).astype(np.float32), "base": base_build.astype(np.float32),
                      "thresholds": thresholds_build.astype(np.float32)}
            for key, value in arrays.items():
                tmp = cache / f"{key}.npy.tmp.{os.getpid()}"
                with tmp.open("wb") as handle: np.save(handle, value)
                tmp.replace(cache / f"{key}.npy")
            (cache / "focus_thresholds.json").write_text(json.dumps(focus_thresholds_build, sort_keys=True) + "\n")
            tmp_complete = cache / f"complete.json.tmp.{os.getpid()}"; tmp_complete.write_text(json.dumps(expected, sort_keys=True) + "\n"); tmp_complete.replace(complete)
        finally:
            try: os.rmdir(lock)
            except OSError: pass

    zte = np.load(cache / "zte.npy", mmap_mode="r")
    gradient_matrix = np.load(cache / "gradients.npy", mmap_mode="r")
    ranking_matrix = np.load(cache / "rankings.npy", mmap_mode="r")
    freq = np.load(cache / "freq.npy", mmap_mode="r"); mag = np.load(cache / "mag.npy", mmap_mode="r")
    centroid = np.load(cache / "centroid.npy", mmap_mode="r"); base = np.load(cache / "base.npy", mmap_mode="r")
    thresholds = np.load(cache / "thresholds.npy", mmap_mode="r")
    focus_thresholds = json.loads((cache / "focus_thresholds.json").read_text())
    gradients = {name: gradient_matrix[j] for j, name in enumerate(names)}
    rankings = {name: ranking_matrix[j] for j, name in enumerate(names)}
    rank = rankings[a.target]
    selected = {f"top{k}": rank[:k].astype(int) for k in (1, 5, 10)}

    excluded = np.unique(np.concatenate([rankings[name][:10] for name in names]))
    rng = np.random.default_rng(a.seed + sum(map(ord, a.target)))
    random_groups = []
    for _ in range(a.n_random):
        group = []
        for atom in selected["top5"]:
            dist = np.abs(np.log((freq + 1e-6) / (freq[atom] + 1e-6))) + np.abs(np.log((mag + 1e-6) / (mag[atom] + 1e-6)))
            blocked = np.unique(np.concatenate([excluded.astype(int), np.asarray(group, dtype=int)]))
            dist[blocked] = np.inf
            pool = np.argsort(dist)[:200]
            group.append(int(rng.choice(pool)))
        random_groups.append(group)
    random_groups = np.asarray(random_groups, dtype=int)

    def delta(indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=int)
        dz = centroid[idx][None, :] - zte[:, idx]
        return np.column_stack([dz @ gradients[name][idx] for name in names])

    deltas = {key: delta(atoms) for key, atoms in selected.items()}
    random_delta = np.stack([delta(group) for group in random_groups], axis=1)
    def metrics_for(j: int, edited: np.ndarray) -> dict[str, float]:
        name = names[j]; y = labels[name][te]; valid = np.isfinite(y); yy = y[valid]
        clean = base[valid, j]; edit = edited[valid]
        if kinds[name] == "continuous":
            mu, sd = heads[name]["target_mean"], heads[name]["target_std"]
            ys = (yy - mu) / sd
            return {"baseline_r2": float(r2_score(ys, clean)), "edited_r2": float(r2_score(ys, edit)),
                    "r2_drop": float(r2_score(ys, clean) - r2_score(ys, edit)),
                    "mae_change": float(mean_absolute_error(ys, edit) - mean_absolute_error(ys, clean))}
        yy = yy.astype(int); pc = sigmoid(clean); pe = sigmoid(edit); threshold = thresholds[j]
        pos = yy == 1
        return {"baseline_auroc": float(roc_auc_score(yy, clean)), "edited_auroc": float(roc_auc_score(yy, edit)),
                "auroc_drop": float(roc_auc_score(yy, clean) - roc_auc_score(yy, edit)),
                "baseline_auprc": float(average_precision_score(yy, pc)), "edited_auprc": float(average_precision_score(yy, pe)),
                "baseline_sens_at_95spec": float((clean[pos] >= threshold).mean()),
                "edited_sens_at_95spec": float((edit[pos] >= threshold).mean()),
                "sens_drop_at_95spec": float((clean[pos] >= threshold).mean() - (edit[pos] >= threshold).mean()),
                "decision_flip_rate": float(((clean >= threshold) != (edit >= threshold)).mean()),
                "brier_change": float(brier_score_loss(yy, pe) - brier_score_loss(yy, pc)),
                "ece_change": float(ece(yy, pe) - ece(yy, pc))}

    target_j = names.index(a.target)
    result = {
        "schema_version": 3, "model": a.model, "target": a.target, "target_type": kinds[a.target], "family": families[a.target], "seed": a.seed,
        "checkpoint": str(a.checkpoint), "primary_intervention": "top5_population_centroid",
        "focus_thresholds_train": focus_thresholds,
        "selected_atoms": {key: value.tolist() for key, value in selected.items()},
        "random_groups": random_groups.tolist(),
        "metrics": {key: {name: metrics_for(j, base[:, j] + value[:, j]) for j, name in enumerate(names)}
                    for key, value in deltas.items()},
        "guards": {"selection_split": "train", "threshold_split": "validation_SAE_reconstruction",
                   "evaluation_split": "test", "specificity_target": 0.95, "neutral_clamp": "train_population_centroid",
                   "all_readouts_saved": True, "matched_random_group_size": 5},
    }
    tmp = final.with_suffix(f".json.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(result, indent=2) + "\n"); tmp.replace(final)
    labels_test = np.column_stack([labels[name][te] for name in names])
    target_means = np.asarray([heads[name].get("target_mean", np.nan) for name in names], dtype=float)
    target_stds = np.asarray([heads[name].get("target_std", np.nan) for name in names], dtype=float)
    npz_tmp = out / f"records.npz.tmp.{os.getpid()}"
    with npz_tmp.open("wb") as f:
        np.savez(f, patient_ids=np.asarray(patients[te], dtype="U64"), target_names=np.asarray(names, dtype="U64"),
                            target_types=np.asarray([kinds[n] for n in names], dtype="U16"),
                            labels=labels_test.astype(np.float32), baseline_logits=base.astype(np.float32),
                            top1_delta=deltas["top1"].astype(np.float32), top5_delta=deltas["top5"].astype(np.float32),
                            top10_delta=deltas["top10"].astype(np.float32), random_top5_delta=random_delta.astype(np.float32),
                            thresholds_95spec=thresholds.astype(np.float32),
                            continuous_target_means=target_means, continuous_target_stds=target_stds)
    npz_tmp.replace(records)
    print(json.dumps({"target": a.target, "seed": a.seed, "top5": selected["top5"].tolist(),
                      "target_metric": result["metrics"]["top5"][a.target]}))


if __name__ == "__main__":
    main()
