#!/usr/bin/env python
"""Audit scale invariants, reconstruction fidelity, and frozen-readout retention."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results/sae_reconciliation/matched_scale_v1"
V2 = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"


def reconstruct(checkpoint: Path, acts: np.ndarray, batch_size: int = 256) -> np.ndarray:
    import torch

    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    saved = torch.load(checkpoint, map_location="cuda")
    config = saved["config"]
    sae = BatchTopKSAE(acts.shape[1], int(config["n_features"]), int(config["k"])).cuda()
    sae.load_state_dict(saved["model"])
    sae.eval()
    chunks = []
    with torch.no_grad():
        for lo in range(0, len(acts), batch_size):
            raw = torch.as_tensor(np.asarray(acts[lo : lo + batch_size]), dtype=torch.float32, device="cuda")
            chunks.append(sae.decode(sae.encode(raw)).cpu().numpy())
    return np.concatenate(chunks)


def main() -> None:
    manifest = pd.read_csv(BASE / "training_manifest.csv")
    split = pd.read_csv(
        ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv"
    ).split.to_numpy()
    rows = []
    for cell in manifest.itertuples(index=False):
        checkpoint = Path(cell.checkpoint)
        metrics_path = checkpoint.with_suffix(".metrics.json")
        if not checkpoint.exists() or not metrics_path.exists():
            rows.append({"model": cell.model, "seed": cell.seed, "status": "missing"})
            continue
        metrics = json.loads(metrics_path.read_text())
        acts = np.load(ROOT / f"results/probe_features/{cell.feature_suffix}/pooled.npy", mmap_mode="r")
        test_idx = np.where(split == "test")[0]
        raw = np.asarray(acts[test_idx], dtype=np.float32)
        recon = reconstruct(checkpoint, raw)
        safe = str(cell.model).lower().replace("-", "_")
        bundle = joblib.load(V2 / "models" / safe / "frozen_heads.joblib")
        scaler = bundle["scaler"]
        raw_scaled = scaler.transform(raw)
        recon_scaled = scaler.transform(recon)
        retentions = []
        for target in bundle["targets"]:
            head = bundle["heads"][target]
            labels = np.asarray(head["labels"], dtype=float)[test_idx]
            valid = np.isfinite(labels)
            if head.get("type", "binary") == "binary":
                y = labels[valid].astype(int)
                if len(np.unique(y)) < 2:
                    continue
                raw_metric = roc_auc_score(y, head["clf"].decision_function(raw_scaled[valid]))
                recon_metric = roc_auc_score(y, head["clf"].decision_function(recon_scaled[valid]))
                retention = recon_metric / max(raw_metric, 1e-8)
            else:
                mu, sd = float(head["target_mean"]), float(head["target_std"])
                y = (labels[valid] - mu) / sd
                raw_metric = r2_score(y, head["clf"].predict(raw_scaled[valid]))
                recon_metric = r2_score(y, head["clf"].predict(recon_scaled[valid]))
                retention = recon_metric / raw_metric if raw_metric > 0.05 else np.nan
            if np.isfinite(retention):
                retentions.append(float(retention))
        median_retention = float(np.median(retentions))
        recon_r2 = float(metrics["explained_variance"])
        dead = float(metrics["dead_fraction"])
        scale_pass = (
            int(cell.N) == int(cell.expansion_E * cell.d_hidden)
            and np.isclose(float(cell.k_over_d), int(cell.k) / int(cell.d_hidden))
            and np.isclose(float(cell.k_over_N), int(cell.k) / int(cell.N))
        )
        fidelity_pass = recon_r2 >= 0.90 and dead < 0.20 and median_retention >= 0.95
        rows.append(
            {
                "model": cell.model,
                "seed": int(cell.seed),
                "d_hidden": int(cell.d_hidden),
                "N": int(cell.N),
                "k": int(cell.k),
                "N_over_d": int(cell.N) / int(cell.d_hidden),
                "k_over_d": int(cell.k) / int(cell.d_hidden),
                "k_over_N": int(cell.k) / int(cell.N),
                "recon_R2": recon_r2,
                "dead_fraction": dead,
                "median_readout_retention": median_retention,
                "readouts_evaluated": len(retentions),
                "scale_invariants_pass": bool(scale_pass),
                "fidelity_pass": bool(fidelity_pass),
                "status": "pass" if scale_pass and fidelity_pass else "quality_warning",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(BASE / "matched_scale_fidelity_audit.csv", index=False)
    if audit.status.eq("missing").any():
        raise RuntimeError("Missing matched-scale checkpoints or metrics")
    if not audit.scale_invariants_pass.all():
        raise RuntimeError("Matched-scale invariants failed")
    profile = audit.groupby("model", as_index=False).agg(
        seeds=("seed", "nunique"),
        recon_R2_mean=("recon_R2", "mean"),
        recon_R2_min=("recon_R2", "min"),
        dead_fraction_max=("dead_fraction", "max"),
        readout_retention_median=("median_readout_retention", "median"),
        fidelity_pass_seeds=("fidelity_pass", "sum"),
    )
    profile["matched_scale_primary_eligible"] = profile.fidelity_pass_seeds.eq(3)
    profile.to_csv(BASE / "matched_scale_model_profile.csv", index=False)
    print(audit.to_string(index=False))
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
