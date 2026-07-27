#!/usr/bin/env python
"""Reuse frozen diagnosis heads and fit continuous/metadata steering readouts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/sae_reconciliation/phenotype_steering"
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    p.add_argument("--manifest", type=Path, default=BASE / "manifest.csv")
    p.add_argument("--registry", type=Path, default=BASE / "target_registry.csv")
    p.add_argument("--old-heads", type=Path, default=OLD / "frozen_heads.joblib")
    p.add_argument("--out", type=Path, default=BASE / "frozen_heads.joblib")
    p.add_argument("--seed", type=int, default=4311)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    frame = pd.read_csv(a.manifest)
    registry = pd.read_csv(a.registry)
    split = frame.split.to_numpy()
    x = np.asarray(np.load(a.acts, mmap_mode="r"), dtype=np.float32)
    old = joblib.load(a.old_heads)
    scaler = old["scaler"]
    xs = scaler.transform(x)
    heads = dict(old["heads"])
    metrics = dict(old.get("metrics", {}))

    for ti, spec in enumerate(registry.itertuples(index=False)):
        target = spec.target
        y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        if target in heads:
            heads[target]["type"] = "binary"
            heads[target]["family"] = spec.family
            heads[target]["role"] = spec.analysis_role
            continue
        tr = np.where((split == "train") & np.isfinite(y))[0]
        va = np.where((split == "val") & np.isfinite(y))[0]
        te = np.where((split == "test") & np.isfinite(y))[0]
        if spec.target_type == "continuous":
            mu, sd = float(y[tr].mean()), float(y[tr].std())
            if sd < 1e-8:
                raise RuntimeError(f"Near-constant target: {target}")
            ys = (y - mu) / sd
            fits = []
            for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
                model = Ridge(alpha=alpha, solver="lsqr").fit(xs[tr], ys[tr])
                fits.append((r2_score(ys[va], model.predict(xs[va])), alpha, model))
            val_score, alpha, model = max(fits, key=lambda z: z[0])
            pred = model.predict(xs[te])
            heads[target] = {"clf": model, "labels": y, "type": "continuous", "family": spec.family,
                             "role": spec.analysis_role, "target_mean": mu, "target_std": sd}
            metrics[target] = {"alpha": alpha, "val_r2": float(val_score), "test_r2": float(r2_score(ys[te], pred)),
                               "test_mae_standardized": float(mean_absolute_error(ys[te], pred)),
                               "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        else:
            fits = []
            for C in (0.01, 0.1, 1.0, 10.0):
                model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight="balanced",
                                           max_iter=3000, random_state=a.seed + ti).fit(xs[tr], y[tr].astype(int))
                fits.append((roc_auc_score(y[va], model.decision_function(xs[va])), C, model))
            val_score, C, model = max(fits, key=lambda z: z[0])
            score = model.decision_function(xs[te]); prob = 1.0 / (1.0 + np.exp(-np.clip(score, -50, 50)))
            heads[target] = {"clf": model, "labels": y, "type": "binary", "family": spec.family,
                             "role": spec.analysis_role}
            metrics[target] = {"C": C, "val_auroc": float(val_score), "test_auroc": float(roc_auc_score(y[te], score)),
                               "test_auprc": float(average_precision_score(y[te], prob)),
                               "test_brier": float(brier_score_loss(y[te], prob)),
                               "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
        print(target, json.dumps(metrics[target]), flush=True)

    targets = registry.target.tolist()
    payload = {"scaler": scaler, "heads": heads, "metrics": metrics, "targets": targets,
               "provenance": {"head_input": "raw CSFM Layer-6 mean embedding", "split": "patient-level PTB-XL",
                              "continuous_targets": "train-standardized", "seed": a.seed}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + f".tmp.{os.getpid()}")
    joblib.dump(payload, tmp); tmp.replace(a.out)
    a.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
