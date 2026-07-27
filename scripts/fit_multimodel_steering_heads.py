#!/usr/bin/env python
"""Fit the same frozen readout protocol for one model's pooled representation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--acts", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=BASE / "manifest.csv")
    p.add_argument("--registry", type=Path, default=BASE / "target_registry.csv")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=4311)
    return p.parse_args()


def main() -> None:
    a = parse_args(); frame = pd.read_csv(a.manifest); registry = pd.read_csv(a.registry)
    split = frame.split.to_numpy(); x = np.asarray(np.load(a.acts, mmap_mode="r"), dtype=np.float32)
    scaler = StandardScaler().fit(x[split == "train"]); xs = scaler.transform(x)
    heads = {}; metrics = {}; workers = min(12, int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    for ti, spec in enumerate(registry.itertuples(index=False)):
        target = spec.target; y = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=float)
        tr = np.where((split == "train") & np.isfinite(y))[0]; va = np.where((split == "val") & np.isfinite(y))[0]; te = np.where((split == "test") & np.isfinite(y))[0]
        if spec.target_type == "continuous":
            mu, sd = float(y[tr].mean()), float(y[tr].std()); ys = (y - mu) / sd
            def fit_ridge(alpha):
                model = Ridge(alpha=alpha, solver="lsqr").fit(xs[tr], ys[tr])
                return r2_score(ys[va], model.predict(xs[va])), alpha, model
            fits = Parallel(n_jobs=min(workers, 5), prefer="threads")(delayed(fit_ridge)(v) for v in (0.1, 1., 10., 100., 1000.))
            val, alpha, model = max(fits, key=lambda z: z[0]); pred = model.predict(xs[te])
            heads[target] = {"clf": model, "labels": y, "type": "continuous", "family": spec.family,
                             "role": spec.analysis_role, "target_mean": mu, "target_std": sd}
            metrics[target] = {"alpha": alpha, "val_r2": float(val), "test_r2": float(r2_score(ys[te], pred)),
                               "test_mae_standardized": float(mean_absolute_error(ys[te], pred))}
        else:
            def fit_logistic(C):
                model = LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight="balanced",
                                           max_iter=3000, random_state=a.seed + ti).fit(xs[tr], y[tr].astype(int))
                return roc_auc_score(y[va], model.decision_function(xs[va])), C, model
            fits = Parallel(n_jobs=min(workers, 4), prefer="threads")(delayed(fit_logistic)(v) for v in (0.01, 0.1, 1., 10.))
            val, C, model = max(fits, key=lambda z: z[0]); score = model.decision_function(xs[te]); prob = 1 / (1 + np.exp(-np.clip(score, -50, 50)))
            heads[target] = {"clf": model, "labels": y, "type": "binary", "family": spec.family, "role": spec.analysis_role}
            metrics[target] = {"C": C, "val_auroc": float(val), "test_auroc": float(roc_auc_score(y[te], score)),
                               "test_auprc": float(average_precision_score(y[te], prob)), "test_brier": float(brier_score_loss(y[te], prob))}
        print(target, json.dumps(metrics[target]), flush=True)
    payload = {"model": a.model, "scaler": scaler, "heads": heads, "metrics": metrics,
               "targets": registry.target.tolist(), "provenance": {"representation": "model_pooled", "protocol": "shared_v1", "seed": a.seed}}
    a.out.parent.mkdir(parents=True, exist_ok=True); tmp = a.out.with_suffix(a.out.suffix + f".tmp.{os.getpid()}")
    joblib.dump(payload, tmp); tmp.replace(a.out); a.out.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
