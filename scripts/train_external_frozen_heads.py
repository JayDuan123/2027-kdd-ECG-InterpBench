#!/usr/bin/env python
"""Train cohort-specific linear heads on full external pooled activations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ACT_ROOT = ROOT / "results/activations_external_full_v1/pooled"
OUT_ROOT = ROOT / "results/external_benchmark_v1"
CHALLENGE_LABELS = ROOT / "results/multicohort/challenge_native_label_matrix.csv"
MIMIC_LABELS = ROOT / "results/multicohort/mimic_icd_label_matrix.csv"
TASKS = {
    "chapman_f": ("af_rhythm_native", "bbb_conduction_native", "qt_interval_native", "st_t_abnormal_native"),
    "cpsc_f": ("af_rhythm_native", "bbb_conduction_native"),
    "ningbo_f": ("af_rhythm_native", "bbb_conduction_native", "qt_interval_native", "st_t_abnormal_native"),
    "mimic_f": ("af_rhythm_icd", "bbb_conduction_icd", "qt_interval_icd", "mi_ischemia_icd", "hypertrophy_icd"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-suffix", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--activation-root", type=Path, default=ACT_ROOT)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return p.parse_args()


def split_for(record_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"external-head-v1:{record_id}".encode()).digest()[:8], "big") % 10
    return "train" if value < 7 else "val" if value < 8 else "test"


def load_activations(index_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    ids = []
    with (index_dir / "shards.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            pooled = Path(row["pooled_file"])
            if not pooled.is_absolute():
                pooled = ROOT / pooled
            record_file = Path(row["record_ids_file"])
            if not record_file.is_absolute():
                record_file = ROOT / record_file
            values = np.asarray(np.load(pooled, mmap_mode="r"), dtype=np.float32)
            with record_file.open(newline="") as records:
                shard_ids = [r["record_name"] for r in csv.DictReader(records)]
            if len(values) != len(shard_ids):
                raise RuntimeError(f"{pooled}: {len(values)} activations != {len(shard_ids)} IDs")
            arrays.append(values); ids.extend(shard_ids)
    if not arrays:
        raise RuntimeError(f"No indexed activation shards in {index_dir}")
    acts = np.concatenate(arrays)
    record_ids = np.asarray(ids, dtype="U64")
    if len(set(record_ids)) != len(record_ids):
        raise RuntimeError(f"Duplicate record IDs in {index_dir}")
    if not np.isfinite(acts).all():
        raise RuntimeError(f"Non-finite pooled activations in {index_dir}")
    return acts, record_ids


def main() -> None:
    a = parse_args()
    cohort = a.cohort.lower().replace("-", "_")
    if cohort not in TASKS:
        raise ValueError(f"Unsupported cohort {cohort}")
    index_dir = a.activation_root / a.model_suffix / cohort
    acts, record_ids = load_activations(index_dir)
    if cohort == "mimic_f":
        labels = pd.read_csv(MIMIC_LABELS).set_index("study_id")
        labels.index = labels.index.astype(str)
        split_unit = "patient"
    else:
        labels = pd.read_csv(CHALLENGE_LABELS)
        wanted = {"chapman_f": "Chapman-F", "cpsc_f": "CPSC-F", "ningbo_f": "Ningbo-F"}[cohort]
        labels = labels[labels.cohort.eq(wanted)].set_index("record_id")
        group_ids = record_ids.copy()
        split_unit = "record"
    missing = [record_id for record_id in record_ids if record_id not in labels.index]
    if missing:
        raise RuntimeError(f"Missing native labels for {len(missing)} records; first={missing[:3]}")
    group_ids = labels.loc[record_ids, "subject_id"].astype(str).to_numpy() if cohort == "mimic_f" else record_ids.copy()
    split = np.asarray([split_for(group_id) for group_id in group_ids], dtype="U5")
    tr = split == "train"; va = split == "val"; te = split == "test"
    scaler = StandardScaler().fit(acts[tr])
    xtr = scaler.transform(acts[tr]); xva = scaler.transform(acts[va]); xte = scaler.transform(acts[te])
    heads = {}; metrics = []
    for task in TASKS[cohort]:
        y = labels.loc[record_ids, task].to_numpy(dtype=int)
        if y.sum() < 20 or (len(y) - y.sum()) < 20:
            metrics.append({"task": task, "status": "insufficient_labels", "positives": int(y.sum()), "negatives": int((1-y).sum())})
            continue
        if any(len(np.unique(y[mask])) < 2 for mask in (tr, va, te)):
            metrics.append({
                "task": task, "status": "insufficient_split_support",
                "positives": int(y.sum()), "negatives": int((1-y).sum()),
                "train_positives": int(y[tr].sum()), "val_positives": int(y[va].sum()),
                "test_positives": int(y[te].sum()),
            })
            continue
        best = None
        for c in (0.01, 0.1, 1.0, 10.0):
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=3000, solver="liblinear", random_state=4311)
            clf.fit(xtr, y[tr])
            score = roc_auc_score(y[va], clf.decision_function(xva))
            if best is None or score > best[0]:
                best = (score, c, clf)
        assert best is not None
        clf = best[2]
        test_score = clf.decision_function(xte)
        row = {
            "task": task, "status": "ok", "positives": int(y.sum()), "negatives": int((1-y).sum()),
            "train_n": int(tr.sum()), "val_n": int(va.sum()), "test_n": int(te.sum()),
            "best_C": best[1], "val_auroc": best[0], "test_auroc": roc_auc_score(y[te], test_score),
            "test_auprc": average_precision_score(y[te], test_score),
        }
        metrics.append(row)
        heads[task] = {"clf": clf, "labels": y, "type": "binary", "metrics": row}
    out = a.out_root / a.model_suffix / cohort
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_suffix": a.model_suffix, "cohort": cohort, "scaler": scaler, "heads": heads,
        "targets": list(heads), "record_ids": record_ids, "group_ids": group_ids, "split": split,
        "activation_index": str(index_dir), "split_unit": split_unit,
    }
    tmp = out / f"frozen_heads.joblib.tmp.{os.getpid()}"
    joblib.dump(bundle, tmp); os.replace(tmp, out / "frozen_heads.joblib")
    pd.DataFrame(metrics).to_csv(out / "frozen_heads_metrics.csv", index=False)
    pd.DataFrame({"record_id": record_ids, "split": split}).to_csv(out / "manifest.csv", index=False)
    summary = {
        "model_suffix": a.model_suffix, "cohort": cohort, "records": len(record_ids),
        "hidden_dim": acts.shape[1], "tasks_trained": len(heads), "tasks_requested": len(TASKS[cohort]),
        "all_finite": True, "split_unit": split_unit,
    }
    (out / "head_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    print(pd.DataFrame(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
