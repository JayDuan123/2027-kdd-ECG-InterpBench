#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results" / "multicohort"

PRIMARY_TASKS = {"af_rhythm_icd", "bbb_conduction_icd"}
SENSITIVITY_TASKS = {"qt_interval_icd"}
TASKS = [
    "af_rhythm_icd",
    "bbb_conduction_icd",
    "qt_interval_icd",
    "mi_ischemia_icd",
    "hypertrophy_icd",
]
CONCEPTS = [
    "hr_ventricular",
    "hr_atrial",
    "rr_mean",
    "pr_interval",
    "pq_interval",
    "p_duration_global",
    "qrs_duration",
    "qt_interval",
    "qtc_bazett",
    "qtc_fridericia",
    "qtc_framingham",
    "p_axis_front",
    "qrs_axis_front",
    "t_axis_front",
    "qrst_angle",
]
MINIMAL_CONCEPTS = [
    "rr_mean",
    "hr_ventricular",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "qtc_bazett",
    "qrs_axis_front",
    "t_axis_front",
    "qrst_angle",
]


def subject_split(subject_id: str) -> str:
    digest = hashlib.sha1(f"mimic_v_closure::{subject_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "val"
    return "test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIMIC-V vendor-measurement closure baselines.")
    parser.add_argument("--concepts", type=Path, default=DEFAULT_DIR / "mimic_vendor_concepts.csv")
    parser.add_argument("--labels", type=Path, default=DEFAULT_DIR / "mimic_icd_label_matrix.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIR / "mimic_v_closure")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260706)
    return parser.parse_args()


def load_joined(args: argparse.Namespace):
    import numpy as np
    import pandas as pd

    concepts = pd.read_csv(args.concepts)
    labels = pd.read_csv(args.labels)
    if args.max_rows is not None:
        # Keep smoke deterministic while preserving real study_id join.
        labels = labels.head(args.max_rows)
    merged = labels.merge(concepts, on=["subject_id", "study_id", "ecg_time"], how="inner")
    splits = np.array([subject_split(str(x)) for x in merged["subject_id"].tolist()])
    x = merged[CONCEPTS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    y = merged[TASKS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return merged, splits, x, y


def impute_and_scale(x, train_idx):
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    out = np.array(x, dtype=np.float32, copy=True)
    med = np.nanmedian(out[train_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    missing = ~np.isfinite(out)
    out[missing] = med[np.where(missing)[1]]
    scaler = StandardScaler()
    out[train_idx] = scaler.fit_transform(out[train_idx])
    non_train = np.setdiff1d(np.arange(out.shape[0]), train_idx)
    if len(non_train):
        out[non_train] = scaler.transform(out[non_train])
    return out


def evaluate_block(block: str, x, y, splits, alpha: float) -> list[dict[str, object]]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    rows = []
    for j, task in enumerate(TASKS):
        y_train = y[train_idx, j]
        y_val = y[val_idx, j]
        y_test = y[test_idx, j]
        train_ok = np.isfinite(y_train)
        val_ok = np.isfinite(y_val)
        test_ok = np.isfinite(y_test)
        if task in PRIMARY_TASKS:
            status_scope = "primary"
        elif task in SENSITIVITY_TASKS:
            status_scope = "sensitivity_label_measurement_proximal"
        else:
            status_scope = "out_of_scope_missing_measurement_family"
        train_classes = set(y_train[train_ok].astype(int).tolist())
        test_classes = set(y_test[test_ok].astype(int).tolist())
        val_classes = set(y_val[val_ok].astype(int).tolist())
        if len(train_classes) < 2 or len(test_classes) < 2:
            rows.append(
                {
                    "block": block,
                    "task": task,
                    "task_scope": status_scope,
                    "status": "skipped_single_class",
                    "n_train": int(train_ok.sum()),
                    "n_val": int(val_ok.sum()),
                    "n_test": int(test_ok.sum()),
                    "val_auroc": "",
                    "val_auprc": "",
                    "test_auroc": "",
                    "test_auprc": "",
                }
            )
            continue
        model = RidgeClassifier(alpha=alpha)
        model.fit(x[train_idx][train_ok], y_train[train_ok].astype(int))
        test_score = model.decision_function(x[test_idx][test_ok])
        if len(val_classes) >= 2:
            val_score = model.decision_function(x[val_idx][val_ok])
            val_auroc = f"{roc_auc_score(y_val[val_ok], val_score):.8g}"
            val_auprc = f"{average_precision_score(y_val[val_ok], val_score):.8g}"
        else:
            val_auroc = ""
            val_auprc = ""
        rows.append(
            {
                "block": block,
                "task": task,
                "task_scope": status_scope,
                "status": "ok",
                "n_train": int(train_ok.sum()),
                "n_val": int(val_ok.sum()),
                "n_test": int(test_ok.sum()),
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "test_auroc": f"{roc_auc_score(y_test[test_ok], test_score):.8g}",
                "test_auprc": f"{average_precision_score(y_test[test_ok], test_score):.8g}",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np

    merged, splits, x_raw, y = load_joined(args)
    train_idx = np.where(splits == "train")[0]
    rng = np.random.default_rng(args.seed)
    x_all = impute_and_scale(x_raw, train_idx)
    minimal_idx = [CONCEPTS.index(c) for c in MINIMAL_CONCEPTS]
    blocks = [
        ("Bcommon_mimic_v_all15", x_all, len(CONCEPTS)),
        ("Bminimal_mimic_v_9", x_all[:, minimal_idx], len(minimal_idx)),
        ("Brand_gaussian_dim15", rng.normal(size=x_all.shape).astype(np.float32), len(CONCEPTS)),
    ]
    rows: list[dict[str, object]] = []
    block_rows = []
    for block, x, dim in blocks:
        block_rows.append({"block": block, "n_features": dim})
        rows.extend(evaluate_block(block, x, y, splits, args.alpha))

    fields = [
        "block",
        "task",
        "task_scope",
        "status",
        "n_train",
        "n_val",
        "n_test",
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
    ]
    write_csv(args.out_dir / "mimic_v_closure_scores.csv", rows, fields)
    write_csv(args.out_dir / "mimic_v_closure_blocks.csv", block_rows, ["block", "n_features"])
    split_counts = {split: int((splits == split).sum()) for split in ["train", "val", "test"]}
    report = {
        "n_joined_records": int(len(merged)),
        "split_counts": split_counts,
        "tasks": TASKS,
        "primary_tasks": sorted(PRIMARY_TASKS),
        "concepts": CONCEPTS,
        "minimal_concepts": MINIMAL_CONCEPTS,
        "alpha": args.alpha,
        "seed": args.seed,
        "note": "MIMIC-V closure uses ICD-linked labels and interval/rate/axis vendor concepts only; AF/BBB are primary tasks, QT is sensitivity-only because it is measurement-proximal, and MI/HYP rows are out-of-scope audit rows.",
    }
    (args.out_dir / "mimic_v_closure_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
