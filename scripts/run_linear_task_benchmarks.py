#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.config import ROOT


MINIMAL_CONCEPTS = [
    "hr_ventricular",
    "rr_mean",
    "pr_interval",
    "qrs_duration",
    "qt_interval",
    "qtc_bazett",
    "qrs_axis_front",
    "t_axis_front",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run linear FM-head and concept closure baselines.")
    parser.add_argument("--probe-features-dir", required=True, type=Path)
    parser.add_argument("--probe-atlas-dir", required=True, type=Path)
    parser.add_argument("--concepts-matrix", default=ROOT / "results" / "manifest" / "concepts_matrix.csv", type=Path)
    parser.add_argument("--tasks-matrix", default=ROOT / "results" / "manifest" / "tasks_matrix.csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_feature_manifest(probe_features_dir: Path, feature_name: str):
    import numpy as np

    for row in read_csv(probe_features_dir / "features.csv"):
        if row["feature"] == feature_name:
            return np.load(ROOT / row["file"], mmap_mode="r")
    raise ValueError(f"feature {feature_name!r} not found in {probe_features_dir / 'features.csv'}")


def concept_matrix(records: list[dict[str, str]], concepts_matrix: Path, concept_ids: list[str]):
    import numpy as np

    concept_rows = {row["ecg_id"]: row for row in read_csv(concepts_matrix)}
    x = np.empty((len(records), len(concept_ids)), dtype=np.float32)
    for i, record in enumerate(records):
        row = concept_rows[record["ecg_id"]]
        x[i] = [parse_float(row[cid]) for cid in concept_ids]
    return x


def task_matrix(records: list[dict[str, str]], tasks_matrix: Path):
    import numpy as np

    task_rows = {row["ecg_id"]: row for row in read_csv(tasks_matrix)}
    task_ids = [field for field in next(iter(task_rows.values())).keys() if field != "ecg_id"]
    y = np.empty((len(records), len(task_ids)), dtype=np.float32)
    for i, record in enumerate(records):
        row = task_rows[record["ecg_id"]]
        y[i] = [parse_float(row[task]) for task in task_ids]
    return task_ids, y


def clean_concept_features(x, train_idx):
    import numpy as np

    out = np.array(x, dtype=np.float32, copy=True)
    med = np.nanmedian(out[train_idx], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(out))
    out[inds] = med[inds[1]]
    return out


def evaluate_block(name: str, x, y, task_ids: list[str], splits, alpha: float):
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(np.asarray(x[train_idx], dtype=np.float32))
    x_val = scaler.transform(np.asarray(x[val_idx], dtype=np.float32))
    x_test = scaler.transform(np.asarray(x[test_idx], dtype=np.float32))

    rows: list[dict[str, object]] = []
    for task_j, task_id in enumerate(task_ids):
        y_train = y[train_idx, task_j]
        y_val = y[val_idx, task_j]
        y_test = y[test_idx, task_j]
        valid_train = np.isfinite(y_train)
        valid_val = np.isfinite(y_val)
        valid_test = np.isfinite(y_test)
        train_classes = set(y_train[valid_train].astype(int).tolist())
        test_classes = set(y_test[valid_test].astype(int).tolist())
        if len(train_classes) < 2 or len(test_classes) < 2:
            rows.append(
                {
                    "block": name,
                    "task_id": task_id,
                    "status": "skipped_single_class",
                    "n_train": int(valid_train.sum()),
                    "n_val": int(valid_val.sum()),
                    "n_test": int(valid_test.sum()),
                    "test_auroc": "",
                    "test_auprc": "",
                    "val_auroc": "",
                    "val_auprc": "",
                }
            )
            continue

        model = RidgeClassifier(alpha=alpha)
        model.fit(x_train[valid_train], y_train[valid_train].astype(int))
        val_score = model.decision_function(x_val[valid_val])
        test_score = model.decision_function(x_test[valid_test])
        rows.append(
            {
                "block": name,
                "task_id": task_id,
                "status": "ok",
                "n_train": int(valid_train.sum()),
                "n_val": int(valid_val.sum()),
                "n_test": int(valid_test.sum()),
                "val_auroc": f"{roc_auc_score(y_val[valid_val], val_score):.8g}",
                "val_auprc": f"{average_precision_score(y_val[valid_val], val_score):.8g}",
                "test_auroc": f"{roc_auc_score(y_test[valid_test], test_score):.8g}",
                "test_auprc": f"{average_precision_score(y_test[valid_test], test_score):.8g}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    records = read_csv(args.probe_features_dir / "records.csv")
    splits = np.array([row["split"] for row in records])
    task_ids, y = task_matrix(records, args.tasks_matrix)

    atlas = read_csv(args.probe_atlas_dir / "probe_peak_by_concept.csv")
    encoded_concepts = [row["concept_id"] for row in atlas if row["encoded"] == "yes"]
    all_concepts = [field for field in read_csv(args.concepts_matrix)[0].keys() if field != "ecg_id"]
    minimal_concepts = [cid for cid in MINIMAL_CONCEPTS if cid in all_concepts]

    rng = np.random.default_rng(args.seed)
    train_idx = np.where(splits == "train")[0]
    blocks: list[tuple[str, object, int]] = []
    fm_pooled = load_feature_manifest(args.probe_features_dir, "pooled")
    blocks.append(("FM_pooled", fm_pooled, int(fm_pooled.shape[1])))
    ball = clean_concept_features(concept_matrix(records, args.concepts_matrix, all_concepts), train_idx)
    blocks.append(("Ball_all_concepts", ball, len(all_concepts)))
    benc = clean_concept_features(concept_matrix(records, args.concepts_matrix, encoded_concepts), train_idx)
    blocks.append(("Benc_encoded_concepts", benc, len(encoded_concepts)))
    b0 = clean_concept_features(concept_matrix(records, args.concepts_matrix, minimal_concepts), train_idx)
    blocks.append(("B0_minimal_concepts", b0, len(minimal_concepts)))
    brand = rng.normal(size=(len(records), len(encoded_concepts))).astype(np.float32)
    blocks.append(("Brand_gaussian_dim_Benc", brand, len(encoded_concepts)))

    score_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for block_name, x, dim in blocks:
        block_rows.append({"block": block_name, "n_features": dim})
        score_rows.extend(evaluate_block(block_name, x, y, task_ids, splits, args.alpha))

    fields = ["block", "task_id", "status", "n_train", "n_val", "n_test", "val_auroc", "val_auprc", "test_auroc", "test_auprc"]
    write_csv(args.out_dir / "linear_task_scores.csv", score_rows, fields)
    write_csv(args.out_dir / "linear_blocks.csv", block_rows, ["block", "n_features"])

    report = {
        "probe_features_dir": str(args.probe_features_dir),
        "probe_atlas_dir": str(args.probe_atlas_dir),
        "out_dir": str(args.out_dir),
        "n_records": len(records),
        "task_ids": task_ids,
        "all_concepts": len(all_concepts),
        "encoded_concepts": len(encoded_concepts),
        "minimal_concepts": minimal_concepts,
        "blocks": block_rows,
        "alpha": args.alpha,
        "seed": args.seed,
    }
    (args.out_dir / "linear_task_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
