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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ridge probes from compact activation features to measurement concepts.")
    parser.add_argument("--probe-features-dir", required=True, type=Path)
    parser.add_argument("--concepts-matrix", default=ROOT / "results" / "manifest" / "concepts_matrix.csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=10.0)
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


def r2_score_np(y_true, y_pred) -> float:
    import numpy as np

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def robust_scale_train(y_train):
    import numpy as np

    med = np.nanmedian(y_train)
    q25, q75 = np.nanpercentile(y_train, [25, 75])
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = np.nanstd(y_train)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    return float(med), float(scale)


def main() -> None:
    args = parse_args()
    args.probe_features_dir = args.probe_features_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    records = read_csv(args.probe_features_dir / "records.csv")
    features = read_csv(args.probe_features_dir / "features.csv")
    concept_rows = read_csv(args.concepts_matrix)
    concept_by_id = {row["ecg_id"]: row for row in concept_rows}
    concept_ids = [field for field in concept_rows[0].keys() if field != "ecg_id"]
    ecg_ids = [row["ecg_id"] for row in records]
    splits = np.array([row["split"] for row in records])
    train_idx = np.where(splits == "train")[0]
    val_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]

    y_matrix = np.empty((len(records), len(concept_ids)), dtype=np.float32)
    for i, ecg_id in enumerate(ecg_ids):
        row = concept_by_id[ecg_id]
        y_matrix[i] = [parse_float(row[cid]) for cid in concept_ids]

    rng = np.random.default_rng(args.seed)
    score_rows: list[dict[str, str]] = []
    for feature in features:
        feature_name = feature["feature"]
        x_path = ROOT / feature["file"]
        x = np.load(x_path, mmap_mode="r")
        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.asarray(x[train_idx], dtype=np.float32))
        x_val = scaler.transform(np.asarray(x[val_idx], dtype=np.float32))
        x_test = scaler.transform(np.asarray(x[test_idx], dtype=np.float32))

        for j, concept_id in enumerate(concept_ids):
            y = y_matrix[:, j]
            valid_train = train_idx[np.isfinite(y[train_idx])]
            valid_val = val_idx[np.isfinite(y[val_idx])]
            valid_test = test_idx[np.isfinite(y[test_idx])]
            if len(valid_train) < 100 or len(valid_val) < 20 or len(valid_test) < 20:
                continue

            train_pos = np.searchsorted(train_idx, valid_train)
            val_pos = np.searchsorted(val_idx, valid_val)
            test_pos = np.searchsorted(test_idx, valid_test)
            med, scale = robust_scale_train(y[valid_train])
            y_train = (y[valid_train] - med) / scale
            y_val = (y[valid_val] - med) / scale
            y_test = (y[valid_test] - med) / scale

            model = Ridge(alpha=args.alpha)
            model.fit(x_train[train_pos], y_train)
            val_pred = model.predict(x_val[val_pos])
            test_pred = model.predict(x_test[test_pos])

            shuffled = np.array(y_train, copy=True)
            rng.shuffle(shuffled)
            shuffled_model = Ridge(alpha=args.alpha)
            shuffled_model.fit(x_train[train_pos], shuffled)
            shuffled_val_pred = shuffled_model.predict(x_val[val_pos])

            gaussian = rng.normal(size=len(y_train))
            gaussian_model = Ridge(alpha=args.alpha)
            gaussian_model.fit(x_train[train_pos], gaussian)
            gaussian_val_pred = gaussian_model.predict(x_val[val_pos])

            score_rows.append(
                {
                    "feature": feature_name,
                    "concept_id": concept_id,
                    "alpha": f"{args.alpha:g}",
                    "n_train": str(len(valid_train)),
                    "n_val": str(len(valid_val)),
                    "n_test": str(len(valid_test)),
                    "val_r2": f"{r2_score_np(y_val, val_pred):.8g}",
                    "test_r2": f"{r2_score_np(y_test, test_pred):.8g}",
                    "val_r2_shuffled": f"{r2_score_np(y_val, shuffled_val_pred):.8g}",
                    "val_r2_gaussian": f"{r2_score_np(y_val, gaussian_val_pred):.8g}",
                }
            )

    fields = [
        "feature",
        "concept_id",
        "alpha",
        "n_train",
        "n_val",
        "n_test",
        "val_r2",
        "test_r2",
        "val_r2_shuffled",
        "val_r2_gaussian",
    ]
    with (args.out_dir / "probe_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(score_rows)

    report = {
        "probe_features_dir": str(args.probe_features_dir),
        "out_dir": str(args.out_dir),
        "n_records": len(records),
        "n_features": len(features),
        "n_concepts": len(concept_ids),
        "n_scores": len(score_rows),
        "alpha": args.alpha,
        "seed": args.seed,
    }
    (args.out_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
