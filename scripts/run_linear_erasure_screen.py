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
    parser = argparse.ArgumentParser(description="Screen concept erasure effects in compact linear feature space.")
    parser.add_argument("--probe-features-dir", required=True, type=Path)
    parser.add_argument("--probe-atlas-dir", required=True, type=Path)
    parser.add_argument("--concepts-matrix", default=ROOT / "results" / "manifest" / "concepts_matrix.csv", type=Path)
    parser.add_argument("--tasks-matrix", default=ROOT / "results" / "manifest" / "tasks_matrix.csv", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--task-alpha", type=float, default=1.0)
    parser.add_argument("--concept-alpha", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_feature(probe_features_dir: Path, feature_name: str):
    import numpy as np

    for row in read_csv(probe_features_dir / "features.csv"):
        if row["feature"] == feature_name:
            return np.load(ROOT / row["file"], mmap_mode="r")
    raise ValueError(f"feature {feature_name!r} not found")


def matrix_from_csv(records: list[dict[str, str]], matrix_path: Path):
    import numpy as np

    rows_by_id = {row["ecg_id"]: row for row in read_csv(matrix_path)}
    fields = [field for field in next(iter(rows_by_id.values())).keys() if field != "ecg_id"]
    matrix = np.empty((len(records), len(fields)), dtype=np.float32)
    for i, record in enumerate(records):
        row = rows_by_id[record["ecg_id"]]
        matrix[i] = [parse_float(row[field]) for field in fields]
    return fields, matrix


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


def unit_vector_from_probe(x_train, y_train, alpha: float):
    import numpy as np
    from sklearn.linear_model import Ridge

    valid = np.isfinite(y_train)
    if valid.sum() < 100:
        return None
    med, scale = robust_scale_train(y_train[valid])
    y_scaled = (y_train[valid] - med) / scale
    probe = Ridge(alpha=alpha)
    probe.fit(x_train[valid], y_scaled)
    direction = np.asarray(probe.coef_, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return direction / norm


def erase_direction(x, direction):
    return x - (x @ direction)[:, None] * direction[None, :]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(args.seed)
    records = read_csv(args.probe_features_dir / "records.csv")
    splits = np.array([row["split"] for row in records])
    train_idx = np.where(splits == "train")[0]
    test_idx = np.where(splits == "test")[0]
    concept_ids, concept_values = matrix_from_csv(records, args.concepts_matrix)
    task_ids, task_values = matrix_from_csv(records, args.tasks_matrix)
    concept_col = {concept_id: i for i, concept_id in enumerate(concept_ids)}

    atlas = [
        row
        for row in read_csv(args.probe_atlas_dir / "probe_peak_by_concept.csv")
        if row.get("encoded") == "yes"
    ]
    concepts_by_feature: dict[str, list[dict[str, str]]] = {}
    for row in atlas:
        concepts_by_feature.setdefault(row["peak_feature"], []).append(row)

    screen_rows: list[dict[str, object]] = []
    feature_reports: list[dict[str, object]] = []
    for feature_name, concept_rows in sorted(concepts_by_feature.items()):
        raw_x = load_feature(args.probe_features_dir, feature_name)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.asarray(raw_x[train_idx], dtype=np.float32))
        x_test = scaler.transform(np.asarray(raw_x[test_idx], dtype=np.float32))
        feature_reports.append(
            {
                "feature": feature_name,
                "n_concepts": len(concept_rows),
                "feature_dim": int(x_train.shape[1]),
            }
        )

        task_models: dict[str, tuple[object, np.ndarray, np.ndarray]] = {}
        for task_j, task_id in enumerate(task_ids):
            y_train = task_values[train_idx, task_j]
            y_test = task_values[test_idx, task_j]
            valid_train = np.isfinite(y_train)
            valid_test = np.isfinite(y_test)
            train_classes = set(y_train[valid_train].astype(int).tolist())
            test_classes = set(y_test[valid_test].astype(int).tolist())
            if len(train_classes) < 2 or len(test_classes) < 2:
                continue
            model = RidgeClassifier(alpha=args.task_alpha)
            model.fit(x_train[valid_train], y_train[valid_train].astype(int))
            task_models[task_id] = (model, y_test, valid_test)

        for concept_row in concept_rows:
            concept_id = concept_row["concept_id"]
            cidx = concept_col[concept_id]
            direction = unit_vector_from_probe(
                x_train,
                concept_values[train_idx, cidx],
                alpha=args.concept_alpha,
            )
            if direction is None:
                continue
            random_direction = rng.normal(size=direction.shape[0]).astype(np.float32)
            random_direction = random_direction / np.linalg.norm(random_direction)
            x_test_erased = erase_direction(x_test, direction)
            x_test_random = erase_direction(x_test, random_direction)

            for task_id, (task_model, y_test, valid_test) in task_models.items():
                base_score = task_model.decision_function(x_test[valid_test])
                erased_score = task_model.decision_function(x_test_erased[valid_test])
                random_score = task_model.decision_function(x_test_random[valid_test])
                base_auroc = roc_auc_score(y_test[valid_test], base_score)
                erased_auroc = roc_auc_score(y_test[valid_test], erased_score)
                random_auroc = roc_auc_score(y_test[valid_test], random_score)
                base_auprc = average_precision_score(y_test[valid_test], base_score)
                erased_auprc = average_precision_score(y_test[valid_test], erased_score)
                random_auprc = average_precision_score(y_test[valid_test], random_score)
                screen_rows.append(
                    {
                        "concept_id": concept_id,
                        "family": concept_row["family"],
                        "feature": feature_name,
                        "task_id": task_id,
                        "base_auroc": f"{base_auroc:.8g}",
                        "erased_auroc": f"{erased_auroc:.8g}",
                        "random_auroc": f"{random_auroc:.8g}",
                        "delta_auroc": f"{base_auroc - erased_auroc:.8g}",
                        "delta_auroc_minus_random": f"{(base_auroc - erased_auroc) - (base_auroc - random_auroc):.8g}",
                        "base_auprc": f"{base_auprc:.8g}",
                        "erased_auprc": f"{erased_auprc:.8g}",
                        "random_auprc": f"{random_auprc:.8g}",
                        "delta_auprc": f"{base_auprc - erased_auprc:.8g}",
                    }
                )

    fields = [
        "concept_id",
        "family",
        "feature",
        "task_id",
        "base_auroc",
        "erased_auroc",
        "random_auroc",
        "delta_auroc",
        "delta_auroc_minus_random",
        "base_auprc",
        "erased_auprc",
        "random_auprc",
        "delta_auprc",
    ]
    write_csv(args.out_dir / "linear_erasure_screen.csv", screen_rows, fields)
    report = {
        "probe_features_dir": str(args.probe_features_dir),
        "probe_atlas_dir": str(args.probe_atlas_dir),
        "out_dir": str(args.out_dir),
        "n_records": len(records),
        "n_concepts": len(atlas),
        "n_rows": len(screen_rows),
        "features": feature_reports,
        "task_alpha": args.task_alpha,
        "concept_alpha": args.concept_alpha,
        "seed": args.seed,
        "note": "Linear feature-space erasure screen only; not full model continuation erasure.",
    }
    (args.out_dir / "linear_erasure_screen_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
