#!/usr/bin/env python3
"""Compare original and leakage-controlled CSFM L6 feature-count protocols."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.csfm_feature_count import (  # noqa: E402
    feature_count_summary,
    frozen_feature_profile,
    validate_original_scores,
)

DATA_ROOT = Path(
    "/rhf/allocations/wq8/yd68/csfm_embed/fullscale/fs_data"
)
DEFAULT_OUTPUT = ROOT / "results/csfm_l6_feature_count_v1"
EXPECTED = {"SAE": (8192, 3296), "Dense": (768, 585)}
COLORS = {"SAE": "#D55E00", "Dense": "#0072B2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-size", type=int, default=16384)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.70)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record(path: Path, *, hash_file: bool) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
        record.update(shape=list(array.shape), dtype=str(array.dtype))
    if hash_file:
        record["sha256"] = sha256(path)
    return record


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compute_frozen(
    method: str,
    train_path: Path,
    test_path: Path,
    y_train: np.ndarray,
    y_test: np.ndarray,
    targets: list[str],
    chunk_size: int,
    threshold: float,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    train = np.load(train_path, mmap_mode="r")
    test = np.load(test_path, mmap_mode="r")
    if train.shape[0] < y_train.shape[0] or test.shape[0] != y_test.shape[0]:
        raise ValueError(f"{method} feature and label row counts differ")

    scores = np.empty(train.shape[1], dtype=np.float32)
    rows: list[dict[str, object]] = []
    for start in range(0, train.shape[1], chunk_size):
        stop = min(start + chunk_size, train.shape[1])
        result = frozen_feature_profile(
            np.asarray(train[: y_train.shape[0], start:stop]),
            y_train,
            np.asarray(test[:, start:stop]),
            y_test,
        )
        scores[start:stop] = result.test_oriented_auc
        for local, feature in enumerate(range(start, stop)):
            target_index = int(result.selected_target[local])
            rows.append(
                {
                    "protocol": "frozen_train_test",
                    "method": method,
                    "feature_index": feature,
                    "selected_target": targets[target_index],
                    "direction": int(result.direction[local]),
                    "train_auc": float(result.train_auc[local]),
                    "train_oriented_auc": float(result.train_oriented_auc[local]),
                    "test_auc": float(result.test_auc[local]),
                    "test_oriented_auc": float(result.test_oriented_auc[local]),
                    "associated": int(result.test_oriented_auc[local] > threshold),
                }
            )
        print(f"[{method}] features {start}:{stop}/{train.shape[1]}", flush=True)
    return scores, rows


def make_figure(summary_rows: list[dict[str, object]], output: Path) -> None:
    protocols = ["original_test_selected", "frozen_train_test"]
    labels = ["Original\ntest-selected", "Frozen\ntrain to test"]
    methods = ["SAE", "Dense"]
    lookup = {(r["protocol"], r["method"]): r for r in summary_rows}
    x = np.arange(len(protocols), dtype=float)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))

    for offset, method in zip((-width / 2, width / 2), methods):
        counts = [lookup[(protocol, method)]["associated_count"] for protocol in protocols]
        fractions = [lookup[(protocol, method)]["associated_fraction"] for protocol in protocols]
        bars = axes[0].bar(x + offset, counts, width, color=COLORS[method], label=method)
        axes[0].bar_label(bars, padding=3, fontsize=9, fmt="%.0f")
        bars = axes[1].bar(x + offset, fractions, width, color=COLORS[method], label=method)
        axes[1].bar_label(bars, padding=3, fontsize=9, fmt="%.3f")

    axes[0].set_ylabel("Concept-associated features (AUROC > 0.70)")
    axes[0].set_title("Raw count")
    axes[1].set_ylabel("Fraction of available features")
    axes[1].set_title("Width-normalized fraction")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("CSFM final layer (L6): SAE atoms vs dense dimensions", fontsize=12)
    fig.tight_layout()
    fig.savefig(output / "csfm_l6_feature_count_protocols.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "csfm_l6_feature_count_protocols.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.train_size < 2 or args.chunk_size < 1:
        raise ValueError("train-size and chunk-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    probe = args.data_root / "probe"
    fig5 = args.data_root / "fig5"
    source_paths = {
        "original": args.data_root / "steer/atom_vs_dense.npz",
        "metadata": probe / "meta.json",
        "y_train": probe / "Y_train.npy",
        "y_test": probe / "Y_test.npy",
        "study_train": probe / "study_train.npy",
        "study_test": probe / "study_test.npy",
        "sae_train": probe / "X_train.npy",
        "sae_test": probe / "X_test.npy",
        "dense_train": fig5 / "csfm_mean_L6_train.npy",
        "dense_test": fig5 / "csfm_mean_L6_test.npy",
    }
    metadata = json.loads(source_paths["metadata"].read_text())
    targets = list(metadata["targets"])
    y_train_all = np.load(source_paths["y_train"], mmap_mode="r")
    y_test = np.asarray(np.load(source_paths["y_test"], mmap_mode="r"))
    if args.train_size > y_train_all.shape[0]:
        raise ValueError("train-size exceeds available training rows")
    y_train = np.asarray(y_train_all[: args.train_size])
    if np.any(y_train.sum(axis=0) == 0) or np.any(y_train.sum(axis=0) == len(y_train)):
        raise ValueError("training prefix must contain both classes for every target")

    original_npz = np.load(source_paths["original"])
    original_scores = {
        "SAE": validate_original_scores(
            original_npz["L6_atom"], expected_width=8192,
            expected_count=3296, threshold=args.threshold,
        ),
        "Dense": validate_original_scores(
            original_npz["L6_dense"], expected_width=768,
            expected_count=585, threshold=args.threshold,
        ),
    }
    paths = {
        "SAE": (source_paths["sae_train"], source_paths["sae_test"]),
        "Dense": (source_paths["dense_train"], source_paths["dense_test"]),
    }
    frozen_scores: dict[str, np.ndarray] = {}
    feature_rows: list[dict[str, object]] = []
    for method in ("SAE", "Dense"):
        train_path, test_path = paths[method]
        frozen_scores[method], rows = compute_frozen(
            method, train_path, test_path, y_train, y_test, targets,
            args.chunk_size, args.threshold,
        )
        feature_rows.extend(rows)

    summary_rows: list[dict[str, object]] = []
    for protocol, values_by_method in (
        ("original_test_selected", original_scores),
        ("frozen_train_test", frozen_scores),
    ):
        for method in ("SAE", "Dense"):
            summary = feature_count_summary(values_by_method[method], args.threshold)
            summary_rows.append(
                {"protocol": protocol, "method": method, **summary}
            )

    write_csv(
        args.output_dir / "summary.csv",
        summary_rows,
        ["protocol", "method", "width", "associated_count", "associated_fraction", "max_score"],
    )
    write_csv(
        args.output_dir / "frozen_feature_profiles.csv",
        feature_rows,
        ["protocol", "method", "feature_index", "selected_target", "direction",
         "train_auc", "train_oriented_auc", "test_auc", "test_oriented_auc", "associated"],
    )
    np.savez_compressed(
        args.output_dir / "score_arrays.npz",
        original_sae=original_scores["SAE"], original_dense=original_scores["Dense"],
        frozen_sae=frozen_scores["SAE"], frozen_dense=frozen_scores["Dense"],
    )
    make_figure(summary_rows, args.output_dir)

    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "layer": "CSFM paper L6 (final transformer layer)",
        "threshold": {"operator": ">", "value": args.threshold},
        "train_selection": {
            "rows": args.train_size,
            "rule": "deterministic first N rows of existing subject-disjoint training split",
            "target_and_orientation_selected_on": "train",
            "evaluated_on": "complete fixed test split",
        },
        "pooling": {"SAE": "token max", "Dense": "token mean", "matched": False},
        "targets": targets,
        "expected_original": {
            method: {"width": width, "count_gt_0.70": count}
            for method, (width, count) in EXPECTED.items()
        },
        "summary": summary_rows,
        "sources": {
            name: source_record(path, hash_file=name in {
                "original", "metadata", "y_train", "y_test", "study_train", "study_test"
            })
            for name, path in source_paths.items()
        },
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    original_sae = summary_rows[0]
    original_dense = summary_rows[1]
    frozen_sae = summary_rows[2]
    frozen_dense = summary_rows[3]
    report = f"""# CSFM L6 concept-associated feature count audit

Threshold: strict best/frozen oriented AUROC > {args.threshold:.2f}. The concept panel contains {len(targets)} labels.

| Protocol | Method | Raw count | Width | Fraction |
|---|---:|---:|---:|---:|
| Original test-selected | SAE | {original_sae['associated_count']} | {original_sae['width']} | {original_sae['associated_fraction']:.4f} |
| Original test-selected | Dense | {original_dense['associated_count']} | {original_dense['width']} | {original_dense['associated_fraction']:.4f} |
| Frozen train to test | SAE | {frozen_sae['associated_count']} | {frozen_sae['width']} | {frozen_sae['associated_fraction']:.4f} |
| Frozen train to test | Dense | {frozen_dense['associated_count']} | {frozen_dense['width']} | {frozen_dense['associated_fraction']:.4f} |

## Interpretation

The original panel selects the best concept separately for every feature on the test set and folds the AUROC direction on that same test set. It exactly reproduces the saved CSFM L6 counts (SAE 3296; dense 585). Raw counts are not capacity-normalized: the SAE dictionary has 8192 atoms, whereas the dense representation has 768 dimensions.

The corrected protocol selects both concept and direction on a deterministic {args.train_size}-record prefix of the existing subject-disjoint training split, then freezes both choices on all {len(y_test)} test records. It uses tie-aware AUROC. This removes test-set feature selection leakage.

## Limitation

This audit preserves the original activation files to isolate the selection-protocol effect. SAE atoms use token-max pooling while dense dimensions use token-mean pooling, so the comparison is not pooling-matched and should not be interpreted as a causal advantage of the SAE transformation.
"""
    (args.output_dir / "report.md").write_text(report)
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
