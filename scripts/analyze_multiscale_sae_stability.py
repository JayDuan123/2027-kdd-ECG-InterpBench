#!/usr/bin/env python
"""Measure cross-seed decoder stability over the complete layer-scale grid."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multiscale_sae import read_csv


SEEDS = (4311, 4312, 4313)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty stability table")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def load_decoder(checkpoint: Path) -> np.ndarray:
    import torch

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["model"]
    decoder = state["W_dec"].detach().cpu().numpy().astype(np.float32)
    decoder /= np.maximum(np.linalg.norm(decoder, axis=0, keepdims=True), 1e-12)
    del saved, state
    gc.collect()
    return decoder


def matched_cosines(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    similarity = a.T @ b
    row, column = linear_sum_assignment(-similarity)
    return similarity[row, column], similarity


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(a, mode="reduced")
    qb, _ = np.linalg.qr(b, mode="reduced")
    rank = max(1, min(qa.shape[1], qb.shape[1]))
    return float(np.square(qa.T @ qb).sum() / rank)


def normalized_log_auc(points: list[tuple[float, float]]) -> float:
    points = sorted((x, y) for x, y in points if x > 0 and np.isfinite(y))
    if len(points) < 2:
        return float("nan")
    x = np.log(np.asarray([point[0] for point in points]))
    y = np.asarray([point[1] for point in points])
    return float(np.trapz(y, x=x) / (x[-1] - x[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--top-features", type=int, default=256)
    parser.add_argument("--random-permutations", type=int, default=100)
    args = parser.parse_args()

    audit = json.loads((args.root / "audit.json").read_text())
    if not audit.get("audit_pass"):
        raise RuntimeError(f"multi-scale audit is not complete: {audit}")
    manifest = read_csv(args.root / "training_manifest.csv")
    grouped: dict[tuple[str, int, float, float, str], list[dict[str, str]]] = {}
    for row in manifest:
        key = (
            row["model"],
            int(row["layer"]),
            float(row["relative_depth"]),
            float(row["expansion_E"]),
            row["sparsity_arm"],
        )
        grouped.setdefault(key, []).append(row)

    pair_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        by_seed = {int(row["seed"]): row for row in rows}
        if set(by_seed) != set(SEEDS):
            raise RuntimeError(f"incomplete seed group {key}: {sorted(by_seed)}")
        decoders: dict[int, np.ndarray] = {}
        selected: dict[int, np.ndarray] = {}
        active_counts: dict[int, int] = {}
        for seed in SEEDS:
            row = by_seed[seed]
            firing = np.asarray(np.load(row["firing_rate"]), dtype=float)
            active = np.flatnonzero(firing > 0)
            active_counts[seed] = len(active)
            if len(active) == 0:
                raise RuntimeError(f"no active features for {key}, seed {seed}")
            selected[seed] = active[np.argsort(-firing[active])]
            decoders[seed] = load_decoder(Path(row["checkpoint"]))
        keep = min(args.top_features, *(active_counts[seed] for seed in SEEDS))
        model, layer, relative_depth, expansion, arm = key
        for seed_i, seed_j in itertools.combinations(SEEDS, 2):
            a = decoders[seed_i][:, selected[seed_i][:keep]]
            b = decoders[seed_j][:, selected[seed_j][:keep]]
            matched, similarity = matched_cosines(a, b)
            rng = np.random.default_rng(
                stable_seed(model, layer, relative_depth, expansion, arm, seed_i, seed_j)
            )
            floors = np.empty(args.random_permutations, dtype=float)
            for permutation_index in range(args.random_permutations):
                permutation = rng.permutation(keep)
                floors[permutation_index] = float(
                    similarity[np.arange(keep), permutation].mean()
                )
            pair_rows.append(
                {
                    "model": model,
                    "layer": layer,
                    "relative_depth": relative_depth,
                    "expansion_E": expansion,
                    "sparsity_arm": arm,
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "top_active_features": keep,
                    "active_features_i": active_counts[seed_i],
                    "active_features_j": active_counts[seed_j],
                    "matched_cosine_mean": float(matched.mean()),
                    "matched_cosine_median": float(np.median(matched)),
                    "matched_cosine_q10": float(np.quantile(matched, 0.10)),
                    "random_pairing_floor_mean": float(floors.mean()),
                    "stability_above_random": float(matched.mean() - floors.mean()),
                    "subspace_overlap": subspace_overlap(a, b),
                    "random_permutations": args.random_permutations,
                }
            )
        del decoders
        gc.collect()
        print(f"completed stability group {model}/L{layer}/E{expansion:g}", flush=True)

    write_csv(args.root / "stability_seed_pairs.csv", pair_rows)
    summaries = []
    summary_groups: dict[tuple[str, int, float, float], list[dict[str, Any]]] = {}
    for row in pair_rows:
        key = (row["model"], row["layer"], row["relative_depth"], row["expansion_E"])
        summary_groups.setdefault(key, []).append(row)
    for (model, layer, depth, expansion), rows in sorted(summary_groups.items()):
        summaries.append(
            {
                "model": model,
                "layer": layer,
                "relative_depth": depth,
                "expansion_E": expansion,
                "seed_pairs": len(rows),
                "matched_cosine_mean": float(np.mean([row["matched_cosine_mean"] for row in rows])),
                "stability_above_random_mean": float(
                    np.mean([row["stability_above_random"] for row in rows])
                ),
                "subspace_overlap_mean": float(np.mean([row["subspace_overlap"] for row in rows])),
            }
        )
    write_csv(args.root / "stability_layer_scale.csv", summaries)

    profiles = []
    for model in sorted({row["model"] for row in summaries}):
        model_rows = [row for row in summaries if row["model"] == model]
        expansion_points = []
        for expansion in sorted({row["expansion_E"] for row in model_rows}):
            rows = [row for row in model_rows if row["expansion_E"] == expansion]
            expansion_points.append(
                (
                    float(expansion),
                    float(np.mean([row["stability_above_random_mean"] for row in rows])),
                    float(np.mean([row["subspace_overlap_mean"] for row in rows])),
                )
            )
        profiles.append(
            {
                "model": model,
                "multiscale_stability_auc": normalized_log_auc(
                    [(point[0], point[1]) for point in expansion_points]
                ),
                "multiscale_subspace_auc": normalized_log_auc(
                    [(point[0], point[2]) for point in expansion_points]
                ),
                "n_layer_scale_cells": len(model_rows),
            }
        )
    write_csv(args.root / "stability_model_profiles.csv", profiles)
    metadata = {
        "status": "complete",
        "seed_pair_rows": len(pair_rows),
        "layer_scale_rows": len(summaries),
        "model_profiles": len(profiles),
        "top_active_features": args.top_features,
        "random_permutations": args.random_permutations,
    }
    (args.root / "stability_audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
