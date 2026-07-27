#!/usr/bin/env python
"""Cross-capacity and target-functional SAE stability analyses."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_v2_common import V2, stable_seed, write_json  # noqa: E402


INVENTORY = (
    ROOT
    / "results"
    / "sae_extension"
    / "six_model_sae_audit"
    / "sae_2d_profile"
    / "checkpoint_inventory.csv"
)
EXTERNAL = ROOT / "results" / "external_benchmark_v1"
OUT = V2 / "sae_stability"
CAPACITIES = (8.0, 16.0, 32.0)
SEEDS = (4311, 4312, 4313)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--random-permutations", type=int, default=200)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def checkpoint_state(path: Path) -> dict:
    import torch

    loaded = torch.load(path, map_location="cpu", weights_only=False)
    if "sae" in loaded:
        return loaded["sae"]
    if "model" in loaded:
        return loaded["model"]
    if "state_dict" in loaded:
        return loaded["state_dict"]
    raise RuntimeError(f"Unsupported checkpoint schema: {path}")


def load_decoder(path: Path, raw_space: bool = False) -> np.ndarray:
    state = checkpoint_state(path)
    decoder = state["W_dec"].detach().cpu().numpy().astype(np.float32)
    if raw_space and "sigma" in state:
        sigma = state["sigma"].detach().cpu().numpy().astype(np.float32)
        decoder = decoder * sigma[:, None]
    decoder /= np.maximum(np.linalg.norm(decoder, axis=0, keepdims=True), 1e-12)
    del state
    gc.collect()
    return decoder


def load_firing(path: Path) -> np.ndarray:
    firing_path = path.with_name(path.stem + "_firing_rate.npy")
    if not firing_path.exists():
        raise FileNotFoundError(firing_path)
    return np.asarray(np.load(firing_path), dtype=float).reshape(-1)


def matched_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    similarity = a.T @ b
    row, column = linear_sum_assignment(-similarity)
    return similarity[row, column]


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    rank = max(1, min(qa.shape[1], qb.shape[1]))
    return float(np.square(qa.T @ qb).sum() / rank)


def capacity_stability(
    inventory: pd.DataFrame, top_features: int, random_permutations: int
) -> pd.DataFrame:
    rows = []
    selected = inventory[
        inventory.N_over_d.astype(float).isin(CAPACITIES)
        & inventory.firing_rate_present.astype(bool)
        & inventory.seed.astype(int).isin(SEEDS)
    ].copy()
    for (model, capacity), group in selected.groupby(["model", "N_over_d"], sort=True):
        by_seed = {int(row.seed): row for row in group.itertuples()}
        if set(by_seed) != set(SEEDS):
            raise RuntimeError(f"Incomplete seed grid for {model}/N_over_d={capacity}: {sorted(by_seed)}")
        decoders = {}
        indices = {}
        active_counts = []
        for seed, row in by_seed.items():
            path = Path(row.checkpoint)
            firing = load_firing(path)
            active = np.flatnonzero(firing > 0)
            active_counts.append(len(active))
            rank = active[np.argsort(-firing[active])]
            indices[seed] = rank
            decoders[seed] = load_decoder(path, raw_space=False)
        keep = min(top_features, *active_counts)
        for seed_i, seed_j in itertools.combinations(SEEDS, 2):
            a = decoders[seed_i][:, indices[seed_i][:keep]]
            b = decoders[seed_j][:, indices[seed_j][:keep]]
            matched = matched_cosines(a, b)
            similarity = a.T @ b
            rng = np.random.default_rng(
                stable_seed("capacity-stability", model, capacity, seed_i, seed_j)
            )
            floors = np.empty(random_permutations, dtype=float)
            for index in range(random_permutations):
                permutation = rng.permutation(keep)
                floors[index] = float(similarity[np.arange(keep), permutation].mean())
            rows.append(
                {
                    "model": model,
                    "N_over_d": float(capacity),
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "top_active_features": keep,
                    "matched_cosine_mean": float(matched.mean()),
                    "matched_cosine_median": float(np.median(matched)),
                    "matched_cosine_q10": float(np.quantile(matched, 0.1)),
                    "random_pairing_floor_mean": float(floors.mean()),
                    "stability_above_random": float(matched.mean() - floors.mean()),
                    "subspace_overlap": subspace_overlap(a, b),
                    "random_permutations": random_permutations,
                }
            )
        del decoders
        gc.collect()
    return pd.DataFrame(rows)


def discover_functional_results() -> pd.DataFrame:
    rows = []
    for path in sorted(EXTERNAL.glob("*/*/steering/*/seed*/*/result.json")):
        payload = json.loads(path.read_text())
        if not all(key in payload for key in ("checkpoint", "selected_atoms", "random_groups")):
            continue
        top5 = payload.get("selected_atoms", {}).get("top5", [])
        random_groups = payload.get("random_groups", [])
        if len(top5) != 5 or len(random_groups) < 20 or any(len(group) != 5 for group in random_groups[:20]):
            raise RuntimeError(f"Invalid selected/random groups: {path}")
        rows.append(
            {
                "model": payload["model"],
                "model_suffix": payload["model_suffix"],
                "cohort": payload["cohort"],
                "target": payload["target"],
                "protocol": payload["protocol"],
                "seed": int(payload["seed"]),
                "checkpoint": payload["checkpoint"],
                "top5": tuple(map(int, top5)),
                "random_groups": tuple(tuple(map(int, group)) for group in random_groups[:20]),
                "result_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def functional_stability(results: pd.DataFrame) -> pd.DataFrame:
    unit_columns = ["model", "model_suffix", "cohort", "target", "protocol"]
    groups = []
    for keys, group in results.groupby(unit_columns, sort=True):
        if set(group.seed) != set(SEEDS) or len(group) != len(SEEDS):
            raise RuntimeError(f"Incomplete functional seed unit {keys}: {sorted(group.seed)}")
        groups.append((keys, group.sort_values("seed")))

    checkpoint_paths = sorted(set(results.checkpoint))
    decoder_cache: dict[str, np.ndarray] = {}
    for index, checkpoint in enumerate(checkpoint_paths, start=1):
        decoder_cache[checkpoint] = load_decoder(Path(checkpoint), raw_space=True)
        print(f"loaded functional decoder {index}/{len(checkpoint_paths)}: {checkpoint}", flush=True)

    output = []
    for keys, group in groups:
        by_seed = {int(row.seed): row for row in group.itertuples()}
        for seed_i, seed_j in itertools.combinations(SEEDS, 2):
            row_i = by_seed[seed_i]
            row_j = by_seed[seed_j]
            decoder_i = decoder_cache[row_i.checkpoint]
            decoder_j = decoder_cache[row_j.checkpoint]
            selected_i = decoder_i[:, list(row_i.top5)]
            selected_j = decoder_j[:, list(row_j.top5)]
            selected_cosines = matched_cosines(selected_i, selected_j)
            selected_overlap = subspace_overlap(selected_i, selected_j)
            random_cosines = []
            random_overlaps = []
            for random_i, random_j in zip(row_i.random_groups, row_j.random_groups):
                a = decoder_i[:, list(random_i)]
                b = decoder_j[:, list(random_j)]
                random_cosines.append(float(matched_cosines(a, b).mean()))
                random_overlaps.append(subspace_overlap(a, b))
            random_cosines_array = np.asarray(random_cosines)
            random_overlaps_array = np.asarray(random_overlaps)
            output.append(
                {
                    **dict(zip(unit_columns, keys)),
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "selected_matched_cosine_mean": float(selected_cosines.mean()),
                    "selected_matched_cosine_min": float(selected_cosines.min()),
                    "random_matched_cosine_mean": float(random_cosines_array.mean()),
                    "selected_minus_random_cosine": float(
                        selected_cosines.mean() - random_cosines_array.mean()
                    ),
                    "selected_subspace_overlap": selected_overlap,
                    "random_subspace_overlap_mean": float(random_overlaps_array.mean()),
                    "random_subspace_overlap_q95": float(
                        np.quantile(random_overlaps_array, 0.95)
                    ),
                    "selected_minus_random_subspace_overlap": float(
                        selected_overlap - random_overlaps_array.mean()
                    ),
                    "selected_overlap_empirical_p": float(
                        (1.0 + (random_overlaps_array >= selected_overlap).sum())
                        / (len(random_overlaps_array) + 1.0)
                    ),
                    "random_groups": len(random_overlaps_array),
                }
            )
    del decoder_cache
    gc.collect()
    return pd.DataFrame(output)


def main() -> None:
    args = parse_args()
    inventory = pd.read_csv(INVENTORY)
    functional_results = discover_functional_results()
    capacity_grid = inventory[
        inventory.N_over_d.astype(float).isin(CAPACITIES)
        & inventory.firing_rate_present.astype(bool)
        & inventory.seed.astype(int).isin(SEEDS)
    ]
    functional_units = functional_results.groupby(
        ["model", "model_suffix", "cohort", "target", "protocol"]
    ).size()
    preflight = {
        "capacity_checkpoint_rows": len(capacity_grid),
        "capacity_models": int(capacity_grid.model.nunique()),
        "capacity_levels": sorted(capacity_grid.N_over_d.astype(float).unique().tolist()),
        "functional_result_rows": len(functional_results),
        "functional_units": len(functional_units),
        "functional_incomplete_units": int((functional_units != 3).sum()),
        "functional_unique_checkpoints": int(functional_results.checkpoint.nunique()),
    }
    if preflight["capacity_checkpoint_rows"] != 54:
        raise RuntimeError(f"Expected 54 capacity checkpoints, got {preflight}")
    if preflight["functional_incomplete_units"]:
        raise RuntimeError(f"Incomplete functional units: {preflight}")
    if args.preflight_only:
        print(preflight)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    capacity = capacity_stability(
        inventory, args.top_features, args.random_permutations
    )
    functional = functional_stability(functional_results)
    capacity_summary = (
        capacity.groupby(["model", "N_over_d"], as_index=False)
        .agg(
            seed_pairs=("seed_i", "size"),
            matched_cosine_mean=("matched_cosine_mean", "mean"),
            stability_above_random_mean=("stability_above_random", "mean"),
            subspace_overlap_mean=("subspace_overlap", "mean"),
        )
    )
    functional_summary = (
        functional.groupby(["model", "protocol"], as_index=False)
        .agg(
            seed_pair_cells=("seed_i", "size"),
            selected_matched_cosine_mean=("selected_matched_cosine_mean", "mean"),
            selected_minus_random_cosine_mean=("selected_minus_random_cosine", "mean"),
            selected_subspace_overlap_mean=("selected_subspace_overlap", "mean"),
            random_subspace_overlap_mean=("random_subspace_overlap_mean", "mean"),
            selected_minus_random_subspace_overlap_mean=(
                "selected_minus_random_subspace_overlap",
                "mean",
            ),
            selected_overlap_empirical_p_lt_0_05=(
                "selected_overlap_empirical_p",
                lambda values: int((values < 0.05).sum()),
            ),
        )
    )
    capacity.to_csv(args.out / "capacity_seed_pair_stability.csv", index=False)
    capacity_summary.to_csv(args.out / "capacity_stability_summary.csv", index=False)
    functional.to_csv(args.out / "functional_top5_seed_pair_stability.csv", index=False)
    functional_summary.to_csv(args.out / "functional_top5_stability_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        **preflight,
        "capacity_seed_pair_rows": len(capacity),
        "functional_seed_pair_rows": len(functional),
        "top_active_features": args.top_features,
        "capacity_random_permutations": args.random_permutations,
        "functional_random_control": "20 pre-existing frequency/magnitude-matched top-5 groups per seed",
        "new_sae_training": False,
        "all_complete": True,
    }
    write_json(args.out / "metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
