#!/usr/bin/env python
"""Run one joint-target SAE intervention from an existing shared code cache."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-random", type=int, default=20)
    return parser.parse_args()


def unique_union(rankings: dict[str, np.ndarray], members: list[str], k: int) -> np.ndarray:
    ordered = []
    seen = set()
    for rank_position in range(k):
        for member in members:
            atom = int(rankings[member][rank_position])
            if atom not in seen:
                seen.add(atom)
                ordered.append(atom)
    return np.asarray(ordered, dtype=int)


def matched_random_groups(
    selected: np.ndarray,
    excluded: np.ndarray,
    freq: np.ndarray,
    mag: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
) -> np.ndarray:
    groups = []
    for _ in range(n_random):
        group = []
        for atom in selected:
            distance = np.abs(np.log((freq + 1e-6) / (freq[atom] + 1e-6)))
            distance += np.abs(np.log((mag + 1e-6) / (mag[atom] + 1e-6)))
            blocked = np.unique(np.concatenate([excluded, np.asarray(group, dtype=int)]))
            distance[blocked] = np.inf
            pool = np.flatnonzero(np.isfinite(distance))
            if len(pool) == 0:
                raise RuntimeError("No atoms remain for a matched-random group")
            pool = pool[np.argsort(distance[pool])[: min(200, len(pool))]]
            group.append(int(rng.choice(pool)))
        groups.append(group)
    return np.asarray(groups, dtype=int)


def main() -> None:
    args = parse_args()
    manifest_path = args.base / "joint_steering/joint_steering_manifest.csv"
    row = pd.read_csv(manifest_path).iloc[args.group_index]
    model = str(row.model)
    members = json.loads(row.members_json)
    safe = model.lower().replace("-", "_")
    output = args.base / "joint_steering/tasks" / safe / f"seed{args.seed}" / str(row.group_id)
    final = output / "result.json"
    records = output / "records.npz"
    output.mkdir(parents=True, exist_ok=True)
    if final.exists() and records.exists():
        try:
            if json.loads(final.read_text()).get("schema_version") == 1:
                print(f"already complete: {output}")
                return
        except (OSError, json.JSONDecodeError):
            pass

    model_root = args.base / "models" / safe
    candidates = sorted((model_root / "shared_cache").glob(f"seed{args.seed}_N*_k*"))
    candidates = [path for path in candidates if (path / "complete.json").exists()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one complete shared cache for {model} seed {args.seed}: {candidates}")
    cache = candidates[0]
    heads = joblib.load(model_root / "frozen_heads.joblib")
    names = list(heads["targets"])
    missing = sorted(set(members) - set(names))
    if missing:
        raise RuntimeError(f"Joint members missing from frozen heads: {missing}")
    name_to_index = {name: i for i, name in enumerate(names)}
    registry = pd.read_csv(args.base / "target_registry.csv").set_index("target")
    offtargets = [
        name
        for name in names
        if name not in members and registry.loc[name, "analysis_role"] != "nuisance_control"
    ]
    ranking_matrix = np.load(cache / "rankings.npy", mmap_mode="r")
    rankings = {name: ranking_matrix[i] for i, name in enumerate(names)}
    zte = np.load(cache / "zte.npy", mmap_mode="r")
    gradients = np.load(cache / "gradients.npy", mmap_mode="r")
    centroid = np.load(cache / "centroid.npy", mmap_mode="r")
    freq = np.load(cache / "freq.npy", mmap_mode="r")
    mag = np.load(cache / "mag.npy", mmap_mode="r")
    base = np.load(cache / "base.npy", mmap_mode="r")
    thresholds = np.load(cache / "thresholds.npy", mmap_mode="r")
    focus_thresholds = json.loads((cache / "focus_thresholds.json").read_text())
    selected5 = unique_union(rankings, members, 5)
    selected10 = unique_union(rankings, members, 10)
    excluded = np.unique(np.concatenate([rankings[name][:10] for name in names])).astype(int)
    rng = np.random.default_rng(args.seed + sum(map(ord, model + str(row.group_id))))
    random5 = matched_random_groups(selected5, excluded, freq, mag, args.n_random, rng)
    random10 = matched_random_groups(selected10, excluded, freq, mag, args.n_random, rng)

    def delta(indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices, dtype=int)
        dz = centroid[idx][None, :] - zte[:, idx]
        return np.asarray(dz @ gradients[:, idx].T, dtype=np.float32)

    top5_delta = delta(selected5)
    top10_delta = delta(selected10)
    random5_delta = np.stack([delta(group) for group in random5], axis=1)
    random10_delta = np.stack([delta(group) for group in random10], axis=1)
    frame = pd.read_csv(args.base / "manifest.csv")
    test = frame.split.eq("test").to_numpy()
    labels = np.column_stack([np.asarray(heads["heads"][name]["labels"], dtype=float)[test] for name in names])
    kinds = [heads["heads"][name].get("type", "binary") for name in names]
    result = {
        "schema_version": 1,
        "model": model,
        "seed": args.seed,
        "group_id": str(row.group_id),
        "group_type": str(row.group_type),
        "family_scope": str(row.family_scope),
        "members": members,
        "member_indices": [name_to_index[name] for name in members],
        "offtarget_names": offtargets,
        "offtarget_indices": [name_to_index[name] for name in offtargets],
        "selected_atoms": {
            "top5_union": selected5.tolist(),
            "top10_union": selected10.tolist(),
        },
        "random_group_sizes": {
            "top5_union": int(len(selected5)),
            "top10_union": int(len(selected10)),
        },
        "focus_thresholds_train": focus_thresholds,
        "intervention": "joint_train_centroid_clamp_with_atom_deduplication",
        "guards": {
            "selection": "v2_main_headline_quality_qualified_tier2_3of3",
            "evaluation_split": "test",
            "matched_random_frequency_magnitude": True,
            "random_groups": args.n_random,
        },
    }
    tmp_json = final.with_suffix(f".json.tmp.{os.getpid()}")
    tmp_json.write_text(json.dumps(result, indent=2) + "\n")
    tmp_json.replace(final)
    target_means = np.asarray([heads["heads"][name].get("target_mean", np.nan) for name in names])
    target_stds = np.asarray([heads["heads"][name].get("target_std", np.nan) for name in names])
    tmp_npz = output / f"records.npz.tmp.{os.getpid()}"
    with tmp_npz.open("wb") as handle:
        np.savez(
            handle,
            patient_ids=frame.loc[test, "patient_id"].astype(str).to_numpy(dtype="U64"),
            target_names=np.asarray(names, dtype="U64"),
            target_types=np.asarray(kinds, dtype="U16"),
            labels=labels.astype(np.float32),
            baseline_logits=np.asarray(base, dtype=np.float32),
            top5_union_delta=top5_delta,
            top10_union_delta=top10_delta,
            random_top5_union_delta=random5_delta,
            random_top10_union_delta=random10_delta,
            thresholds_95spec=np.asarray(thresholds, dtype=np.float32),
            continuous_target_means=target_means,
            continuous_target_stds=target_stds,
        )
    tmp_npz.replace(records)
    print(
        json.dumps(
            {
                "model": model,
                "group": row.group_id,
                "seed": args.seed,
                "members": members,
                "top5_union_atoms": len(selected5),
                "top10_union_atoms": len(selected10),
            }
        )
    )


if __name__ == "__main__":
    main()
