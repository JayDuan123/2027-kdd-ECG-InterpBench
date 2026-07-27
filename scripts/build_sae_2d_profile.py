#!/usr/bin/env python
"""Build the 2D SAE interpretability profile.

The profile follows the locked outline in
`/home/yd68/.codex/attachments/359b9c36-f6e2-4108-8ecf-f5c86cd7d1c1/pasted-text-1.txt`.

This script is intentionally conservative. It computes X from the available
reconstruction grid, but it refuses to fabricate Y when the required
cross-seed decoder dictionaries and firing-rate vectors are not present.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECON_CSV = (
    ROOT
    / "results"
    / "sae_extension"
    / "six_model_sae_audit"
    / "phase0_recon_curves_combined.csv"
)
DEFAULT_CHECKPOINT_ROOT = ROOT / "results" / "sae_extension" / "six_model_sae_audit"
DEFAULT_OUT_DIR = ROOT / "results" / "sae_extension" / "six_model_sae_audit" / "sae_2d_profile"
EXPECTED_MODELS = ["CARDIAC-FM", "CSFM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]


def parse_float(value: Any, default: float = float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def infer_seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.search(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
    return 4311


def discover_recon_rows(recon_csv: Path, checkpoint_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen_paths: set[Path] = set()
    if recon_csv.exists():
        seen_paths.add(recon_csv.resolve())
        for row in read_csv(recon_csv):
            row = dict(row)
            row.setdefault("source_recon_csv", str(recon_csv))
            rows.append(row)
    for path in sorted(checkpoint_root.rglob("sae_recon_curve.csv")):
        resolved = path.resolve()
        if resolved in seen_paths:
            continue
        seed = infer_seed_from_path(path)
        for row in read_csv(path):
            row = dict(row)
            row.setdefault("source_recon_csv", str(path))
            if not row.get("sae_seed") and not row.get("seed"):
                row["sae_seed"] = str(seed)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def ffmt(value: float, digits: int = 8) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{digits}g}"


def canonical_e(value: float) -> float:
    return float(round(float(value), 12))


@dataclass(frozen=True)
class ReconPoint:
    model: str
    seed: int
    e_value: float
    n_capacity: float
    d_hidden: float
    recon_r2: float
    dead_frac: float
    active_feature_count: float
    n_source_rows: int


@dataclass(frozen=True)
class CheckpointArtifact:
    model: str
    seed: int
    n_over_d: float
    n_capacity: int
    d_hidden: int
    k: int
    checkpoint: Path
    firing_rate_path: Path | None
    recon_r2: float
    dead_frac: float


def aggregate_recon_points(rows: list[dict[str, str]]) -> list[ReconPoint]:
    grouped: dict[tuple[str, int, float], list[dict[str, str]]] = {}
    for row in rows:
        model = row.get("model", "").strip()
        if not model:
            continue
        e_value = parse_float(row.get("E"))
        recon = parse_float(row.get("recon_R2", row.get("recon_r2")))
        n_capacity = parse_float(row.get("N_capacity"))
        if not (np.isfinite(e_value) and e_value > 0 and np.isfinite(recon)):
            continue
        e_value = canonical_e(e_value)
        seed = parse_int(row.get("sae_seed", row.get("seed", 4311)), 4311)
        grouped.setdefault((model, seed, e_value), []).append(row)

    points: list[ReconPoint] = []
    for (model, seed, e_value), part in sorted(grouped.items()):
        recons = np.array([parse_float(r.get("recon_R2", r.get("recon_r2"))) for r in part], dtype=float)
        n_vals = np.array([parse_float(r.get("N_capacity")) for r in part], dtype=float)
        dead = np.array([parse_float(r.get("dead_frac")) for r in part], dtype=float)
        n_capacity = float(np.nanmedian(n_vals))
        d_hidden = n_capacity / e_value if e_value > 0 else float("nan")
        dead_mean = float(np.nanmean(dead)) if len(dead) else float("nan")
        active = n_capacity * (1.0 - dead_mean) if np.isfinite(dead_mean) else float("nan")
        points.append(
            ReconPoint(
                model=model,
                seed=seed,
                e_value=float(e_value),
                n_capacity=n_capacity,
                d_hidden=d_hidden,
                recon_r2=float(np.nanmean(recons)),
                dead_frac=dead_mean,
                active_feature_count=active,
                n_source_rows=len(part),
            )
        )
    return points


def choose_x_metric(points: list[ReconPoint]) -> tuple[str, list[float], dict[str, list[float]], dict[str, Any]]:
    by_model: dict[str, set[float]] = {}
    by_model_seed: dict[tuple[str, int], set[float]] = {}
    for p in points:
        by_model.setdefault(p.model, set()).add(p.e_value)
        by_model_seed.setdefault((p.model, p.seed), set()).add(p.e_value)
    model_e = {m: sorted(v) for m, v in by_model.items()}

    seeds_by_model = {
        model: {seed for (m, seed) in by_model_seed if m == model}
        for model in EXPECTED_MODELS
    }
    if all(seeds_by_model.get(model) for model in EXPECTED_MODELS):
        common_seeds = set.intersection(*(seeds_by_model[model] for model in EXPECTED_MODELS))
    else:
        common_seeds = set()

    x_support_mode = "model_level"
    common: set[float] = set()
    if common_seeds:
        x_support_mode = "model_seed_level"
        support_sets = [
            by_model_seed[(model, seed)]
            for model in EXPECTED_MODELS
            for seed in sorted(common_seeds)
            if (model, seed) in by_model_seed
        ]
        common = set.intersection(*support_sets) if support_sets else set()
    if len(common) < 3 and len(model_e) >= 6:
        x_support_mode = "model_level"
        common = set.intersection(*(set(model_e[m]) for m in EXPECTED_MODELS if m in model_e))
    common_e = sorted(common)
    x_metric = "X2" if len(common_e) >= 3 else "X1"
    return x_metric, common_e, model_e, {
        "x_support_mode": x_support_mode,
        "common_seeds": sorted(common_seeds),
        "seeds_by_model": {m: sorted(v) for m, v in seeds_by_model.items()},
    }


def compute_x2(points: list[ReconPoint], common_e: list[float]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], float]]:
    if len(common_e) < 3:
        return [], {}
    grid = np.exp(np.linspace(np.log(min(common_e)), np.log(max(common_e)), 50))
    point_by_key = {(p.model, p.seed, p.e_value): p for p in points}
    profile_rows: list[dict[str, Any]] = []
    x_values: dict[tuple[str, int], float] = {}
    for p in points:
        profile_rows.append(
            {
                "row_type": "observed",
                "model": p.model,
                "seed": p.seed,
                "d_hidden": ffmt(p.d_hidden),
                "E": ffmt(p.e_value),
                "N": ffmt(p.n_capacity),
                "N_over_d": ffmt(p.e_value),
                "recon_R2": ffmt(p.recon_r2),
                "on_common_grid": str(p.e_value in common_e).lower(),
                "interpolated_recon_R2": "",
                "n_source_rows": p.n_source_rows,
            }
        )

    model_seed_keys = sorted({(p.model, p.seed) for p in points})
    for model, seed in model_seed_keys:
        support = []
        for e in common_e:
            p = point_by_key.get((model, seed, e))
            if p is not None:
                support.append((e, p.recon_r2, p.d_hidden, p.n_capacity))
        if len(support) != len(common_e):
            continue
        e_arr = np.array([x[0] for x in support], dtype=float)
        r_arr = np.array([x[1] for x in support], dtype=float)
        interp = np.interp(np.log(grid), np.log(e_arr), r_arr)
        auc = float(np.trapz(interp, x=np.log(grid)) / (np.log(grid[-1]) - np.log(grid[0])))
        x_values[(model, seed)] = auc
        d_hidden = float(np.nanmedian([x[2] for x in support]))
        for e, recon in zip(grid, interp):
            profile_rows.append(
                {
                    "row_type": "interpolated",
                    "model": model,
                    "seed": seed,
                    "d_hidden": ffmt(d_hidden),
                    "E": ffmt(float(e)),
                    "N": ffmt(float(e) * d_hidden),
                    "N_over_d": ffmt(float(e)),
                    "recon_R2": "",
                    "on_common_grid": "true",
                    "interpolated_recon_R2": ffmt(float(recon)),
                    "n_source_rows": "",
                }
            )
    return profile_rows, x_values


def compute_x1(points: list[ReconPoint]) -> tuple[list[dict[str, Any]], dict[tuple[str, int], float]]:
    profile_rows: list[dict[str, Any]] = []
    x_values: dict[tuple[str, int], float] = {}
    by_model_seed: dict[tuple[str, int], list[ReconPoint]] = {}
    for p in points:
        by_model_seed.setdefault((p.model, p.seed), []).append(p)
        profile_rows.append(
            {
                "row_type": "observed",
                "model": p.model,
                "seed": p.seed,
                "d_hidden": ffmt(p.d_hidden),
                "E": ffmt(p.e_value),
                "N": ffmt(p.n_capacity),
                "N_over_d": ffmt(p.e_value),
                "recon_R2": ffmt(p.recon_r2),
                "on_common_grid": "false",
                "interpolated_recon_R2": "",
                "n_source_rows": p.n_source_rows,
            }
        )
    for key, part in by_model_seed.items():
        in_band = [p for p in part if 0.90 <= p.recon_r2 <= 0.92]
        if not in_band:
            continue
        best = min(in_band, key=lambda p: p.e_value)
        x_values[key] = 1.0 / best.e_value
    return profile_rows, x_values


def parent_metric_json(path: Path) -> dict[str, Any]:
    for parent in [path.parent.parent, path.parent.parent.parent, path.parent.parent.parent.parent]:
        candidate = parent / "metrics_stage.json"
        if candidate.exists():
            try:
                return json.loads(candidate.read_text())
            except json.JSONDecodeError:
                return {}
    return {}


def parent_recon_curve_row(path: Path) -> dict[str, str]:
    for parent in [path.parent.parent, path.parent.parent.parent, path.parent.parent.parent.parent]:
        candidate = parent / "sae_recon_curve.csv"
        if candidate.exists() and candidate.stat().st_size > 1:
            try:
                rows = read_csv(candidate)
            except Exception:
                return {}
            return rows[0] if rows else {}
    return {}


def find_firing_rate(checkpoint: Path) -> Path | None:
    candidates = [
        checkpoint.with_name(checkpoint.stem + "_firing_rate.npy"),
        checkpoint.parent / "firing_rate.npy",
        checkpoint.parent.parent / "firing_rate.npy",
        checkpoint.parent / "firing_rate.csv",
        checkpoint.parent.parent / "firing_rate.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def collect_checkpoints(root: Path) -> list[CheckpointArtifact]:
    import torch

    artifacts: list[CheckpointArtifact] = []
    for checkpoint in sorted(root.rglob("checkpoints/*.pt")):
        metrics = parent_metric_json(checkpoint)
        recon_curve = parent_recon_curve_row(checkpoint)
        model = str(metrics.get("model", recon_curve.get("model", ""))).strip()
        if not model:
            continue
        try:
            loaded = torch.load(checkpoint, map_location="cpu")
        except Exception:
            continue
        meta = loaded.get("meta", {}) if isinstance(loaded, dict) else {}
        state = loaded.get("sae", {}) if isinstance(loaded, dict) else {}
        d_hidden = parse_int(meta.get("d"), 0)
        n_capacity = parse_int(meta.get("n_features"), 0)
        k = parse_int(meta.get("k"), 0)
        seed = parse_int(meta.get("seed", metrics.get("sae_seed", 0)), 0)
        if not d_hidden and "W_dec" in state:
            d_hidden = int(state["W_dec"].shape[0])
        if not n_capacity and "W_dec" in state:
            n_capacity = int(state["W_dec"].shape[1])
        if not (d_hidden > 0 and n_capacity > 0 and seed > 0):
            continue
        n_over_d = parse_float(metrics.get("E", recon_curve.get("E")), float(n_capacity) / float(d_hidden))
        if not (np.isfinite(n_over_d) and n_over_d > 0):
            n_over_d = float(n_capacity) / float(d_hidden)
        n_over_d = canonical_e(n_over_d)
        artifacts.append(
            CheckpointArtifact(
                model=model,
                seed=seed,
                n_over_d=float(n_over_d),
                n_capacity=n_capacity,
                d_hidden=d_hidden,
                k=k,
                checkpoint=checkpoint,
                firing_rate_path=find_firing_rate(checkpoint),
                recon_r2=parse_float(metrics.get("recon_R2", metrics.get("recon_r2", recon_curve.get("recon_R2", recon_curve.get("recon_r2"))))),
                dead_frac=parse_float(metrics.get("dead_frac", recon_curve.get("dead_frac"))),
            )
        )
    return artifacts


def choose_y_point(artifacts: list[CheckpointArtifact]) -> tuple[float | None, dict[str, str], dict[str, dict[str, Any]]]:
    def artifact_priority(artifact: CheckpointArtifact) -> tuple[int, int, float, str]:
        recon_ok = bool(np.isfinite(artifact.recon_r2) and artifact.recon_r2 >= 0.85)
        firing_ok = artifact.firing_rate_path is not None
        dead = artifact.dead_frac if np.isfinite(artifact.dead_frac) else float("inf")
        return (0 if recon_ok else 1, 0 if firing_ok else 1, dead, str(artifact.checkpoint))

    preflight: dict[str, dict[str, Any]] = {}
    deduped: dict[tuple[str, float, int], CheckpointArtifact] = {}
    for artifact in artifacts:
        key = (artifact.model, artifact.n_over_d, artifact.seed)
        current = deduped.get(key)
        if current is None or artifact_priority(artifact) < artifact_priority(current):
            deduped[key] = artifact
    by_model: dict[str, list[CheckpointArtifact]] = {}
    for artifact in deduped.values():
        by_model.setdefault(artifact.model, []).append(artifact)

    for candidate in [8.0, 16.0, 32.0]:
        ok = True
        details: dict[str, str] = {}
        for model in EXPECTED_MODELS:
            rows = [a for a in by_model.get(model, []) if math.isclose(a.n_over_d, candidate, rel_tol=1e-9, abs_tol=1e-9)]
            seeds = sorted({a.seed for a in rows})
            recon_ok = bool(rows) and all(np.isfinite(a.recon_r2) and a.recon_r2 >= 0.85 for a in rows)
            if not rows:
                details[model] = "missing_candidate_checkpoint"
                ok = False
            elif len(seeds) < 3:
                firing = sum(1 for a in rows if a.firing_rate_path is not None)
                details[model] = f"insufficient_seeds:{seeds}; firing_rate_files:{firing}/{len(rows)}"
                ok = False
            elif not recon_ok:
                details[model] = "recon_below_0.85_or_missing_recon"
                ok = False
            elif any(a.firing_rate_path is None for a in rows):
                details[model] = "missing_firing_rate"
                ok = False
            else:
                details[model] = "ok"
        preflight[f"candidate_{int(candidate)}"] = details
        if ok:
            return candidate, details, preflight
    return None, {}, preflight


def checkpoint_inventory_rows(artifacts: list[CheckpointArtifact]) -> list[dict[str, Any]]:
    rows = []
    for artifact in sorted(artifacts, key=lambda a: (a.model, a.n_over_d, a.seed, str(a.checkpoint))):
        rows.append(
            {
                "model": artifact.model,
                "seed": artifact.seed,
                "N_over_d": ffmt(artifact.n_over_d),
                "N": artifact.n_capacity,
                "d_hidden": artifact.d_hidden,
                "k": artifact.k,
                "recon_R2": ffmt(artifact.recon_r2),
                "dead_frac": ffmt(artifact.dead_frac),
                "firing_rate_present": str(artifact.firing_rate_path is not None).lower(),
                "checkpoint": str(artifact.checkpoint),
            }
        )
    return rows


def load_decoder_and_firing(path: Path, firing_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    loaded = torch.load(path, map_location="cpu")
    state = loaded["sae"]
    w_dec = state["W_dec"].detach().cpu().numpy().astype(np.float64)
    norms = np.linalg.norm(w_dec, axis=0, keepdims=True)
    w_dec = w_dec / np.clip(norms, 1e-12, None)
    if firing_path.suffix == ".npy":
        firing = np.load(firing_path).astype(np.float64)
    else:
        firing = np.loadtxt(firing_path, delimiter=",", dtype=np.float64)
    firing = np.ravel(firing)
    return w_dec, firing


def compute_y(
    artifacts: list[CheckpointArtifact],
    y_point: float,
    random_permutations: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    from scipy.optimize import linear_sum_assignment

    def artifact_priority(artifact: CheckpointArtifact) -> tuple[int, int, float, str]:
        recon_ok = bool(np.isfinite(artifact.recon_r2) and artifact.recon_r2 >= 0.85)
        firing_ok = artifact.firing_rate_path is not None
        dead = artifact.dead_frac if np.isfinite(artifact.dead_frac) else float("inf")
        return (0 if recon_ok else 1, 0 if firing_ok else 1, dead, str(artifact.checkpoint))

    by_key: dict[tuple[str, int], CheckpointArtifact] = {}
    for artifact in artifacts:
        if math.isclose(artifact.n_over_d, y_point, rel_tol=1e-9, abs_tol=1e-9):
            key = (artifact.model, artifact.seed)
            current = by_key.get(key)
            if current is None or artifact_priority(artifact) < artifact_priority(current):
                by_key[key] = artifact

    by_model: dict[str, list[CheckpointArtifact]] = {}
    for artifact in by_key.values():
        by_model.setdefault(artifact.model, []).append(artifact)

    active_counts = []
    loaded_by_key: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, CheckpointArtifact]] = {}
    for model, rows in by_model.items():
        for artifact in rows:
            if artifact.firing_rate_path is None:
                continue
            w_dec, firing = load_decoder_and_firing(artifact.checkpoint, artifact.firing_rate_path)
            active_counts.append(int((firing > 0).sum()))
            loaded_by_key[(model, artifact.seed)] = (w_dec, firing, artifact)
    k_top = min(512, min(active_counts)) if active_counts else 0
    out: list[dict[str, Any]] = []
    if k_top <= 0:
        return out

    for model in sorted(by_model):
        seeds = sorted({a.seed for a in by_model[model]})
        for seed_i, seed_j in itertools.combinations(seeds, 2):
            if (model, seed_i) not in loaded_by_key or (model, seed_j) not in loaded_by_key:
                continue
            w_i, firing_i, art_i = loaded_by_key[(model, seed_i)]
            w_j, firing_j, art_j = loaded_by_key[(model, seed_j)]
            idx_i = np.argsort(-firing_i)[:k_top]
            idx_j = np.argsort(-firing_j)[:k_top]
            d_i = w_i[:, idx_i]
            d_j = w_j[:, idx_j]
            c = d_i.T @ d_j
            row_ind, col_ind = linear_sum_assignment(-c)
            matched = float(np.mean(c[row_ind, col_ind]))
            floors = []
            for _ in range(random_permutations):
                perm = rng.permutation(k_top)
                floors.append(float(np.mean(c[np.arange(k_top), perm])))
            floor = float(np.mean(floors))
            out.append(
                {
                    "status": "computed",
                    "model": model,
                    "N_over_d_fixed": ffmt(y_point),
                    "topk_K": k_top,
                    "seed_i": seed_i,
                    "seed_j": seed_j,
                    "matched_cosine": ffmt(matched),
                    "random_matched_cosine_floor": ffmt(floor),
                    "stability_above_random": ffmt(matched - floor),
                    "dead_frac": ffmt(float(np.nanmean([art_i.dead_frac, art_j.dead_frac]))),
                    "active_feature_count": k_top,
                    "reason": "",
                }
            )
    return out


def seed_ci(values: list[float]) -> float:
    vals = [v for v in values if np.isfinite(v)]
    if len(vals) <= 1:
        return float("nan")
    return float(1.96 * np.std(vals, ddof=1) / math.sqrt(len(vals)))


def aggregate_profile(
    x_metric: str,
    x_values: dict[tuple[str, int], float],
    y_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    x_by_model: dict[str, list[float]] = {}
    for (model, _seed), value in x_values.items():
        x_by_model.setdefault(model, []).append(value)
    y_by_model: dict[str, list[float]] = {}
    for row in y_rows:
        if row.get("status") != "computed":
            continue
        y_by_model.setdefault(row["model"], []).append(parse_float(row.get("stability_above_random")))

    x_means = {m: float(np.mean(v)) for m, v in x_by_model.items() if v}
    y_means = {m: float(np.mean(v)) for m, v in y_by_model.items() if v}
    between_x = float(np.nanstd(list(x_means.values()), ddof=1)) if len(x_means) > 1 else float("nan")
    between_y = float(np.nanstd(list(y_means.values()), ddof=1)) if len(y_means) > 1 else float("nan")
    within_x_vals = [seed_ci(v) for v in x_by_model.values() if len(v) > 1]
    within_y_vals = [seed_ci(v) for v in y_by_model.values() if len(v) > 1]
    within_x = float(np.nanmean(within_x_vals)) if within_x_vals else float("nan")
    within_y = float(np.nanmean(within_y_vals)) if within_y_vals else float("nan")

    rows: list[dict[str, Any]] = []
    for model in EXPECTED_MODELS:
        xv = x_by_model.get(model, [])
        yv = y_by_model.get(model, [])
        rows.append(
            {
                "model": model,
                "X_metric": x_metric,
                "X_value": ffmt(float(np.mean(xv)) if xv else float("nan")),
                "X_seed_CI": ffmt(seed_ci(xv)),
                "Y_value": ffmt(float(np.mean(yv)) if yv else float("nan")),
                "Y_seed_CI": ffmt(seed_ci(yv)),
                "between_model_spread_X": ffmt(between_x),
                "within_model_var_X": ffmt(within_x),
                "between_model_spread_Y": ffmt(between_y),
                "within_model_var_Y": ffmt(within_y),
                "profile_status": "computed_xy" if xv and yv else ("x_only_y_unavailable" if xv else "missing_x"),
            }
        )
    spread = {
        "between_model_spread_X": between_x,
        "within_model_var_X": within_x,
        "between_model_spread_Y": between_y,
        "within_model_var_Y": within_y,
    }
    return rows, spread


def make_plot(profile_rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [parse_float(r["X_value"]) for r in profile_rows]
    ys = [parse_float(r["Y_value"]) for r in profile_rows]
    y_available = any(np.isfinite(y) for y in ys)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row, x, y in zip(profile_rows, xs, ys):
        if not np.isfinite(x):
            continue
        if y_available and np.isfinite(y):
            ax.scatter(x, y, s=70)
            ax.text(x, y, " " + row["model"], fontsize=8, va="center")
        else:
            ax.scatter(x, 0.0, s=70, color="#7f8c8d")
            ax.text(x, 0.0, " " + row["model"], fontsize=8, va="center")
    ax.set_xlabel("X: capacity-normalized sparse reconstructibility")
    if y_available:
        ax.set_ylabel("Y: dictionary stability above real-direction random floor")
    else:
        ax.set_ylabel("Y unavailable in current artifacts")
        ax.set_yticks([0.0])
        ax.set_yticklabels(["missing seed/firing-rate inputs"])
    ax.set_title("2D SAE Interpretability Profile")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def render_report(
    x_metric: str,
    common_e: list[float],
    model_e: dict[str, list[float]],
    y_point: float | None,
    y_preflight: dict[str, dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    spread: dict[str, float],
    profile_rows: list[dict[str, Any]],
) -> str:
    def spread_decision(axis: str) -> str:
        between = spread[f"between_model_spread_{axis}"]
        within = spread[f"within_model_var_{axis}"]
        if not (np.isfinite(between) and np.isfinite(within) and within > 0):
            return f"- {axis}: insufficient seed-resolution evidence for the spread-vs-seed-noise test."
        ratio = between / within
        if between <= within:
            return (
                f"- {axis}: negative-consistency headline supported "
                f"(between/within={ratio:.2f}; between-model spread <= seed resolution)."
            )
        return (
            f"- {axis}: negative-consistency headline NOT supported "
            f"(between/within={ratio:.2f}; between-model spread exceeds seed resolution)."
        )

    inventory_by_model: dict[str, list[dict[str, Any]]] = {}
    for row in inventory_rows:
        inventory_by_model.setdefault(row["model"], []).append(row)
    lines = [
        "# 2D SAE Interpretability Profile",
        "",
        "This report follows the locked 2D SAE profile outline. It is a profile, not a leaderboard.",
        "",
        "## Preflight 1: X Support",
        "",
        f"- X metric selected: `{x_metric}`",
        f"- common E/N_over_d support across six models: {', '.join(ffmt(x) for x in common_e) if common_e else 'none'}",
        "",
        "| Model | Available E/N_over_d levels |",
        "|---|---|",
    ]
    for model in EXPECTED_MODELS:
        levels = model_e.get(model, [])
        lines.append(f"| {model} | {', '.join(ffmt(x) for x in levels) if levels else 'missing'} |")
    lines.extend(
        [
            "",
            "## Preflight 2: Y Matched-Fidelity Point",
            "",
        ]
    )
    if y_point is None:
        lines.append("- Y_Nd_point: not selected.")
        lines.append("- Reason: no candidate in {8, 16, 32} has all six models with >=3 seeds, recon_R2 >= 0.85, and firing-rate vectors.")
    else:
        lines.append(f"- Y_Nd_point: {ffmt(y_point)}")
    lines.extend(["", "Candidate diagnostics:", ""])
    for candidate, details in y_preflight.items():
        lines.append(f"- `{candidate}`: " + "; ".join(f"{m}={reason}" for m, reason in sorted(details.items())))
    lines.extend(
        [
            "",
            "Checkpoint inventory summary:",
            "",
            "| Model | Checkpoints | Seeds | N/d levels | Firing-rate files |",
            "|---|---:|---|---|---:|",
        ]
    )
    for model in EXPECTED_MODELS:
        rows = inventory_by_model.get(model, [])
        seeds = sorted({str(r["seed"]) for r in rows})
        levels = sorted({str(r["N_over_d"]) for r in rows}, key=lambda x: parse_float(x))
        firing = sum(str(r.get("firing_rate_present", "")).lower() == "true" for r in rows)
        lines.append(
            f"| {model} | {len(rows)} | {', '.join(seeds) if seeds else 'missing'} | "
            f"{', '.join(levels[:10]) + (' ...' if len(levels) > 10 else '') if levels else 'missing'} | {firing} |"
        )
    lines.extend(
        [
            "",
            "## Preflight 3: Random Floor",
            "",
            "Locked definition: real decoder directions plus random one-to-one pairings. Gaussian floors are not used.",
            "",
            "## Profile Points",
            "",
            "| Model | X | X CI | Y | Y CI | Status |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in profile_rows:
        lines.append(
            f"| {row['model']} | {row['X_value']} | {row['X_seed_CI']} | "
            f"{row['Y_value']} | {row['Y_seed_CI']} | {row['profile_status']} |"
        )
    lines.extend(
        [
            "",
            "## Spread-vs-Seed-Resolution Check",
            "",
            f"- between_model_spread_X: {ffmt(spread['between_model_spread_X'])}",
            f"- within_model_var_X: {ffmt(spread['within_model_var_X'])}",
            f"- between_model_spread_Y: {ffmt(spread['between_model_spread_Y'])}",
            f"- within_model_var_Y: {ffmt(spread['within_model_var_Y'])}",
            "",
            "Decision:",
            "",
            spread_decision("X"),
            spread_decision("Y"),
            "",
        ]
    )
    if not np.isfinite(spread["within_model_var_X"]):
        lines.append("X seed-resolution cannot be estimated from current six-model artifacts because only one seed is present per model.")
    if y_point is None:
        lines.append("Y cannot be computed for the six-model profile from current artifacts. Required missing inputs: cross-seed dictionaries and firing-rate vectors at a matched N/d point.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build 2D SAE interpretability profile artifacts.")
    parser.add_argument("--recon-csv", type=Path, default=DEFAULT_RECON_CSV)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--random-permutations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260707)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    recon_rows = discover_recon_rows(args.recon_csv, args.checkpoint_root)
    recon_points = aggregate_recon_points(recon_rows)
    x_metric, common_e, model_e, x_preflight = choose_x_metric(recon_points)
    if x_metric == "X2":
        recon_profile, x_values = compute_x2(recon_points, common_e)
    else:
        recon_profile, x_values = compute_x1(recon_points)

    artifacts = collect_checkpoints(args.checkpoint_root)
    inventory = checkpoint_inventory_rows(artifacts)
    y_point, _details, y_preflight = choose_y_point(artifacts)
    if y_point is None:
        stability_rows = [
            {
                "status": "not_computed",
                "model": model,
                "N_over_d_fixed": "",
                "topk_K": "",
                "seed_i": "",
                "seed_j": "",
                "matched_cosine": "",
                "random_matched_cosine_floor": "",
                "stability_above_random": "",
                "dead_frac": "",
                "active_feature_count": "",
                "reason": "missing_matched_3seed_dictionaries_or_firing_rates",
            }
            for model in EXPECTED_MODELS
        ]
    else:
        rng = np.random.default_rng(args.seed)
        stability_rows = compute_y(artifacts, y_point, args.random_permutations, rng)

    profile_rows, spread = aggregate_profile(x_metric, x_values, stability_rows)
    profile_fields = [
        "row_type",
        "model",
        "seed",
        "d_hidden",
        "E",
        "N",
        "N_over_d",
        "recon_R2",
        "on_common_grid",
        "interpolated_recon_R2",
        "n_source_rows",
    ]
    stability_fields = [
        "status",
        "model",
        "N_over_d_fixed",
        "topk_K",
        "seed_i",
        "seed_j",
        "matched_cosine",
        "random_matched_cosine_floor",
        "stability_above_random",
        "dead_frac",
        "active_feature_count",
        "reason",
    ]
    point_fields = [
        "model",
        "X_metric",
        "X_value",
        "X_seed_CI",
        "Y_value",
        "Y_seed_CI",
        "between_model_spread_X",
        "within_model_var_X",
        "between_model_spread_Y",
        "within_model_var_Y",
        "profile_status",
    ]
    write_csv(args.out_dir / "recon_profile.csv", recon_profile, profile_fields)
    write_csv(
        args.out_dir / "checkpoint_inventory.csv",
        inventory,
        ["model", "seed", "N_over_d", "N", "d_hidden", "k", "recon_R2", "dead_frac", "firing_rate_present", "checkpoint"],
    )
    write_csv(args.out_dir / "dictionary_stability.csv", stability_rows, stability_fields)
    write_csv(args.out_dir / "profile_points.csv", profile_rows, point_fields)
    (args.out_dir / "preflight.json").write_text(
        json.dumps(
            {
                "x_metric": x_metric,
                "common_E": common_e,
                "model_E": model_e,
                "X_preflight": x_preflight,
                "recon_source_rows": len(recon_rows),
                "Y_Nd_point": y_point,
                "Y_preflight": y_preflight,
                "checkpoint_inventory_rows": len(inventory),
                "random_floor": "real_decoder_directions_random_pairing",
            },
            indent=2,
        )
        + "\n"
    )
    make_plot(profile_rows, args.out_dir / "sae_2d_profile.svg")
    (args.out_dir / "sae_2d_profile_report.md").write_text(
        render_report(x_metric, common_e, model_e, y_point, y_preflight, inventory, spread, profile_rows)
    )
    print(args.out_dir / "recon_profile.csv")
    print(args.out_dir / "checkpoint_inventory.csv")
    print(args.out_dir / "dictionary_stability.csv")
    print(args.out_dir / "profile_points.csv")
    print(args.out_dir / "sae_2d_profile.svg")
    print(args.out_dir / "sae_2d_profile_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
