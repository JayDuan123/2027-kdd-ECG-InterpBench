#!/usr/bin/env python
"""Shared definitions for the fair method-comparison benchmark."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "benchmark_method_comparison_v1"
EXTERNAL = ROOT / "results" / "external_benchmark_v1"
ACTIVATIONS = ROOT / "results" / "activations_external_full_v1" / "pooled"

MODEL_NAMES = {
    "cardiac_fm_cu118_commons": "CARDIAC-FM",
    "csfm_cu118_commons": "CSFM",
    "ecg_fm_cu118_commons": "ECG-FM",
    "ecg_jepa_cu118_commons": "ECG-JEPA",
    "hubert_ecg_cu118_commons": "HuBERT-ECG",
    "st_mem_cu118_commons": "ST-MEM",
}
COHORTS = ("chapman_f", "cpsc_f", "mimic_f", "ningbo_f")
SEEDS = (4311, 4312, 4313)
COMMON_RANK = 64
COMMON_K = 5
LABEL_BUDGETS = (32, 128, 512, 2048)
RATE_DISTORTION_K = (1, 2, 5, 10, 20, 64)
METHODS = (
    "sae_common64",
    "pca64",
    "ica64",
    "semi_nmf64",
    "random_basis64",
    "sparse_probe",
    "supervised_cav",
)
RECONSTRUCTIVE_METHODS = (
    "sae_common64",
    "pca64",
    "ica64",
    "semi_nmf64",
    "random_basis64",
)
REGIMES = ("common64_energy", "existing_sae_energy")
METHOD_METRICS = ("ste", "otd_mean", "selectivity_margin", "wbi", "behavior_effect")


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def stable_subset(
    indices: np.ndarray,
    identifiers: np.ndarray,
    limit: int,
    *key: object,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if limit <= 0 or len(indices) <= limit:
        return indices
    prefix = "|".join(map(str, key))
    scores = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{prefix}|{identifiers[index]}".encode()).digest()[:8],
                "big",
            )
            for index in indices
        ],
        dtype=np.uint64,
    )
    return indices[np.argsort(scores)[:limit]]


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(content)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def hard_topk(codes: np.ndarray, k: int, positive_only: bool = False) -> np.ndarray:
    values = np.asarray(codes, dtype=np.float32)
    if k >= values.shape[1]:
        return values.copy()
    score = values if positive_only else np.abs(values)
    selected = np.argpartition(score, -k, axis=1)[:, -k:]
    output = np.zeros_like(values)
    rows = np.arange(len(values))[:, None]
    output[rows, selected] = values[rows, selected]
    return output


def reconstruction_metrics(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.float64)
    reconstruction = np.asarray(reconstruction, dtype=np.float64)
    residual = reference - reconstruction
    centered = reference - reference.mean(axis=0, keepdims=True)
    denominator = float(np.square(centered).sum())
    mse = float(np.square(residual).mean())
    variance = float(np.square(centered).mean())
    numerator = np.sum(reference * reconstruction, axis=1)
    norm = np.linalg.norm(reference, axis=1) * np.linalg.norm(reconstruction, axis=1)
    cosine = np.divide(numerator, norm, out=np.zeros_like(numerator), where=norm > 1e-12)
    return {
        "recon_r2": float(1.0 - np.square(residual).sum() / max(denominator, 1e-12)),
        "normalized_mse": float(mse / max(variance, 1e-12)),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
    }


def norm_match(delta: np.ndarray, reference_norm: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=np.float64).copy()
    reference_norm = np.asarray(reference_norm, dtype=np.float64)
    norms = np.linalg.norm(delta, axis=1)
    valid = norms > 1e-10
    delta[valid] *= (reference_norm[valid] / norms[valid])[:, None]
    if (~valid).any():
        unit = np.asarray(fallback, dtype=np.float64)
        unit /= max(float(np.linalg.norm(unit)), 1e-12)
        delta[~valid] = reference_norm[~valid, None] * unit[None, :]
    return delta.astype(np.float32)


def component_association(codes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    valid = np.isfinite(labels)
    finite_labels = labels[valid]
    unique = np.unique(finite_labels)
    if len(unique) <= 2:
        positive = valid & (labels == unique.max())
        negative = valid & (labels == unique.min())
        difference = codes[positive].mean(axis=0) - codes[negative].mean(axis=0)
        pooled = np.sqrt(
            0.5 * (codes[positive].var(axis=0) + codes[negative].var(axis=0))
        )
        return np.divide(difference, pooled, out=np.zeros_like(difference), where=pooled > 1e-8)
    centered_labels = finite_labels - finite_labels.mean()
    centered_codes = codes[valid] - codes[valid].mean(axis=0, keepdims=True)
    numerator = centered_labels @ centered_codes
    denominator = np.linalg.norm(centered_labels) * np.linalg.norm(centered_codes, axis=0)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def selected_component_delta(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    decoder: np.ndarray,
    labels: np.ndarray,
    k: int = COMMON_K,
) -> tuple[np.ndarray, np.ndarray]:
    association = component_association(train_codes, labels)
    selected = np.argsort(np.abs(association))[::-1][:k]
    centroid = np.asarray(train_codes[:, selected].mean(axis=0), dtype=np.float32)
    dz = centroid[None, :] - np.asarray(test_codes[:, selected], dtype=np.float32)
    delta = dz @ np.asarray(decoder[selected], dtype=np.float32)
    return delta.astype(np.float32), selected.astype(np.int32)


def random_component_deltas(
    train_codes: np.ndarray,
    test_codes: np.ndarray,
    decoder: np.ndarray,
    selected: Iterable[int],
    n_random: int,
    seed: int,
    k: int = COMMON_K,
) -> list[np.ndarray]:
    selected_set = set(map(int, selected))
    available = np.asarray([i for i in range(train_codes.shape[1]) if i not in selected_set])
    if len(available) < k:
        raise RuntimeError("Not enough non-selected components for matched random groups")
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(n_random):
        group = np.sort(rng.choice(available, size=k, replace=False))
        centroid = train_codes[:, group].mean(axis=0)
        output.append(((centroid[None, :] - test_codes[:, group]) @ decoder[group]).astype(np.float32))
    return output


def draw_random_component_groups(
    width: int,
    selected: Iterable[int],
    n_random: int,
    seed: int,
    k: int = COMMON_K,
) -> list[np.ndarray]:
    selected_set = set(map(int, selected))
    available = np.asarray([index for index in range(width) if index not in selected_set])
    if len(available) < k:
        raise RuntimeError("Not enough non-selected components for random groups")
    rng = np.random.default_rng(seed)
    return [
        np.sort(rng.choice(available, size=k, replace=False)).astype(int)
        for _ in range(n_random)
    ]


def random_unit_directions(dimension: int, n_random: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n_random, dimension))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    return directions.astype(np.float32)


def component_control_logit_deltas(
    test_codes: np.ndarray,
    decoder: np.ndarray,
    groups: Iterable[np.ndarray],
    reference_norm: np.ndarray,
    fallback: np.ndarray,
    coefficients: np.ndarray,
    train_codes: np.ndarray | None = None,
    centroid: np.ndarray | None = None,
) -> np.ndarray:
    """Compute norm-matched random component logits from low-rank sufficient statistics."""
    if (train_codes is None) == (centroid is None):
        raise ValueError("Provide exactly one of train_codes or centroid")
    test_codes = np.asarray(test_codes, dtype=np.float64)
    decoder = np.asarray(decoder, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    reference = np.asarray(reference_norm, dtype=np.float64)
    fallback_unit = np.asarray(fallback, dtype=np.float64)
    fallback_unit /= max(float(np.linalg.norm(fallback_unit)), 1e-12)
    fallback_logits = fallback_unit @ coefficients.T
    output = []
    for raw_group in groups:
        group = np.asarray(raw_group, dtype=int)
        center = (
            np.asarray(train_codes[:, group], dtype=np.float64).mean(axis=0)
            if train_codes is not None
            else np.asarray(centroid, dtype=np.float64)[group]
        )
        dz = center[None, :] - test_codes[:, group]
        directions = decoder[group]
        gram = directions @ directions.T
        norm_squared = np.einsum("ni,ij,nj->n", dz, gram, dz, optimize=True)
        raw_norm = np.sqrt(np.maximum(norm_squared, 0.0))
        logits = dz @ (directions @ coefficients.T)
        valid = raw_norm > 1e-10
        scale = np.divide(reference, raw_norm, out=np.zeros_like(reference), where=valid)
        logits[valid] *= scale[valid, None]
        logits[~valid] = reference[~valid, None] * fallback_logits[None, :]
        output.append(logits.astype(np.float32))
    return np.stack(output, axis=1)


def direction_control_logit_deltas(
    train: np.ndarray,
    test: np.ndarray,
    directions: np.ndarray,
    reference_norm: np.ndarray,
    fallback: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Compute norm-matched random direction logits without n x d delta matrices."""
    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    reference = np.asarray(reference_norm, dtype=np.float64)
    fallback_unit = np.asarray(fallback, dtype=np.float64)
    fallback_unit /= max(float(np.linalg.norm(fallback_unit)), 1e-12)
    fallback_logits = fallback_unit @ coefficients.T
    output = []
    train_mean = train.mean(axis=0)
    for direction in directions:
        unit = direction / max(float(np.linalg.norm(direction)), 1e-12)
        scalar = float(train_mean @ unit) - test @ unit
        raw_norm = np.abs(scalar)
        logits = scalar[:, None] * (unit @ coefficients.T)[None, :]
        valid = raw_norm > 1e-10
        scale = np.divide(reference, raw_norm, out=np.zeros_like(reference), where=valid)
        logits[valid] *= scale[valid, None]
        logits[~valid] = reference[~valid, None] * fallback_logits[None, :]
        output.append(logits.astype(np.float32))
    return np.stack(output, axis=1)


def direction_delta(train: np.ndarray, test: np.ndarray, direction: np.ndarray) -> np.ndarray:
    unit = np.asarray(direction, dtype=np.float64)
    unit /= max(float(np.linalg.norm(unit)), 1e-12)
    center = float(np.asarray(train, dtype=np.float64).mean(axis=0) @ unit)
    score = np.asarray(test, dtype=np.float64) @ unit
    return ((center - score)[:, None] * unit[None, :]).astype(np.float32)


def random_direction_deltas(
    train: np.ndarray,
    test: np.ndarray,
    n_random: int,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(n_random):
        direction = rng.normal(size=train.shape[1])
        output.append(direction_delta(train, test, direction))
    return output


def logit_delta(delta_standardized: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return np.asarray(delta_standardized @ coefficients.T, dtype=np.float32)


def norm_matched_logit_delta(
    delta_standardized: np.ndarray,
    reference_norm: np.ndarray,
    fallback: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply exact per-record norm matching without materializing a scaled d-vector."""
    delta = np.asarray(delta_standardized, dtype=np.float64)
    reference = np.asarray(reference_norm, dtype=np.float64)
    norms = np.linalg.norm(delta, axis=1)
    logits = delta @ np.asarray(coefficients, dtype=np.float64).T
    valid = norms > 1e-10
    scale = np.divide(reference, norms, out=np.zeros_like(reference), where=valid)
    logits[valid] *= scale[valid, None]
    if (~valid).any():
        unit = np.asarray(fallback, dtype=np.float64)
        unit /= max(float(np.linalg.norm(unit)), 1e-12)
        logits[~valid] = reference[~valid, None] * (
            unit @ np.asarray(coefficients, dtype=np.float64).T
        )[None, :]
    matched_norm = np.where(valid, norms * scale, reference)
    error = float(np.max(np.abs(matched_norm - reference)))
    return logits.astype(np.float32), error
