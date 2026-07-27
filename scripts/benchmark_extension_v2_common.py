#!/usr/bin/env python
"""Shared helpers for benchmark_extension_v2 analyses."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "results" / "benchmark_extension_v1"
V2 = ROOT / "results" / "benchmark_extension_v2"


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def interval_and_p(samples: np.ndarray) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"ci_low": np.nan, "ci_high": np.nan, "p_two_sided": np.nan}
    low, high = np.quantile(values, [0.025, 0.975])
    left = (1.0 + float((values <= 0).sum())) / (len(values) + 1.0)
    right = (1.0 + float((values >= 0).sum())) / (len(values) + 1.0)
    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "p_two_sided": float(min(1.0, 2.0 * min(left, right))),
    }


def bootstrap_mean(values: np.ndarray, n_bootstrap: int, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.full(n_bootstrap, np.nan)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    return values[index].mean(axis=1)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
