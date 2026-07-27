"""Concrete data/artifact loader for the experimental SAE extension.

This environment wires the non-forward pieces of Module S to the artifacts
already produced by the v1 benchmark:

- pooled layer features from `results/probe_features/<suffix>`
- exported LEACE/CAV artifacts from `results/sae_artifacts`
- concept/task matrices from `results/manifest`

`forward_with_patch` intentionally remains unimplemented. A valid SAE steering
run must reuse the exact model-specific continuation patcher from the LEACE
pipeline; returning a surrogate metric here would create fake steering results.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmark_v1.config import ROOT

from .environment import Environment


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class BenchmarkSAEEnvironment(Environment):
    def __init__(
        self,
        artifact_root: str | Path = ROOT / "results" / "sae_artifacts",
        probe_features_root: str | Path = ROOT / "results" / "probe_features",
        concepts_matrix: str | Path = ROOT / "results" / "manifest" / "concepts_matrix.csv",
        tasks_matrix: str | Path = ROOT / "results" / "manifest" / "tasks_matrix.csv",
    ):
        self.artifact_root = Path(artifact_root)
        self.probe_features_root = Path(probe_features_root)
        self.concepts_matrix = Path(concepts_matrix)
        self.tasks_matrix = Path(tasks_matrix)
        self.manifest = _read_csv(self.artifact_root / "manifest.csv")
        self._active_suffix: str | None = None
        self._measurement_raw_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
        self._measurement_stats_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _artifact_row(self, model: str, concept: str, task: str, layer: int) -> dict[str, str]:
        matches = [
            row
            for row in self.manifest
            if row["model"] == model
            and row["concept_id"] == concept
            and (row["canonical_task_id"] == task or row["task_id"] == task)
            and int(float(row["layer"])) == int(layer)
        ]
        if not matches:
            raise KeyError(f"no SAE artifact row for {(model, concept, task, layer)}")
        if len(matches) > 1:
            raise KeyError(f"ambiguous SAE artifact rows for {(model, concept, task, layer)}")
        return matches[0]

    def _suffix_for_model_layer(self, model: str, layer: int) -> str:
        matches = [
            row["suffix"]
            for row in self.manifest
            if row["model"] == model and int(float(row["layer"])) == int(layer)
        ]
        if not matches:
            raise KeyError(f"no SAE artifact suffix for {(model, layer)}")
        return matches[0]

    def _records(self, suffix: str) -> list[dict[str, str]]:
        return _read_csv(self.probe_features_root / suffix / "records.csv")

    def _feature_path(self, suffix: str, feature_name: str) -> Path:
        for row in _read_csv(self.probe_features_root / suffix / "features.csv"):
            if row["feature"] == feature_name:
                return ROOT / row["file"]
        raise KeyError(f"feature {feature_name!r} not found for {suffix}")

    def _split_indices(self, suffix: str, split: str) -> np.ndarray:
        records = self._records(suffix)
        return np.array([idx for idx, row in enumerate(records) if row["split"] == split], dtype=np.int64)

    def _matrix_values(self, suffix: str, matrix_path: Path, column: str, split: str) -> np.ndarray:
        records = self._records(suffix)
        rows_by_id = {row["ecg_id"]: row for row in _read_csv(matrix_path)}
        idx = self._split_indices(suffix, split)
        values = np.empty(len(idx), dtype=np.float32)
        for out_i, record_i in enumerate(idx):
            ecg_id = records[int(record_i)]["ecg_id"]
            values[out_i] = _parse_float(rows_by_id[ecg_id][column])
        return values

    def _active_or_default_suffix(self) -> str:
        if self._active_suffix is not None:
            return self._active_suffix
        if not self.manifest:
            raise RuntimeError("SAE artifact manifest is empty")
        self._active_suffix = self.manifest[0]["suffix"]
        return self._active_suffix

    def _load_measurements_raw(self, suffix: str, split: str) -> tuple[np.ndarray, list[str]]:
        key = f"{suffix}:{split}"
        if key in self._measurement_raw_cache:
            return self._measurement_raw_cache[key]
        records = self._records(suffix)
        idx = self._split_indices(suffix, split)
        matrix_rows = _read_csv(self.concepts_matrix)
        names = [name for name in matrix_rows[0].keys() if name != "ecg_id"]
        rows_by_id = {row["ecg_id"]: row for row in matrix_rows}
        out = np.empty((len(idx), len(names)), dtype=np.float32)
        for out_i, record_i in enumerate(idx):
            ecg_id = records[int(record_i)]["ecg_id"]
            row = rows_by_id[ecg_id]
            out[out_i] = [_parse_float(row[name]) for name in names]
        self._measurement_raw_cache[key] = (out, names)
        return out, names

    def _measurement_train_stats(self, suffix: str) -> tuple[np.ndarray, np.ndarray]:
        if suffix in self._measurement_stats_cache:
            return self._measurement_stats_cache[suffix]
        train_raw, _ = self._load_measurements_raw(suffix, "train")
        mu = np.nanmean(train_raw, axis=0)
        sigma = np.nanstd(train_raw, axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.0)
        self._measurement_stats_cache[suffix] = (mu.astype(np.float32), sigma.astype(np.float32))
        return self._measurement_stats_cache[suffix]

    def load_activations(self, model: str, layer: int, split: str) -> torch.Tensor:
        suffix = self._suffix_for_model_layer(model, layer)
        self._active_suffix = suffix
        feature = f"layer_{int(layer):02d}_mean"
        path = self._feature_path(suffix, feature)
        x = np.load(path, mmap_mode="r")
        idx = self._split_indices(suffix, split)
        return torch.as_tensor(np.asarray(x[idx], dtype=np.float32))

    def load_leace_subspace(self, model: str, concept: str, task: str, layer: int) -> np.ndarray:
        row = self._artifact_row(model, concept, task, layer)
        self._active_suffix = row["suffix"]
        return np.load(ROOT / row["artifact_dir"] / "leace_u_sae_norm.npy")

    def load_cav(self, model: str, concept: str, layer: int) -> np.ndarray:
        matches = [
            row
            for row in self.manifest
            if row["model"] == model and row["concept_id"] == concept and int(float(row["layer"])) == int(layer)
        ]
        if not matches:
            raise KeyError(f"no CAV artifact for {(model, concept, layer)}")
        row = matches[0]
        self._active_suffix = row["suffix"]
        return np.load(ROOT / row["artifact_dir"] / "cav_sae_norm.npy")

    def load_measurements(self, split: str) -> tuple[np.ndarray, list[str]]:
        suffix = self._active_or_default_suffix()
        out, names = self._load_measurements_raw(suffix, split)
        mu, sigma = self._measurement_train_stats(suffix)
        out = (out - mu) / sigma
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.astype(np.float32), names

    def concept_column(self, concept: str) -> str:
        return concept

    def task_labels(self, task: str, split: str) -> np.ndarray:
        suffix = self._active_or_default_suffix()
        return self._matrix_values(suffix, self.tasks_matrix, task, split)

    def forward_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        raise NotImplementedError(
            "forward_with_patch must reuse the model-specific LEACE continuation patcher; "
            "this loader intentionally does not return surrogate steering metrics."
        )
