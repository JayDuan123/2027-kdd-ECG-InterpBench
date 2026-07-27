"""CSFM forward-patch environment for the experimental SAE extension.

The SAE extension trains on pooled layer activations (`layer_XX_mean`). CSFM
continuation, however, resumes from post-block token activations. This
environment bridges the two by applying a per-record pooled delta as a mean shift
to every token in that record:

    token_patch = token + (patched_pooled - original_pooled)

Identity patches therefore reproduce the clean continuation path. Non-identity
SAE interventions should be reported as pooled-mean interventions, not tokenwise
SAE feature clamping.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmark_v1.adapters.csfm import CSFM_DEPTH, try_load_model
from benchmark_v1.config import ROOT

from .benchmark_environment import BenchmarkSAEEnvironment, _read_csv, _parse_float


def continue_from_post_block(model, x: torch.Tensor, layer_idx: int, mask=None) -> torch.Tensor:
    for attn, ff in model.transformer.layers[layer_idx + 1 :]:
        x = attn(x, mask=mask) + x
        x = ff(x) + x
    x = model.transformer.norm(x)
    x = x.mean(dim=1) if model.pool == "mean" else x[:, 0]
    x = model.to_latent(x)
    return model.mlp_head(x)


class CSFMSAEEnvironment(BenchmarkSAEEnvironment):
    def __init__(
        self,
        activation_index_dir: str | Path = ROOT / "results" / "activation_index" / "csfm_cu118_commons",
        split_csv: str | Path = ROOT / "results" / "manifest" / "split.csv",
        device: str = "cpu",
        task_alpha: float = 1.0,
        max_test_shards: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_index_dir = Path(activation_index_dir)
        self.split_csv = Path(split_csv)
        self.device = device
        self.task_alpha = task_alpha
        self.max_test_shards = max_test_shards
        self._model = None
        self._model_status = None
        self._head_cache: dict[str, tuple[object, object]] = {}
        self._patient_by_ecg: dict[str, str] | None = None

    def _load_model(self):
        if self._model is None:
            model, status = try_load_model(device=self.device)
            if model is None:
                raise RuntimeError(status)
            model.eval()
            self._model = model
            self._model_status = status
        return self._model

    def _matrix_column_by_ecg_id(self, ecg_ids: list[str], matrix_path: Path, column: str) -> np.ndarray:
        rows_by_id = {row["ecg_id"]: row for row in _read_csv(matrix_path)}
        values = np.empty(len(ecg_ids), dtype=np.float32)
        for idx, ecg_id in enumerate(ecg_ids):
            values[idx] = _parse_float(rows_by_id[ecg_id][column])
        return values

    def _train_task_head(self, task: str):
        if task in self._head_cache:
            return self._head_cache[task]
        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler

        suffix = "csfm_cu118_commons"
        records = self._records(suffix)
        splits = np.array([row["split"] for row in records])
        train_idx = np.where(splits == "train")[0]
        pooled_path = self._feature_path(suffix, "pooled")
        pooled = np.load(pooled_path, mmap_mode="r")
        ecg_ids = [row["ecg_id"] for row in records]
        labels_all = self._matrix_column_by_ecg_id(ecg_ids, self.tasks_matrix, task)
        valid_train = train_idx[np.isfinite(labels_all[train_idx])]
        if len(set(labels_all[valid_train].astype(int).tolist())) < 2:
            raise ValueError(f"task {task} has fewer than two train classes")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(np.asarray(pooled[valid_train], dtype=np.float32))
        head = RidgeClassifier(alpha=self.task_alpha)
        head.fit(x_train, labels_all[valid_train].astype(int))
        self._head_cache[task] = (scaler, head)
        return scaler, head

    def _test_pooled_matrix(self, model: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
        clean = self.load_activations(model, layer, "test").numpy()
        return clean, clean.copy()

    def _patient_id(self, ecg_id: str) -> str:
        if self._patient_by_ecg is None:
            self._patient_by_ecg = {
                row["ecg_id"]: row.get("patient_id") or row["ecg_id"]
                for row in _read_csv(self.split_csv)
                if row.get("ecg_id")
            }
        return self._patient_by_ecg.get(ecg_id, ecg_id)

    def forward_scores_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, np.ndarray]:
        if model != "CSFM":
            raise ValueError("CSFMSAEEnvironment only supports model='CSFM'")
        if split != "test":
            raise ValueError("CSFM SAE forward patch currently evaluates the test split only")
        if layer < 0 or layer >= CSFM_DEPTH:
            raise ValueError(f"layer must be in 0..{CSFM_DEPTH - 1}")

        from sklearn.metrics import roc_auc_score

        task = getattr(self, "_active_task", None)
        if task is None:
            raise RuntimeError("set_active_task(task) must be called before forward_with_patch")

        clean_test = self.load_activations(model, layer, "test")
        patched_test = patch_fn(clean_test)
        if not isinstance(patched_test, torch.Tensor):
            patched_test = torch.as_tensor(patched_test, dtype=torch.float32)
        delta_by_row = (patched_test.detach().cpu().numpy().astype(np.float32) - clean_test.numpy().astype(np.float32))

        scaler, head = self._train_task_head(task)
        csfm = self._load_model()
        shard_rows = [row for row in _read_csv(self.activation_index_dir / "shards.csv") if row["split"] == "test"]
        if self.max_test_shards > 0:
            shard_rows = shard_rows[: self.max_test_shards]

        # Probe-feature records define the row order used by clean_test.
        probe_records = [row for row in self._records("csfm_cu118_commons") if row["split"] == "test"]
        test_row_by_ecg = {row["ecg_id"]: idx for idx, row in enumerate(probe_records)}

        y_true: list[int] = []
        scores: list[float] = []
        ecg_ids_out: list[str] = []
        patient_ids: list[str] = []
        row_indices: list[int] = []
        with torch.no_grad():
            for shard in shard_rows:
                shard_dir = ROOT / Path(shard["activation_metadata"]).parent
                record_ids = [row["ecg_id"] for row in _read_csv(shard_dir / "record_ids.csv")]
                labels = self._matrix_column_by_ecg_id(record_ids, self.tasks_matrix, task)
                valid = np.isfinite(labels)
                if not valid.any():
                    continue
                tokens_np = np.asarray(
                    np.load(shard_dir / f"layer_{int(layer):02d}.npy", mmap_mode="r"),
                    dtype=np.float32,
                )
                row_idx = np.array([test_row_by_ecg[ecg_id] for ecg_id in record_ids], dtype=np.int64)
                token_delta = delta_by_row[row_idx][:, None, :]
                patched_tokens = torch.as_tensor(tokens_np + token_delta, dtype=torch.float32, device=self.device)
                pooled = continue_from_post_block(csfm, patched_tokens, int(layer)).detach().cpu().numpy().astype(np.float32)
                score = head.decision_function(scaler.transform(pooled[valid]))
                valid_positions = np.where(valid)[0]
                y_true.extend(labels[valid].astype(int).tolist())
                scores.extend(np.asarray(score).reshape(-1).tolist())
                for pos in valid_positions.tolist():
                    ecg_id = record_ids[pos]
                    ecg_ids_out.append(ecg_id)
                    patient_ids.append(self._patient_id(ecg_id))
                    row_indices.append(int(row_idx[pos]))

        y = np.asarray(y_true, dtype=np.int32)
        if len(set(y.tolist())) < 2:
            raise ValueError(f"task {task} has fewer than two evaluated classes")
        order = np.argsort(np.asarray(row_indices, dtype=np.int64))
        return {
            "row_indices": np.asarray(row_indices, dtype=np.int64)[order],
            "ecg_ids": np.asarray(ecg_ids_out, dtype=object)[order],
            "patient_ids": np.asarray(patient_ids, dtype=object)[order],
            "y": y[order],
            "scores": np.asarray(scores, dtype=np.float64)[order],
        }

    def forward_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        from sklearn.metrics import roc_auc_score

        out = self.forward_scores_with_patch(model, layer, split, patch_fn)
        return float(roc_auc_score(out["y"], out["scores"]))

    def set_active_task(self, task: str) -> None:
        self._active_task = task
