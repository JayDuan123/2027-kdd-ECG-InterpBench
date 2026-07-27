"""Forward-patch environment for Transformer ECG foundation models.

This wires the SAE steering layer to the same continuation functions used by
the LEACE causal audit. The SAE itself operates on pooled layer activations
(`layer_XX_mean`). For steering, we apply the pooled delta back to every token
at that layer and continue the frozen encoder/head path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch

from benchmark_v1.config import ROOT
from scripts.run_transformer_continuation_erase import continue_model, load_model

from .benchmark_environment import BenchmarkSAEEnvironment, _parse_float, _read_csv


MODEL_TO_KEY = {
    "CARDIAC-FM": "cardiac_fm",
    "ECG-FM": "ecg_fm",
    "ECG-JEPA": "ecg_jepa",
    "HuBERT-ECG": "hubert_ecg",
    "ST-MEM": "st_mem",
}


class TransformerSAEEnvironment(BenchmarkSAEEnvironment):
    def __init__(
        self,
        activation_index_root: str | Path = ROOT / "results" / "activation_index",
        split_csv: str | Path = ROOT / "results" / "manifest" / "split.csv",
        device: str = "cpu",
        task_alpha: float = 1.0,
        max_test_shards: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.activation_index_root = Path(activation_index_root)
        self.split_csv = Path(split_csv)
        self.device = device
        self.task_alpha = task_alpha
        self.max_test_shards = max_test_shards
        self._models: dict[str, tuple[object, str]] = {}
        self._head_cache: dict[tuple[str, str], tuple[object, object]] = {}
        self._patient_by_ecg: dict[str, str] | None = None

    def _load_model(self, model: str):
        model_key = MODEL_TO_KEY.get(model)
        if model_key is None:
            raise ValueError(f"unsupported Transformer SAE model {model!r}")
        if model_key not in self._models:
            loaded, status = load_model(model_key, self.device)
            self._models[model_key] = (loaded, status)
        return model_key, self._models[model_key][0]

    def _matrix_column_by_ecg_id(self, ecg_ids: list[str], matrix_path: Path, column: str) -> np.ndarray:
        rows_by_id = {row["ecg_id"]: row for row in _read_csv(matrix_path)}
        values = np.empty(len(ecg_ids), dtype=np.float32)
        for idx, ecg_id in enumerate(ecg_ids):
            values[idx] = _parse_float(rows_by_id[ecg_id][column])
        return values

    def _activation_index_dir(self, suffix: str) -> Path:
        path = self.activation_index_root / suffix
        if not path.exists():
            raise FileNotFoundError(f"missing activation index directory: {path}")
        return path

    def _train_task_head(self, suffix: str, task: str):
        key = (suffix, task)
        if key in self._head_cache:
            return self._head_cache[key]

        from sklearn.linear_model import RidgeClassifier
        from sklearn.preprocessing import StandardScaler

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
        self._head_cache[key] = (scaler, head)
        return scaler, head

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
        if split != "test":
            raise ValueError("Transformer SAE forward patch currently evaluates the test split only")
        task = getattr(self, "_active_task", None)
        if task is None:
            raise RuntimeError("set_active_task(task) must be called before forward_with_patch")

        suffix = self._suffix_for_model_layer(model, layer)
        self._active_suffix = suffix
        model_key, loaded_model = self._load_model(model)
        activation_index_dir = self._activation_index_dir(suffix)

        clean_test = self.load_activations(model, layer, "test")
        patched_test = patch_fn(clean_test)
        if not isinstance(patched_test, torch.Tensor):
            patched_test = torch.as_tensor(patched_test, dtype=torch.float32)
        delta_by_row = (
            patched_test.detach().cpu().numpy().astype(np.float32)
            - clean_test.numpy().astype(np.float32)
        )

        scaler, head = self._train_task_head(suffix, task)
        shard_rows = [row for row in _read_csv(activation_index_dir / "shards.csv") if row["split"] == "test"]
        if self.max_test_shards > 0:
            shard_rows = shard_rows[: self.max_test_shards]

        probe_records = [row for row in self._records(suffix) if row["split"] == "test"]
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
                row_idx = np.array([test_row_by_ecg[ecg_id] for ecg_id in record_ids], dtype=np.int64)
                tokens_np = np.asarray(
                    np.load(shard_dir / f"layer_{int(layer):02d}.npy", mmap_mode="r"),
                    dtype=np.float32,
                )
                if tokens_np.shape[-1] != delta_by_row.shape[-1]:
                    raise ValueError(
                        f"cannot patch {model} layer {layer}: token dim {tokens_np.shape[-1]} "
                        f"!= pooled SAE dim {delta_by_row.shape[-1]}"
                    )
                token_delta = delta_by_row[row_idx][:, None, :]
                patched_tokens = torch.as_tensor(tokens_np + token_delta, dtype=torch.float32, device=self.device)
                pooled = (
                    continue_model(model_key, loaded_model, patched_tokens, int(layer))
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
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
