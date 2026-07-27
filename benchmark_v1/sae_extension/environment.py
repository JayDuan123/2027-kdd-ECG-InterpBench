"""Environment protocol for wiring the SAE extension to LEACE artifacts."""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import torch


class Environment(Protocol):
    def load_activations(self, model: str, layer: int, split: str) -> torch.Tensor:
        """Return raw pooled activations row-aligned with the requested split."""

    def load_leace_subspace(self, model: str, concept: str, task: str, layer: int) -> np.ndarray:
        """Return the real LEACE removed subspace U_r with shape (d, r)."""

    def load_cav(self, model: str, concept: str, layer: int) -> np.ndarray:
        """Return the dense-probe concept direction v_q with shape (d,)."""

    def load_measurements(self, split: str) -> tuple[np.ndarray, list[str]]:
        """Return standardised PTB-XL+ measurement matrix and column names."""

    def concept_column(self, concept: str) -> str:
        """Map benchmark concept id to the measurement column used as ground truth."""

    def task_labels(self, task: str, split: str) -> np.ndarray:
        """Return labels row-aligned with activations."""

    def forward_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        """Patch raw activations and continue frozen forward, returning task AUROC."""

    def forward_scores_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, np.ndarray]:
        """Optional: return row-aligned row_indices, patient_ids, labels, and task scores for bootstrap."""


class StubEnvironment:
    """Fail-loud placeholder.

    Copy this class when wiring the extension to the existing benchmark. Do not
    replace methods with synthetic data; A_geo and steering metrics are only
    meaningful with real LEACE subspaces, real CAVs, and the exact continuation
    patcher used by the LEACE run.
    """

    def __init__(self, artifact_root: str):
        self.artifact_root = artifact_root

    def load_activations(self, model: str, layer: int, split: str) -> torch.Tensor:
        raise NotImplementedError("wire to raw pooled activation cache")

    def load_leace_subspace(self, model: str, concept: str, task: str, layer: int) -> np.ndarray:
        raise NotImplementedError("wire to persisted LEACE U_r basis")

    def load_cav(self, model: str, concept: str, layer: int) -> np.ndarray:
        raise NotImplementedError("wire to persisted dense-probe CAV/probe direction")

    def load_measurements(self, split: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("wire to standardised PTB-XL+ measurement matrix")

    def concept_column(self, concept: str) -> str:
        raise NotImplementedError("wire concept id to measurement column mapping")

    def task_labels(self, task: str, split: str) -> np.ndarray:
        raise NotImplementedError("wire to benchmark task labels")

    def forward_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> float:
        raise NotImplementedError("wire to the existing LEACE continuation patcher")

    def forward_scores_with_patch(
        self,
        model: str,
        layer: int,
        split: str,
        patch_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> dict[str, np.ndarray]:
        raise NotImplementedError("wire to row-aligned continuation scores for patient bootstrap")
