#!/usr/bin/env python
"""Fitting and inference helpers for common-budget representation methods."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import random
import warnings

import joblib
import numpy as np
from sklearn.decomposition import FastICA, PCA
from sklearn.exceptions import ConvergenceWarning


def atomic_joblib_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(path)


def fit_or_load_pca(path: Path, train: np.ndarray, rank: int, seed: int) -> PCA:
    if path.exists():
        return joblib.load(path)
    model = PCA(
        n_components=rank,
        svd_solver="randomized",
        random_state=seed,
        iterated_power=4,
    )
    model.fit(train)
    atomic_joblib_dump(model, path)
    return model


def fit_or_load_ica(
    path: Path,
    train: np.ndarray,
    rank: int,
    seed: int,
    max_iter: int,
    tolerance: float,
    force: bool = False,
) -> tuple[FastICA, bool]:
    if path.exists() and not force:
        model = joblib.load(path)
        return model, bool(getattr(model, "n_iter_", max_iter) < max_iter)
    model = FastICA(
        n_components=rank,
        algorithm="parallel",
        whiten="unit-variance",
        fun="logcosh",
        max_iter=max_iter,
        tol=tolerance,
        whiten_solver="eigh",
        random_state=seed,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train)
    converged = not any(isinstance(item.message, ConvergenceWarning) for item in caught)
    atomic_joblib_dump(model, path)
    return model, converged


def fit_or_load_random_basis(path: Path, dimension: int, rank: int, seed: int) -> dict:
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return {key: saved[key] for key in saved.files}
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dimension, rank)))
    payload = {
        "mean": np.zeros(dimension, dtype=np.float32),
        "basis": q.astype(np.float32),
    }
    atomic_npz(path, **payload)
    return payload


def train_or_load_common_sae(
    path: Path,
    train: np.ndarray,
    dimension: int,
    rank: int,
    k: int,
    seed: int,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: str,
):
    import torch
    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    model = BatchTopKSAE(dimension, rank, k).to(device)
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if not saved.get("final", False):
            raise RuntimeError(f"Incomplete common SAE checkpoint: {path}")
        model.load_state_dict(saved["model"])
        model.eval()
        return model, saved

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train = np.asarray(train, dtype=np.float32)
    model.set_normalisation(
        torch.from_numpy(train.mean(axis=0)).to(device),
        torch.from_numpy(train.std(axis=0).clip(min=1e-6)).to(device),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    rng = np.random.default_rng(seed)
    losses = []
    best_loss = float("inf")
    best_state = None
    model.train()
    for step in range(steps):
        indices = rng.choice(len(train), size=min(batch_size, len(train)), replace=False)
        batch = torch.as_tensor(train[indices], dtype=torch.float32, device=device)
        reconstruction, _, normalized = model(batch)
        reconstruction_loss = (reconstruction - normalized).square().mean()
        auxiliary_loss = model.auxiliary_loss(normalized, reconstruction)
        loss = reconstruction_loss + auxiliary_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100000.0)
        model.project_decoder_grad_and_normalise_()
        optimizer.step()
        model.normalise_decoder_()
        value = float(loss.item())
        losses.append(value)
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())
        if (step + 1) % 250 == 0:
            print(
                f"common SAE seed={seed} step={step + 1}/{steps} "
                f"loss={np.mean(losses[-100:]):.6f}",
                flush=True,
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    payload = {
        "schema_version": 1,
        "architecture": "BatchTopK",
        "model": model.state_dict(),
        "final": True,
        "steps": steps,
        "seed": seed,
        "rank": rank,
        "k": k,
        "best_training_loss": best_loss,
        "last_100_training_loss": float(np.mean(losses[-100:])),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    temporary.replace(path)
    return model, payload


def encode_decode_sae(
    model,
    values: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    codes = []
    reconstructions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(
                np.asarray(values[start : start + batch_size]),
                dtype=torch.float32,
                device=device,
            )
            code = model.encode(batch)
            reconstruction = model.decode(code)
            codes.append(code.cpu().numpy().astype(np.float32))
            reconstructions.append(reconstruction.cpu().numpy().astype(np.float32))
    return np.concatenate(codes), np.concatenate(reconstructions)


def _positive(values):
    import torch

    return torch.clamp(values, min=0)


def _negative(values):
    import torch

    return torch.clamp(-values, min=0)


def _semi_nmf_update_codes(x, z, decoder, iterations: int):
    import torch

    gram = decoder @ decoder.T
    gram_positive = _positive(gram)
    gram_negative = _negative(gram)
    cross = x @ decoder.T
    cross_positive = _positive(cross)
    cross_negative = _negative(cross)
    epsilon = torch.tensor(1e-8, dtype=x.dtype, device=x.device)
    for _ in range(iterations):
        numerator = cross_positive + z @ gram_negative + epsilon
        denominator = cross_negative + z @ gram_positive + epsilon
        z = torch.clamp(z * torch.sqrt(numerator / denominator), min=0, max=1e6)
    return z


def fit_or_load_semi_nmf(
    path: Path,
    train: np.ndarray,
    rank: int,
    seed: int,
    iterations: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    import torch

    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in saved.files}
        return payload, {
            "iterations": int(payload["iterations"].item()),
            "relative_frobenius_error": float(payload["relative_frobenius_error"].item()),
        }

    rng = np.random.default_rng(seed)
    mean = np.asarray(train.mean(axis=0), dtype=np.float32)
    centered = np.asarray(train - mean, dtype=np.float32)
    x = torch.as_tensor(centered, dtype=torch.float32, device=device)
    initial_decoder = rng.normal(size=(rank, centered.shape[1])).astype(np.float32)
    initial_decoder /= np.maximum(np.linalg.norm(initial_decoder, axis=1, keepdims=True), 1e-8)
    decoder = torch.as_tensor(initial_decoder, dtype=torch.float32, device=device)
    z = torch.clamp(x @ decoder.T, min=0) + 1e-4
    identity = torch.eye(rank, dtype=torch.float32, device=device)
    epsilon = torch.tensor(1e-8, dtype=torch.float32, device=device)
    for iteration in range(iterations):
        decoder = torch.linalg.solve(z.T @ z + 1e-5 * identity, z.T @ x)
        gram = decoder @ decoder.T
        cross = x @ decoder.T
        numerator = _positive(cross) + z @ _negative(gram) + epsilon
        denominator = _negative(cross) + z @ _positive(gram) + epsilon
        z = torch.clamp(z * torch.sqrt(numerator / denominator), min=0, max=1e6)
        norms = decoder.norm(dim=1).clamp_min(1e-8)
        decoder = decoder / norms[:, None]
        z = z * norms[None, :]
        if (iteration + 1) % 20 == 0:
            relative = torch.linalg.norm(x - z @ decoder) / torch.linalg.norm(x).clamp_min(1e-8)
            print(
                f"semi-NMF seed={seed} iteration={iteration + 1}/{iterations} "
                f"relative_error={float(relative):.6f}",
                flush=True,
            )
    decoder = torch.linalg.solve(z.T @ z + 1e-5 * identity, z.T @ x)
    relative = torch.linalg.norm(x - z @ decoder) / torch.linalg.norm(x).clamp_min(1e-8)
    payload = {
        "mean": mean,
        "decoder": decoder.detach().cpu().numpy().astype(np.float32),
        "iterations": np.asarray(iterations, dtype=np.int32),
        "relative_frobenius_error": np.asarray(float(relative), dtype=np.float64),
    }
    atomic_npz(path, **payload)
    return payload, {
        "iterations": iterations,
        "relative_frobenius_error": float(relative),
    }


def semi_nmf_transform(
    values: np.ndarray,
    payload: dict[str, np.ndarray],
    iterations: int,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    import torch

    decoder = torch.as_tensor(payload["decoder"], dtype=torch.float32, device=device)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    gram_inverse = torch.linalg.pinv(decoder @ decoder.T)
    codes = []
    for start in range(0, len(values), batch_size):
        centered = np.asarray(values[start : start + batch_size] - mean, dtype=np.float32)
        x = torch.as_tensor(centered, dtype=torch.float32, device=device)
        z = torch.clamp(x @ decoder.T @ gram_inverse, min=0) + 1e-6
        z = _semi_nmf_update_codes(x, z, decoder, iterations)
        codes.append(z.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(codes)
