"""Unified-scale SAE training and operating-point selection."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import signal
from typing import Callable

import torch
from torch.utils.data import DataLoader, TensorDataset

from .topk_sae import TopKSAE


@dataclass
class SAEFit:
    sae: TopKSAE
    E: float
    k0: int
    N_capacity: int
    k: int
    l0_target: int
    l0_actual: float
    l0_relative_error: float
    recon_r2: float
    l0: int
    dead_frac: float
    task_retention: float
    matched_tier: str = "floor"
    quality_warning: bool = False
    nearest_recon_R2: float = float("nan")
    nearest_N: int = 0


def _batched_recon_r2(sae: TopKSAE, acts: torch.Tensor, batch_size: int = 8192) -> float:
    sae.eval()
    device = next(sae.parameters()).device
    mean = acts.mean(0).to(device)
    num = 0.0
    den = 0.0
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(acts), batch_size=batch_size):
            batch = batch.to(device)
            recon, _, acts_norm = sae(batch)
            num += float((acts_norm - recon).pow(2).sum().item())
            den += float((acts_norm - mean).pow(2).sum().item())
    return 1.0 - num / max(den, 1e-12)


def _batched_dead_frac(sae: TopKSAE, acts: torch.Tensor, batch_size: int = 8192) -> float:
    sae.eval()
    device = next(sae.parameters()).device
    fired = torch.zeros(sae.N, dtype=torch.bool, device=device)
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(acts), batch_size=batch_size):
            _, z, _ = sae(batch.to(device))
            fired |= (z > 0).any(dim=0)
    return 1.0 - float(fired.float().mean().item())


def _batched_l0(sae: TopKSAE, acts: torch.Tensor, batch_size: int = 8192) -> float:
    sae.eval()
    device = next(sae.parameters()).device
    total_active = 0.0
    total_rows = 0
    with torch.no_grad():
        for (batch,) in DataLoader(TensorDataset(acts), batch_size=batch_size):
            _, z, _ = sae(batch.to(device))
            total_active += float((z > 0).sum(dim=1).float().sum().item())
            total_rows += int(z.shape[0])
    return total_active / max(total_rows, 1)


def train_topk_sae(
    acts: torch.Tensor,
    n_features: int,
    k: int,
    steps: int = 4000,
    batch_size: int = 4096,
    lr: float = 1e-3,
    resample_every: int = 500,
    device: str = "cpu",
    seed: int = 4311,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 250,
) -> TopKSAE:
    torch.manual_seed(seed)
    d = acts.shape[1]
    sae = TopKSAE(d=d, n_features=n_features, k=k).to(device)
    sae.set_normalisation(acts.mean(0).to(device), acts.std(0).to(device))

    effective_batch = min(batch_size, len(acts))
    loader = DataLoader(
        TensorDataset(acts),
        batch_size=effective_batch,
        shuffle=True,
        drop_last=False,
    )
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
    start_step = 0
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        meta = saved.get("meta", {})
        if meta.get("d") != d or meta.get("n_features") != n_features or meta.get("k") != k:
            raise ValueError(f"checkpoint shape mismatch: {checkpoint}")
        sae.load_state_dict(saved["sae"])
        opt.load_state_dict(saved["optimizer"])
        start_step = int(saved.get("step", 0))
        if "torch_rng_state" in saved:
            torch.set_rng_state(saved["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in saved:
            cuda_state = saved["cuda_rng_state"]
            if isinstance(cuda_state, torch.Tensor) and cuda_state.dtype == torch.uint8:
                torch.cuda.set_rng_state(cuda_state.detach().cpu(), device=device)
        if start_step >= steps:
            sae.eval()
            return sae

    def save_checkpoint(step: int) -> None:
        if checkpoint is None:
            return
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "step": int(step),
            "sae": sae.state_dict(),
            "optimizer": opt.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "meta": {"d": d, "n_features": n_features, "k": k, "seed": seed, "steps": steps},
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(device=device)
        tmp_path = checkpoint.with_name(checkpoint.name + f".tmp.{id(payload)}")
        torch.save(payload, tmp_path)
        tmp_path.replace(checkpoint)

    stop_requested = {"value": False}
    previous_term = signal.getsignal(signal.SIGTERM)

    def handle_sigterm(signum, frame):
        stop_requested["value"] = True

    if checkpoint is not None:
        signal.signal(signal.SIGTERM, handle_sigterm)

    iterator = iter(loader)
    sae.train()
    try:
        for step in range(start_step, steps):
            if stop_requested["value"]:
                save_checkpoint(step)
                raise SystemExit(143)
            try:
                (batch,) = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                (batch,) = next(iterator)
            batch = batch.to(device)
            recon, _, acts_norm = sae(batch)
            loss = (recon - acts_norm).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                sae.normalise_decoder_()
            if resample_every and (step + 1) % resample_every == 0:
                sae.resample_dead(batch.detach())
            if checkpoint_every > 0 and checkpoint is not None and (step + 1) % checkpoint_every == 0:
                save_checkpoint(step + 1)
    finally:
        if checkpoint is not None:
            signal.signal(signal.SIGTERM, previous_term)
    save_checkpoint(steps)
    sae.eval()
    return sae


def sweep_operating_points(
    train_acts: torch.Tensor,
    val_acts: torch.Tensor,
    E_grid: tuple[int, ...] = (4, 8, 16),
    k0_grid: tuple[int, ...] = (16, 32, 64),
    l0_grid: tuple[int, ...] | None = None,
    n_features_grid: tuple[int, ...] | None = None,
    selection_mode: str = "floor",
    recon_target: float = 0.90,
    recon_band_width: float = 0.02,
    relaxed_band_width: float = 0.04,
    recon_r2_floor: float = 0.5,
    max_dead_frac: float = 0.30,
    task_retention_fn: Callable[[TopKSAE], float] | None = None,
    min_task_retention: float = 0.98,
    quality_dead_frac: float = 0.20,
    quality_retention: float = 0.95,
    device: str = "cpu",
    **train_kwargs,
) -> list[SAEFit]:
    d = train_acts.shape[1]
    fits: list[SAEFit] = []
    checkpoint_dir = train_kwargs.pop("checkpoint_dir", None)
    seed = int(train_kwargs.get("seed", 4311))
    capacity_grid: list[tuple[float, int]] = []
    if n_features_grid is not None:
        capacity_grid = [(float(n_features) / float(d), int(n_features)) for n_features in n_features_grid]
    else:
        capacity_grid = [(float(E), d * int(E)) for E in E_grid]
    for E, n_features in capacity_grid:
        if l0_grid is not None:
            sparsity_grid = [(int(l0), int(l0)) for l0 in l0_grid]
        else:
            sparsity_grid = [(int(k0), int(k0) * E) for k0 in k0_grid]
        for k0, k in sparsity_grid:
            if k >= n_features:
                continue
            checkpoint_path = None
            if checkpoint_dir is not None:
                checkpoint_path = Path(checkpoint_dir) / f"N{n_features}_k0{k0}_seed{seed}.pt"
            sae = train_topk_sae(
                train_acts,
                n_features=n_features,
                k=k,
                device=device,
                checkpoint_path=checkpoint_path,
                **train_kwargs,
            )
            r2 = _batched_recon_r2(sae, val_acts)
            dead = _batched_dead_frac(sae, val_acts)
            l0_actual = _batched_l0(sae, val_acts)
            retention = float("nan")
            if task_retention_fn is not None:
                retention = float(task_retention_fn(sae))
            if selection_mode == "recon_band":
                if recon_target <= r2 <= recon_target + recon_band_width:
                    matched_tier = "in_band"
                elif recon_target <= r2 <= recon_target + relaxed_band_width:
                    matched_tier = "relaxed_band"
                else:
                    matched_tier = "no_matched_point"
            else:
                ok = r2 >= recon_r2_floor and dead <= max_dead_frac
                if task_retention_fn is not None:
                    ok = ok and retention >= min_task_retention
                matched_tier = "floor" if ok else "failed_floor"
            quality_warning = bool(
                dead > quality_dead_frac
                or (math.isfinite(retention) and retention < quality_retention)
            )
            l0_relative_error = abs(l0_actual - k) / max(float(k), 1.0)
            fits.append(
                SAEFit(
                    sae=sae,
                    E=E,
                    k0=k0,
                    N_capacity=n_features,
                    k=k,
                    l0_target=k,
                    l0_actual=float(l0_actual),
                    l0_relative_error=float(l0_relative_error),
                    recon_r2=r2,
                    l0=k,
                    dead_frac=dead,
                    task_retention=retention,
                    matched_tier=matched_tier,
                    quality_warning=quality_warning,
                    nearest_recon_R2=float(r2),
                    nearest_N=n_features,
                )
            )
    return fits


def select_operating_point(
    train_acts: torch.Tensor,
    val_acts: torch.Tensor,
    E_grid: tuple[int, ...] = (4, 8, 16),
    k0_grid: tuple[int, ...] = (16, 32, 64),
    l0_grid: tuple[int, ...] | None = None,
    n_features_grid: tuple[int, ...] | None = None,
    selection_mode: str = "floor",
    recon_target: float = 0.90,
    recon_band_width: float = 0.02,
    relaxed_band_width: float = 0.04,
    recon_r2_floor: float = 0.5,
    max_dead_frac: float = 0.30,
    task_retention_fn: Callable[[TopKSAE], float] | None = None,
    min_task_retention: float = 0.98,
    quality_dead_frac: float = 0.20,
    quality_retention: float = 0.95,
    device: str = "cpu",
    **train_kwargs,
) -> SAEFit:
    fits = sweep_operating_points(
        train_acts=train_acts,
        val_acts=val_acts,
        E_grid=E_grid,
        k0_grid=k0_grid,
        l0_grid=l0_grid,
        n_features_grid=n_features_grid,
        selection_mode=selection_mode,
        recon_target=recon_target,
        recon_band_width=recon_band_width,
        relaxed_band_width=relaxed_band_width,
        recon_r2_floor=recon_r2_floor,
        max_dead_frac=max_dead_frac,
        task_retention_fn=task_retention_fn,
        min_task_retention=min_task_retention,
        quality_dead_frac=quality_dead_frac,
        quality_retention=quality_retention,
        device=device,
        **train_kwargs,
    )
    if not fits:
        raise RuntimeError("No SAE operating point candidates were trained")
    if selection_mode == "recon_band":
        for tier in ("in_band", "relaxed_band"):
            candidates = [fit for fit in fits if fit.matched_tier == tier]
            if candidates:
                candidates.sort(key=lambda fit: (fit.N_capacity, fit.dead_frac, fit.l0_actual))
                selected = candidates[0]
                selected.nearest_recon_R2 = selected.recon_r2
                selected.nearest_N = selected.N_capacity
                return selected
        selected = min(fits, key=lambda fit: (abs(fit.recon_r2 - recon_target), fit.N_capacity))
        selected.matched_tier = "no_matched_point"
        selected.quality_warning = True
        selected.nearest_recon_R2 = selected.recon_r2
        selected.nearest_N = selected.N_capacity
        return selected
    survivors = [fit for fit in fits if fit.matched_tier == "floor"]
    if not survivors:
        raise RuntimeError("No SAE operating point cleared the configured floors")
    survivors.sort(key=lambda fit: (fit.E, fit.k0))
    return survivors[0]
