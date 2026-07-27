"""TopK sparse autoencoder for the causally anchored SAE extension.

The SAE operates on raw model activations, normalises them internally with
train-split statistics, and reconstructs in normalised activation space. Patch
callers must de-normalise before continuing a frozen model forward.

TopK is implemented as "at most k positive features": the largest k positive
pre-activations are retained. This avoids silently claiming exactly-k activity
when the selected pre-activations contain non-positive values.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, d: int, n_features: int, k: int, dead_window: int = 500):
        super().__init__()
        if d <= 0 or n_features <= 0:
            raise ValueError("d and n_features must be positive")
        if not 0 < k < n_features:
            raise ValueError("k must satisfy 0 < k < n_features")
        self.d = d
        self.N = n_features
        self.k = k
        self.dead_window = dead_window
        self.track_dead_features = True

        self.W_enc = nn.Parameter(torch.empty(n_features, d))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(d, n_features))
        self.b_dec = nn.Parameter(torch.zeros(d))
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.normalise_decoder_()

        self.register_buffer("mu", torch.zeros(d))
        self.register_buffer("sigma", torch.ones(d))
        self.register_buffer("steps_since_fired", torch.zeros(n_features, dtype=torch.long))

    def set_normalisation(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        if mu.shape[-1] != self.d or sigma.shape[-1] != self.d:
            raise ValueError("mu and sigma must have shape (d,)")
        self.mu.copy_(mu.detach())
        self.sigma.copy_(sigma.detach().clamp_min(1e-6))

    def normalise(self, acts: torch.Tensor) -> torch.Tensor:
        return (acts - self.mu) / self.sigma

    def denormalise(self, acts_norm: torch.Tensor) -> torch.Tensor:
        return acts_norm * self.sigma + self.mu

    @torch.no_grad()
    def normalise_decoder_(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, acts_norm: torch.Tensor) -> torch.Tensor:
        pre = acts_norm @ self.W_enc.t() + self.b_enc
        positive = torch.relu(pre)
        topv, topi = positive.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topi, topv)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, acts_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        acts_norm = self.normalise(acts_raw)
        z = self.encode(acts_norm)
        recon_norm = self.decode(z)
        self._track_dead(z)
        return recon_norm, z, acts_norm

    def _track_dead(self, z: torch.Tensor) -> None:
        if not self.training or not self.track_dead_features:
            return
        with torch.no_grad():
            fired = (z > 0).any(dim=0)
            self.steps_since_fired[fired] = 0
            self.steps_since_fired[~fired] += 1

    @torch.no_grad()
    def resample_dead(self, acts_batch_raw: torch.Tensor) -> int:
        dead = self.steps_since_fired >= self.dead_window
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0

        old_tracking = self.track_dead_features
        self.track_dead_features = False
        recon_norm, _, acts_norm = self.forward(acts_batch_raw)
        self.track_dead_features = old_tracking

        resid = acts_norm - recon_norm
        resid_norm = resid.norm(dim=1)
        if float(resid_norm.sum().item()) <= 0:
            return 0
        probs = (resid_norm / resid_norm.sum()).detach().cpu()
        pick = torch.multinomial(probs, n_dead, replacement=True).to(resid.device)
        dirs = resid[pick]
        dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)

        dead_idx = torch.where(dead)[0]
        self.W_dec[:, dead_idx] = dirs.t()
        self.W_enc[dead_idx] = dirs * 0.2
        self.b_enc[dead_idx] = 0.0
        self.steps_since_fired[dead_idx] = 0
        self.normalise_decoder_()
        return n_dead

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
