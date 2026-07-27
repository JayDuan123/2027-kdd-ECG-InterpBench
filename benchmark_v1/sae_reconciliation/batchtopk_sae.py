"""BatchTopK SAE used by the CSFM Fig. 6 reconciliation test.

This is intentionally separate from the benchmark's per-row TopK SAE. During
training and evaluation, BatchTopK retains the largest ``batch_size * k``
positive pre-activations over the complete batch.
"""
from __future__ import annotations

import torch
from torch import nn


class BatchTopKSAE(nn.Module):
    def __init__(self, d: int = 768, n_features: int = 8192, k: int = 128,
                 n_batches_to_dead: int = 5, top_k_aux: int = 512,
                 aux_penalty: float = 1 / 32):
        super().__init__()
        if not 0 < k < n_features:
            raise ValueError("k must satisfy 0 < k < n_features")
        self.d = d
        self.N = n_features
        self.k = k
        self.n_batches_to_dead = n_batches_to_dead
        self.top_k_aux = top_k_aux
        self.aux_penalty = aux_penalty
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
        self.register_buffer("num_batches_not_active", torch.zeros(n_features))

    def set_normalisation(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        self.mu.copy_(mu.detach())
        self.sigma.copy_(sigma.detach().clamp_min(1e-6))

    def normalise(self, acts: torch.Tensor) -> torch.Tensor:
        return (acts - self.mu) / self.sigma

    def denormalise(self, acts_norm: torch.Tensor) -> torch.Tensor:
        return acts_norm * self.sigma + self.mu

    @torch.no_grad()
    def normalise_decoder_(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def pre_activations(self, acts_norm: torch.Tensor) -> torch.Tensor:
        return torch.relu((acts_norm - self.b_dec) @ self.W_enc.t() + self.b_enc)

    def encode_pre_activations(self, pre: torch.Tensor) -> torch.Tensor:
        flat = pre.flatten()
        keep = min(flat.numel(), pre.shape[0] * self.k)
        values, indices = flat.topk(keep, sorted=False)
        z_flat = torch.zeros_like(flat)
        z_flat.scatter_(0, indices, values)
        return z_flat.view_as(pre)

    def encode_normalised(self, acts_norm: torch.Tensor) -> torch.Tensor:
        return self.encode_pre_activations(self.pre_activations(acts_norm))

    def encode(self, acts_raw: torch.Tensor) -> torch.Tensor:
        return self.encode_normalised(self.normalise(acts_raw))

    def decode_normalised(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.t() + self.b_dec

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.denormalise(self.decode_normalised(z))

    def forward(self, acts_raw: torch.Tensor):
        acts_norm = self.normalise(acts_raw)
        pre = self.pre_activations(acts_norm)
        z = self.encode_pre_activations(pre)
        recon_norm = self.decode_normalised(z)
        if self.training:
            with torch.no_grad():
                active = z.sum(0) > 0
                self.num_batches_not_active.add_((~active).float())
                self.num_batches_not_active[active] = 0
        return recon_norm, z, acts_norm

    def auxiliary_loss(self, acts_norm: torch.Tensor, recon_norm: torch.Tensor) -> torch.Tensor:
        """Canonical TopK auxiliary residual reconstruction over dead latents."""
        dead = self.num_batches_not_active >= self.n_batches_to_dead
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return acts_norm.new_zeros(())
        pre_dead = self.pre_activations(acts_norm)[:, dead]
        keep = min(self.top_k_aux, n_dead)
        values, indices = pre_dead.topk(keep, dim=-1)
        z_dead = torch.zeros_like(pre_dead).scatter(-1, indices, values)
        dead_decoder = self.W_dec[:, dead]
        aux_recon = z_dead @ dead_decoder.t()
        residual = acts_norm.detach() - recon_norm.detach()
        return self.aux_penalty * (aux_recon - residual).square().mean()

    @torch.no_grad()
    def project_decoder_grad_and_normalise_(self) -> None:
        normed = self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
        if self.W_dec.grad is not None:
            parallel = (self.W_dec.grad * normed).sum(dim=0, keepdim=True) * normed
            self.W_dec.grad.sub_(parallel)
        self.W_dec.copy_(normed)

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8)
