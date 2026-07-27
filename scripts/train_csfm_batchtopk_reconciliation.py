#!/usr/bin/env python
from __future__ import annotations

import argparse, csv, json, os, random, signal
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/layer6_mean.npy")
    p.add_argument("--manifest", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/manifest.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results/sae_reconciliation/lbbb_fig6/checkpoints/batchtopk_8192_k128.pt")
    p.add_argument("--n-features", type=int, default=8192)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=4311)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def metrics(model, acts, batch_size, device):
    sse = sst = 0.0
    fired = np.zeros(model.N, dtype=bool)
    mean = acts.mean(axis=0, keepdims=True)
    model.eval()
    with __import__("torch").no_grad():
        for lo in range(0, len(acts), batch_size):
            raw = __import__("torch").as_tensor(acts[lo:lo+batch_size], device=device)
            recon, z, norm = model(raw)
            sse += float((norm-recon).square().sum().item())
            centered = raw - __import__("torch").as_tensor(mean, device=device)
            centered = centered / model.sigma
            sst += float(centered.square().sum().item())
            fired |= (z > 0).any(dim=0).cpu().numpy()
    return {"explained_variance": 1-sse/max(sst, 1e-12), "dead_fraction": float(1-fired.mean())}


def main():
    a = parse_args()
    import torch
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows = list(csv.DictReader(a.manifest.open()))
    acts = np.load(a.acts, mmap_mode="r")
    train_idx = np.array([i for i,r in enumerate(rows) if r["split"] == "train"])
    val_idx = np.array([i for i,r in enumerate(rows) if r["split"] == "val"])
    train = np.asarray(acts[train_idx], dtype=np.float32)
    val = np.asarray(acts[val_idx], dtype=np.float32)
    model = BatchTopKSAE(train.shape[1], a.n_features, a.k).to(a.device)
    model.set_normalisation(torch.from_numpy(train.mean(0)).to(a.device), torch.from_numpy(train.std(0)).to(a.device))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.9, 0.99))
    start = 0
    rng = np.random.default_rng(a.seed)
    if a.out.exists():
        ckpt = torch.load(a.out, map_location=a.device)
        model.load_state_dict(ckpt["model"]); opt.load_state_dict(ckpt["optimizer"])
        start = int(ckpt["step"])
        if "numpy_rng_state" in ckpt:
            rng.bit_generator.state = ckpt["numpy_rng_state"]
        else:
            # Compatibility with checkpoints created before exact RNG-state saves.
            rng = np.random.default_rng(a.seed + start)
        print(f"resuming at step {start}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_requested = {"value": False}
    signal.signal(signal.SIGUSR1, lambda *_: checkpoint_requested.__setitem__("value", True))

    def save(step, final=False):
        payload = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
                   "numpy_rng_state": rng.bit_generator.state,
                   "config": vars(a), "architecture": "BatchTopK", "final": final}
        tmp = a.out.with_suffix(a.out.suffix + ".tmp")
        torch.save(payload, tmp); os.replace(tmp, a.out)

    model.train()
    losses = []
    for step in range(start, a.steps):
        idx = rng.choice(len(train), size=min(a.batch_size, len(train)), replace=False)
        raw = torch.from_numpy(train[idx]).to(a.device)
        recon, _, norm = model(raw)
        recon_loss = (recon-norm).square().mean()
        aux_loss = model.auxiliary_loss(norm, recon)
        loss = recon_loss + aux_loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100000.0)
        model.project_decoder_grad_and_normalise_()
        opt.step(); model.normalise_decoder_()
        losses.append(float(loss.item()))
        done = step + 1
        if done % 50 == 0:
            print(f"step={done} loss={np.mean(losses[-50:]):.6f} recon={recon_loss.item():.6f} aux={aux_loss.item():.6f}", flush=True)
        if done % a.checkpoint_every == 0 or checkpoint_requested["value"]:
            save(done)
        if checkpoint_requested["value"]:
            print("checkpoint saved after preemption warning; awaiting Slurm requeue", flush=True)
            checkpoint_requested["value"] = False
    save(a.steps, final=True)
    report = metrics(model, val, a.batch_size, a.device)
    report.update({"step": a.steps, "N": a.n_features, "k": a.k, "d": train.shape[1],
                   "target_reference_ev": 0.968, "target_reference_dead_fraction": 0.21,
                   "normalisation": "train-only per-dimension z-score",
                   "auxiliary_loss": "official BatchTopK dead-latent residual loss: aux_k=512, coefficient=1/32, dead_after=5_batches",
                   "checkpoint_provenance": "self-trained architecture-matched; paper checkpoint unavailable"})
    metrics_path = a.out.with_suffix(".metrics.json")
    metrics_tmp = metrics_path.with_suffix(metrics_path.suffix + ".tmp")
    metrics_tmp.write_text(json.dumps(report, indent=2)+"\n")
    os.replace(metrics_tmp, metrics_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
