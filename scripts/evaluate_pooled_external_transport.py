#!/usr/bin/env python
"""Evaluate matched-scale SAE transport on external pooled activations."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SAE_ROOT = ROOT / "results/sae_reconciliation/matched_scale_v1"
ACT_ROOT = ROOT / "results/activations_external_pooled_smoke"
ACT_ROOT_V2 = ROOT / "results/activations_external_pooled_smoke_v2"
OUT = ROOT / "results/multicohort/pooled_sae_transport"
COHORTS = ("chapman_f", "cpsc_f", "ningbo_f", "mimic_f")


def load_pooled(index_dir: Path) -> np.ndarray:
    arrays = []
    with (index_dir / "shards.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = Path(row["pooled_file"])
        if not path.is_absolute():
            path = ROOT / path
        arrays.append(np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32))
    if not arrays:
        raise FileNotFoundError(f"No pooled activation shards indexed in {index_dir}")
    return np.concatenate(arrays, axis=0)


def evaluate(checkpoint: Path, acts: np.ndarray, batch_size: int = 256) -> dict[str, float]:
    import torch

    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    saved = torch.load(checkpoint, map_location="cuda")
    config = saved["config"]
    sae = BatchTopKSAE(acts.shape[1], int(config["n_features"]), int(config["k"])).cuda()
    sae.load_state_dict(saved["model"])
    sae.eval()
    finite_rows = np.isfinite(acts).all(axis=1)
    finite_fraction = float(finite_rows.mean())
    acts = acts[finite_rows]
    if not len(acts):
        return {
            "external_recon_R2": float("nan"), "external_dead_fraction": float("nan"),
            "external_l0": float("nan"), "normalized_mean_shift_rms": float("nan"),
            "normalized_variance_mean": float("nan"), "finite_activation_fraction": finite_fraction,
        }
    normalized = (acts - sae.mu.detach().cpu().numpy()) / sae.sigma.detach().cpu().numpy()
    mean = normalized.mean(axis=0, keepdims=True)
    sse = 0.0
    sst = float(((normalized - mean) ** 2).sum())
    fired = np.zeros(sae.N, dtype=bool)
    active_total = 0
    with torch.no_grad():
        for lo in range(0, len(acts), batch_size):
            raw = torch.as_tensor(acts[lo : lo + batch_size], dtype=torch.float32, device="cuda")
            recon, z, norm = sae(raw)
            sse += float((norm - recon).square().sum().item())
            active = (z > 0).cpu().numpy()
            fired |= active.any(axis=0)
            active_total += int(active.sum())
    return {
        "external_recon_R2": 1.0 - sse / max(sst, 1e-12),
        "external_dead_fraction": float(1.0 - fired.mean()),
        "external_l0": float(active_total / len(acts)),
        "normalized_mean_shift_rms": float(np.sqrt(np.mean(mean**2))),
        "normalized_variance_mean": float(normalized.var(axis=0).mean()),
        "finite_activation_fraction": finite_fraction,
    }


def deterministic_ptb_reference(feature_suffix: str, n: int = 512) -> np.ndarray:
    acts = np.load(ROOT / f"results/probe_features/{feature_suffix}/pooled.npy", mmap_mode="r")
    manifest = pd.read_csv(ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv")
    candidates = manifest.index[manifest.split.eq("test")].to_numpy()
    if len(candidates) < n:
        raise RuntimeError(f"PTB test split has only {len(candidates)} rows")
    ids = manifest.loc[candidates, "ecg_id"].astype(str).to_numpy()
    scores = np.asarray([
        int.from_bytes(hashlib.sha256(f"matched-dead-reference-v1:{record_id}".encode()).digest()[:8], "big")
        for record_id in ids
    ], dtype=np.uint64)
    selected = candidates[np.argsort(scores)[:n]]
    return np.asarray(acts[selected], dtype=np.float32)


def main() -> None:
    manifest = pd.read_csv(SAE_ROOT / "training_manifest.csv")
    source_profile = pd.read_csv(SAE_ROOT / "matched_scale_model_profile.csv").set_index("model")
    rows = []
    acts_cache = {}
    ptb_cache = {}
    for cell in manifest.itertuples(index=False):
        for cohort in COHORTS:
            key = (cell.feature_suffix, cohort)
            corrected = ACT_ROOT_V2 / cell.feature_suffix / cohort
            index_dir = corrected if (corrected / "shards.csv").exists() else ACT_ROOT / cell.feature_suffix / cohort
            if key not in acts_cache:
                acts_cache[key] = load_pooled(index_dir)
            acts = acts_cache[key]
            if len(acts) != 512:
                raise RuntimeError(f"{key}: expected 512 activations, got {len(acts)}")
            if acts.shape[1] != int(cell.d_hidden):
                raise RuntimeError(f"{key}: expected width {cell.d_hidden}, got {acts.shape[1]}")
            checkpoint = Path(cell.checkpoint)
            ptb_metrics = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
            if cell.feature_suffix not in ptb_cache:
                ptb_cache[cell.feature_suffix] = deterministic_ptb_reference(cell.feature_suffix)
            ptb_matched = evaluate(checkpoint, ptb_cache[cell.feature_suffix])
            metric = evaluate(checkpoint, acts)
            dead_shift = metric["external_dead_fraction"] - ptb_matched["external_dead_fraction"]
            integrity_pass = metric["finite_activation_fraction"] == 1.0
            source_eligible = bool(source_profile.loc[cell.model, "matched_scale_primary_eligible"])
            rows.append(
                {
                    "model": cell.model,
                    "feature_suffix": cell.feature_suffix,
                    "cohort": cohort,
                    "seed": int(cell.seed),
                    "records": len(acts),
                    "d_hidden": int(cell.d_hidden),
                    "N": int(cell.N),
                    "k": int(cell.k),
                    "ptb_recon_R2": float(ptb_metrics["explained_variance"]),
                    "ptb_dead_fraction": float(ptb_metrics["dead_fraction"]),
                    "ptb_matched512_recon_R2": ptb_matched["external_recon_R2"],
                    "ptb_matched512_dead_fraction": ptb_matched["external_dead_fraction"],
                    **metric,
                    "dead_fraction_shift": dead_shift,
                    "integrity_pass": integrity_pass,
                    "source_fidelity_eligible": source_eligible,
                    "recon_pass_085": metric["external_recon_R2"] >= 0.85,
                    "recon_pass_090": metric["external_recon_R2"] >= 0.90,
                    "dead_shift_pass": dead_shift <= 0.20,
                }
            )
    frame = pd.DataFrame(rows)
    frame["seed_transport_pass"] = (
        frame.source_fidelity_eligible & frame.integrity_pass & frame.recon_pass_085 & frame.dead_shift_pass
    )
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / "pooled_transport_seed_gate.csv", index=False)
    profile = frame.groupby(["model", "feature_suffix", "cohort"], as_index=False).agg(
        seeds=("seed", "nunique"),
        external_recon_R2_mean=("external_recon_R2", "mean"),
        external_recon_R2_min=("external_recon_R2", "min"),
        finite_activation_fraction_min=("finite_activation_fraction", "min"),
        ptb_matched512_dead_fraction_mean=("ptb_matched512_dead_fraction", "mean"),
        external_dead_fraction_max=("external_dead_fraction", "max"),
        dead_fraction_shift_max=("dead_fraction_shift", "max"),
        external_l0_mean=("external_l0", "mean"),
        mean_shift_rms=("normalized_mean_shift_rms", "mean"),
        variance_mean=("normalized_variance_mean", "mean"),
        pass_085_seeds=("recon_pass_085", "sum"),
        pass_090_seeds=("recon_pass_090", "sum"),
        transport_pass_seeds=("seed_transport_pass", "sum"),
        integrity_pass_seeds=("integrity_pass", "sum"),
        source_fidelity_eligible=("source_fidelity_eligible", "all"),
    )
    profile["primary_transport_eligible"] = profile.transport_pass_seeds.eq(3)
    profile["strict_090_eligible"] = profile.pass_090_seeds.eq(3) & profile.primary_transport_eligible
    profile.to_csv(OUT / "pooled_transport_model_cohort_gate.csv", index=False)
    lines = [
        "# Matched-Scale Pooled SAE External Transport Gate",
        "",
        "- Primary external reconstruction floor: 0.85 across all three SAE seeds.",
        "- Strict sensitivity floor: 0.90 across all three SAE seeds.",
        "- External dead-feature fraction may increase by at most 0.20 relative to PTB-XL.",
        "- Dead fractions use deterministic 512-record samples in both PTB-XL and the external cohort.",
        "- Non-finite activations fail the integrity gate; source SAEs must pass the PTB fidelity gate.",
        "- Smoke sample: 512 deterministic records per cohort.",
        "",
        "| Model | Cohort | Recon mean | Recon min | Dead shift max | Primary pass | Strict 0.90 pass |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in profile.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.cohort} | {row.external_recon_R2_mean:.4f} | "
            f"{row.external_recon_R2_min:.4f} | {row.dead_fraction_shift_max:.4f} | "
            f"{bool(row.primary_transport_eligible)} | {bool(row.strict_090_eligible)} |"
        )
    (OUT / "pooled_transport_report.md").write_text("\n".join(lines) + "\n")
    print(profile.to_string(index=False))


if __name__ == "__main__":
    main()
