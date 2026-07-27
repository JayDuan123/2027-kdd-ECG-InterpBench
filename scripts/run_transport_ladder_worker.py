#!/usr/bin/env python
"""Run one model/cohort/seed transport-ladder worker on held-out activations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "external_benchmark_v1"
ACT_ROOT = ROOT / "results" / "activations_external_full_v1" / "pooled"
SOURCE_ROOT = ROOT / "results" / "sae_reconciliation" / "matched_scale_v1"
OUT = ROOT / "results" / "benchmark_extension_v1" / "transport_ladder" / "workers"

from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE  # noqa: E402
from scripts.analyze_external_dose_direction import matched_random_groups  # noqa: E402
from scripts.run_external_sae_steering_task import (  # noqa: E402
    load_activations,
    threshold_at_specificity,
)

SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "steering_summary", ROOT / "scripts" / "summarize_steering_benchmark.py"
)
SUMMARY = importlib.util.module_from_spec(SUMMARY_SPEC)
assert SUMMARY_SPEC.loader is not None
SUMMARY_SPEC.loader.exec_module(SUMMARY)

MODELS = (
    "csfm_cu118_commons", "cardiac_fm_cu118_commons", "ecg_fm_cu118_commons",
    "ecg_jepa_cu118_commons", "hubert_ecg_cu118_commons", "st_mem_cu118_commons",
)
COHORTS = ("chapman_f", "cpsc_f", "ningbo_f", "mimic_f")
SEEDS = (4311, 4312, 4313)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fewshot-sizes", default="128,512,2048")
    parser.add_argument("--max-train-eval", type=int, default=8192)
    parser.add_argument("--max-val-eval", type=int, default=4096)
    parser.add_argument("--max-test-eval", type=int, default=8192)
    parser.add_argument("--covariance-sample", type=int, default=8192)
    parser.add_argument("--n-random", type=int, default=20)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def task_cell(index: int) -> tuple[str, str, int]:
    cells = [(model, cohort, seed) for model in MODELS for cohort in COHORTS for seed in SEEDS]
    if index < 0 or index >= len(cells):
        raise ValueError(f"task-index must be in 0..{len(cells)-1}")
    return cells[index]


def stable_subset(indices: np.ndarray, ids: np.ndarray, n: int, key: str) -> np.ndarray:
    indices = np.asarray(indices, dtype=int)
    if n <= 0 or len(indices) <= n:
        return indices
    scores = np.asarray(
        [
            int.from_bytes(hashlib.sha256(f"{key}:{ids[index]}".encode()).digest()[:8], "big")
            for index in indices
        ],
        dtype=np.uint64,
    )
    return indices[np.argsort(scores)[:n]]


def load_sae(saved: dict, device: str) -> BatchTopKSAE:
    config = saved["config"]
    model = BatchTopKSAE(
        int(saved["model"]["mu"].numel()), int(config["n_features"]), int(config["k"])
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def encode_decode(
    sae: BatchTopKSAE, x: np.ndarray, device: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    codes, recon = [], []
    sae.eval()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            raw = torch.as_tensor(np.asarray(x[lo : lo + batch_size]), dtype=torch.float32, device=device)
            z = sae.encode(raw)
            codes.append(z.cpu().numpy().astype(np.float32))
            recon.append(sae.decode(z).cpu().numpy().astype(np.float32))
    return np.concatenate(codes), np.concatenate(recon)


def covariance_map(
    source: np.ndarray, target: np.ndarray, source_mean: np.ndarray, target_mean: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_centered = np.asarray(source - source_mean, dtype=np.float64)
    target_centered = np.asarray(target - target_mean, dtype=np.float64)
    source_cov = source_centered.T @ source_centered / max(len(source_centered) - 1, 1)
    target_cov = target_centered.T @ target_centered / max(len(target_centered) - 1, 1)

    def root(matrix: np.ndarray, inverse: bool) -> np.ndarray:
        values, vectors = np.linalg.eigh(matrix)
        floor = max(float(np.mean(np.maximum(values, 0))) * 1e-4, 1e-8)
        values = np.maximum(values, floor)
        power = -0.5 if inverse else 0.5
        return (vectors * (values**power)[None, :]) @ vectors.T

    transform = root(target_cov, True) @ root(source_cov, False)
    return transform.astype(np.float32), np.linalg.inv(transform).astype(np.float32)


def fine_tune(
    source_saved: dict,
    aligned: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: str,
) -> BatchTopKSAE:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = load_sae(copy.deepcopy(source_saved), device)
    model.num_batches_not_active.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
    rng = np.random.default_rng(seed)
    model.train()
    for step in range(steps):
        count = min(batch_size, len(aligned))
        index = rng.choice(len(aligned), size=count, replace=False)
        raw = torch.as_tensor(aligned[index], dtype=torch.float32, device=device)
        recon, _, normalized = model(raw)
        recon_loss = (recon - normalized).square().mean()
        loss = recon_loss + model.auxiliary_loss(normalized, recon)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100000.0)
        model.project_decoder_grad_and_normalise_(); optimizer.step(); model.normalise_decoder_()
        if (step + 1) % 250 == 0:
            print(f"fewshot seed={seed} step={step+1}/{steps} loss={loss.item():.6f}", flush=True)
    model.eval()
    return model


def sampled_decoder_alignment(
    source: BatchTopKSAE, target: BatchTopKSAE, seed: int, device: str, sample: int = 512
) -> tuple[float, float]:
    import torch
    source_decoder = source.decoder_directions().detach().to(device)
    target_decoder = target.decoder_directions().detach().to(device)
    rng = np.random.default_rng(seed)
    source_idx = np.sort(rng.choice(source.N, size=min(sample, source.N), replace=False))
    with torch.no_grad():
        similarity = torch.abs(source_decoder[:, source_idx].T @ target_decoder)
        max_cosine = float(similarity.max(dim=1).values.mean().item())
        identity = float(
            torch.abs((source_decoder[:, source_idx] * target_decoder[:, source_idx]).sum(dim=0)).mean().item()
        )
    return max_cosine, identity


def transform_arrays(
    x: np.ndarray,
    mode: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    coral: np.ndarray | None,
) -> np.ndarray:
    if mode == "identity":
        return np.asarray(x, dtype=np.float32)
    if mode == "diagonal":
        return (((x - target_mean) / target_std) * source_std + source_mean).astype(np.float32)
    if mode == "coral":
        assert coral is not None
        return ((x - target_mean) @ coral + source_mean).astype(np.float32)
    raise ValueError(mode)


def inverse_arrays(
    x: np.ndarray,
    mode: str,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    coral_inverse: np.ndarray | None,
) -> np.ndarray:
    if mode == "identity":
        return np.asarray(x, dtype=np.float32)
    if mode == "diagonal":
        return (((x - source_mean) / source_std) * target_std + target_mean).astype(np.float32)
    if mode == "coral":
        assert coral_inverse is not None
        return ((x - source_mean) @ coral_inverse + target_mean).astype(np.float32)
    raise ValueError(mode)


def decoder_in_target_space(
    sae: BatchTopKSAE,
    mode: str,
    target_std: np.ndarray,
    source_std: np.ndarray,
    coral_inverse: np.ndarray | None,
) -> np.ndarray:
    decoder = sae.W_dec.detach().cpu().numpy() * sae.sigma.detach().cpu().numpy()[:, None]
    if mode == "identity":
        return decoder
    if mode == "diagonal":
        return decoder * (target_std / source_std)[:, None]
    if mode == "coral":
        assert coral_inverse is not None
        return coral_inverse.T @ decoder
    raise ValueError(mode)


def evaluate_method(
    method: str,
    sae: BatchTopKSAE,
    transform_mode: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    labels_train: np.ndarray,
    labels_val: np.ndarray,
    labels_test: np.ndarray,
    group_ids_test: np.ndarray,
    names: list[str],
    heads: dict,
    scaler,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    source_mean: np.ndarray,
    source_std: np.ndarray,
    coral: np.ndarray | None,
    coral_inverse: np.ndarray | None,
    source_sae: BatchTopKSAE,
    seed: int,
    device: str,
    batch_size: int,
    n_random: int,
) -> tuple[dict, list[dict]]:
    transformed_train = transform_arrays(x_train, transform_mode, target_mean, target_std, source_mean, source_std, coral)
    transformed_val = transform_arrays(x_val, transform_mode, target_mean, target_std, source_mean, source_std, coral)
    transformed_test = transform_arrays(x_test, transform_mode, target_mean, target_std, source_mean, source_std, coral)
    z_train, _ = encode_decode(sae, transformed_train, device, batch_size)
    z_val, recon_val_pre = encode_decode(sae, transformed_val, device, batch_size)
    z_test, recon_test_pre = encode_decode(sae, transformed_test, device, batch_size)
    del z_val
    recon_val = inverse_arrays(recon_val_pre, transform_mode, target_mean, target_std, source_mean, source_std, coral_inverse)
    recon_test = inverse_arrays(recon_test_pre, transform_mode, target_mean, target_std, source_mean, source_std, coral_inverse)
    mean_test = x_test.mean(axis=0, keepdims=True)
    recon_r2 = 1.0 - float(((x_test - recon_test) ** 2).sum()) / max(float(((x_test - mean_test) ** 2).sum()), 1e-12)
    dead_fraction = float(1.0 - (z_test > 0).any(axis=0).mean())
    l0 = float((z_test > 0).sum(axis=1).mean())
    raw_logits = np.column_stack([heads[name]["clf"].decision_function(scaler.transform(x_test)) for name in names])
    recon_logits = np.column_stack([heads[name]["clf"].decision_function(scaler.transform(recon_test)) for name in names])
    recon_val_logits = np.column_stack([heads[name]["clf"].decision_function(scaler.transform(recon_val)) for name in names])
    raw_aurocs, recon_aurocs = [], []
    for head, name in enumerate(names):
        raw_aurocs.append(roc_auc_score(labels_test[:, head], raw_logits[:, head]))
        recon_aurocs.append(roc_auc_score(labels_test[:, head], recon_logits[:, head]))
    retention = np.asarray(recon_aurocs) / np.maximum(raw_aurocs, 1e-8)
    max_cosine, identity_cosine = sampled_decoder_alignment(source_sae, sae, seed, device)
    quality = {
        "method": method, "recon_r2": recon_r2, "dead_fraction": dead_fraction, "l0": l0,
        "raw_head_auroc_mean": float(np.mean(raw_aurocs)),
        "recon_head_auroc_mean": float(np.mean(recon_aurocs)),
        "readout_retention_median": float(np.median(retention)),
        "decoder_source_max_cosine_sample_mean": max_cosine,
        "decoder_identity_cosine_sample_mean": identity_cosine,
        "train_eval_records": len(x_train), "val_eval_records": len(x_val), "test_eval_records": len(x_test),
    }

    decoder_target = decoder_in_target_space(sae, transform_mode, target_std, source_std, coral_inverse)
    head_coefficients = np.vstack([np.asarray(heads[name]["clf"].coef_).reshape(-1) / scaler.scale_ for name in names])
    gradients = head_coefficients @ decoder_target
    rankings = []
    for head in range(len(names)):
        positive = labels_train[:, head] == 1
        rankings.append(np.argsort((z_train[positive] * gradients[head]).mean(axis=0))[::-1])
    rankings = np.asarray(rankings)
    freq = (z_train > 0).mean(axis=0)
    counts = (z_train > 0).sum(axis=0)
    mag = np.divide(z_train.sum(axis=0), counts, out=np.zeros(sae.N), where=counts > 0)
    centroid = z_train.mean(axis=0)
    thresholds = np.asarray(
        [threshold_at_specificity(labels_val[:, head].astype(int), recon_val_logits[:, head]) for head in range(len(names))]
    )
    steering_rows = []
    for target_j, target in enumerate(names):
        selected = rankings[target_j, :5]
        random_groups = matched_random_groups(
            selected, freq, mag, rankings, n_random,
            seed + sum(map(ord, method + target)),
        )

        def delta(indices: np.ndarray) -> np.ndarray:
            idx = np.asarray(indices, dtype=int)
            dz = centroid[idx][None, :] - z_test[:, idx]
            return np.column_stack([dz @ gradients[head, idx] for head in range(len(names))])

        target_delta = delta(selected)
        random_delta = np.stack([delta(group) for group in random_groups], axis=1)
        data = {
            "patient_ids": group_ids_test.astype("U64"),
            "target_names": np.asarray(names, dtype="U64"),
            "target_types": np.asarray(["binary"] * len(names), dtype="U16"),
            "labels": labels_test.astype(np.float32),
            "baseline_logits": recon_logits.astype(np.float32),
            "top5_delta": target_delta.astype(np.float32),
            "random_top5_delta": random_delta.astype(np.float32),
            "thresholds_95spec": thresholds.astype(np.float32),
            "continuous_target_means": np.full(len(names), np.nan),
            "continuous_target_stds": np.full(len(names), np.nan),
        }
        result = {
            "target": target, "focus_thresholds_train": {name: 1.0 for name in names}
        }
        point = SUMMARY.one_stats(data, result, [])
        steering_rows.append(
            {
                "method": method, "target": target, "top5_atoms": "|".join(map(str, selected.tolist())),
                "ste": point["ste"], "otd_mean": point["otd_mean"],
                "selectivity_margin": point["selectivity_margin"], "wbi": point["wbi"],
                "tier1_excess_attribution": point["tier1_excess_attribution"],
                "excess_selectivity": point["excess_selectivity"],
                "wbi_improvement": point["wbi_improvement"],
                "behavior_effect": point["behavior_effect"], "behavior_excess": point["behavior_excess"],
            }
        )
    return quality, steering_rows


def main() -> None:
    args = parse_args()
    model_suffix, cohort, seed = task_cell(args.task_index)
    worker_out = args.out / model_suffix / cohort / f"seed{seed}"
    complete = worker_out / "complete.json"
    if complete.exists() and not args.force:
        print(f"already complete: {complete}")
        return
    worker_out.mkdir(parents=True, exist_ok=True)

    import torch
    bundle = joblib.load(BASE / model_suffix / cohort / "frozen_heads.joblib")
    names = list(bundle["targets"]); heads = bundle["heads"]; scaler = bundle["scaler"]
    record_ids = np.asarray(bundle["record_ids"]); group_ids = np.asarray(bundle.get("group_ids", record_ids))
    split = np.asarray(bundle["split"])
    x, loaded_ids = load_activations(ACT_ROOT / model_suffix / cohort)
    if not np.array_equal(record_ids.astype(str), loaded_ids.astype(str)):
        raise RuntimeError("Activation/head record order mismatch")
    train_all = np.where(split == "train")[0]; val_all = np.where(split == "val")[0]; test_all = np.where(split == "test")[0]
    train_idx = stable_subset(train_all, record_ids, args.max_train_eval, f"transport-train:{model_suffix}:{cohort}")
    val_idx = stable_subset(val_all, record_ids, args.max_val_eval, f"transport-val:{model_suffix}:{cohort}")
    test_idx = stable_subset(test_all, record_ids, args.max_test_eval, f"transport-test:{model_suffix}:{cohort}")
    label_matrix = np.column_stack([np.asarray(heads[name]["labels"], dtype=float) for name in names])
    x_train, x_val, x_test = (np.asarray(x[index], dtype=np.float32) for index in (train_idx, val_idx, test_idx))
    labels_train, labels_val, labels_test = (label_matrix[index] for index in (train_idx, val_idx, test_idx))

    training = pd.read_csv(SOURCE_ROOT / "training_manifest.csv")
    source_row = training[(training.feature_suffix == model_suffix) & (training.seed == seed)].iloc[0]
    source_checkpoint = Path(source_row.checkpoint)
    source_saved = torch.load(source_checkpoint, map_location="cpu")
    source_sae = load_sae(copy.deepcopy(source_saved), args.device)
    source_mean = source_sae.mu.detach().cpu().numpy(); source_std = source_sae.sigma.detach().cpu().numpy()
    # Alignment statistics are fit on the complete training partition.  The
    # smaller train_idx subset is reserved for bounded-cost evaluation only.
    target_train = np.asarray(x[train_all], dtype=np.float32)
    target_mean = target_train.mean(axis=0)
    target_std = np.maximum(target_train.std(axis=0), 1e-6)

    source_acts = np.load(ROOT / "results" / "probe_features" / model_suffix / "pooled.npy", mmap_mode="r")
    source_manifest = pd.read_csv(ROOT / "results" / "sae_reconciliation" / "steering_benchmark_multimodel_v1" / "manifest.csv")
    source_train_all = np.where(source_manifest.split.eq("train").to_numpy())[0]
    source_ids = source_manifest.ecg_id.astype(str).to_numpy()
    source_cov_idx = stable_subset(source_train_all, source_ids, args.covariance_sample, f"transport-source-cov:{model_suffix}")
    target_cov_idx = stable_subset(
        train_all, record_ids, args.covariance_sample,
        f"transport-target-cov:{model_suffix}:{cohort}",
    )
    coral, coral_inverse = covariance_map(
        np.asarray(source_acts[source_cov_idx], dtype=np.float32),
        np.asarray(x[target_cov_idx], dtype=np.float32), source_mean, target_mean,
    )
    del target_train

    adapted_manifest = pd.read_csv(BASE / "cohort_adapted_sae_manifest.csv")
    adapted_row = adapted_manifest[
        (adapted_manifest.model_suffix == model_suffix) & (adapted_manifest.cohort == cohort) & (adapted_manifest.seed == seed)
    ].iloc[0]
    adapted_saved = torch.load(Path(adapted_row.checkpoint), map_location="cpu")

    methods: list[tuple[str, BatchTopKSAE, str, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = [
        ("frozen", source_sae, "identity", target_mean, target_std, None, None),
        ("diagonal_full_train", load_sae(copy.deepcopy(source_saved), args.device), "diagonal", target_mean, target_std, None, None),
        ("coral_full_train", load_sae(copy.deepcopy(source_saved), args.device), "coral", target_mean, target_std, coral, coral_inverse),
    ]
    fewshot_sizes = [int(value) for value in args.fewshot_sizes.split(",") if value.strip()]
    for n in fewshot_sizes:
        sample_idx = stable_subset(train_all, record_ids, n, f"transport-fewshot:{model_suffix}:{cohort}:{seed}")
        sample = np.asarray(x[sample_idx], dtype=np.float32)
        sample_mean = sample.mean(axis=0); sample_std = np.maximum(sample.std(axis=0), 1e-6)
        aligned = transform_arrays(sample, "diagonal", sample_mean, sample_std, source_mean, source_std, None)
        checkpoint = worker_out / f"fewshot_n{n}.pt"
        if checkpoint.exists() and not args.force:
            few_saved = torch.load(checkpoint, map_location="cpu")
            few_model = load_sae(few_saved, args.device)
        else:
            few_model = fine_tune(
                source_saved, aligned, args.steps, args.batch_size, args.lr,
                seed + n + sum(map(ord, model_suffix + cohort)), args.device,
            )
            payload = {
                "model": {key: value.detach().cpu() for key, value in few_model.state_dict().items()},
                "config": source_saved["config"], "method": f"fewshot_n{n}", "records": n,
                "steps": args.steps, "source_checkpoint": str(source_checkpoint),
                "target_mean": sample_mean, "target_std": sample_std,
            }
            torch.save(payload, checkpoint)
        methods.append((f"fewshot_n{n}", few_model, "diagonal", sample_mean, sample_std, None, None))
    methods.append(("cohort_adapted_full", load_sae(adapted_saved, args.device), "identity", target_mean, target_std, None, None))

    quality_rows, steering_rows = [], []
    for method, sae, mode, method_mean, method_std, method_coral, method_coral_inverse in methods:
        quality, steering = evaluate_method(
            method, sae, mode, x_train, x_val, x_test, labels_train, labels_val, labels_test,
            group_ids[test_idx], names, heads, scaler, method_mean, method_std, source_mean, source_std,
            method_coral, method_coral_inverse, source_sae, seed + sum(map(ord, method)),
            args.device, args.batch_size, args.n_random,
        )
        quality.update({"model": source_row.model, "model_suffix": model_suffix, "cohort": cohort, "seed": seed})
        quality_rows.append(quality)
        for row in steering:
            row.update({"model": source_row.model, "model_suffix": model_suffix, "cohort": cohort, "seed": seed})
            steering_rows.append(row)
        print(f"transport evaluated {model_suffix}/{cohort}/seed{seed}/{method}", flush=True)

    pd.DataFrame(quality_rows).to_csv(worker_out / "quality_metrics.csv", index=False)
    pd.DataFrame(steering_rows).to_csv(worker_out / "steering_metrics.csv", index=False)
    metadata = {
        "schema_version": 1, "task_index": args.task_index, "model_suffix": model_suffix,
        "cohort": cohort, "seed": seed, "methods": [item[0] for item in methods],
        "fewshot_steps": args.steps, "fewshot_sizes": fewshot_sizes,
        "train_eval_records": len(train_idx), "val_eval_records": len(val_idx), "test_eval_records": len(test_idx),
        "alignment_train_records": len(train_all),
        "covariance_sample": min(args.covariance_sample, len(train_all)),
        "source_checkpoint": str(source_checkpoint), "adapted_checkpoint": str(adapted_row.checkpoint),
        "status": "complete",
    }
    complete.write_text(json.dumps(metadata, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)) + "\n")
    print(json.dumps(metadata, default=str))


if __name__ == "__main__":
    main()
