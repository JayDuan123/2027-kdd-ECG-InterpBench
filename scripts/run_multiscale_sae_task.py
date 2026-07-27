#!/usr/bin/env python
"""Train and evaluate one cell of the multi-scale SAE benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import signal
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multiscale_sae import (
    canonical_config_hash,
    correlation_from_sufficient_statistics,
    read_csv,
    selected_concept_metrics,
    standardized_concepts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument(
        "--concept-registry",
        type=Path,
        default=ROOT / "configs/concepts.csv",
    )
    parser.add_argument("--preserve-missing-concepts", action="store_true")
    parser.add_argument("--complete-case-evaluation", action="store_true")
    return parser.parse_args()


def manifest_row(path: Path, task_index: int) -> dict[str, str]:
    matches = [row for row in read_csv(path) if int(row["task_index"]) == task_index]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for task {task_index}, found {len(matches)}")
    return matches[0]


def typed_config(row: dict[str, str]) -> dict[str, Any]:
    config: dict[str, Any] = dict(row)
    for key in ("task_index", "layer", "n_layers", "d_hidden", "expansion_E", "N", "k", "seed", "steps", "batch_size"):
        config[key] = int(float(config[key]))
    for key in ("relative_depth", "actual_relative_depth", "k_over_d", "k_over_N", "learning_rate"):
        config[key] = float(config[key])
    return config


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty concept table")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def save_checkpoint(path: Path, model, optimizer, step: int, config: dict[str, Any], rng, final: bool) -> None:
    import torch

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": int(step),
        "config": config,
        "architecture": "BatchTopK",
        "final": bool(final),
    }
    if not final:
        payload.update(
            {
                "optimizer": optimizer.state_dict(),
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
            }
        )
        if torch.cuda.is_available():
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(device=next(model.parameters()).device)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(config: dict[str, Any], acts: np.ndarray, train_idx: np.ndarray, device: str, checkpoint_every: int):
    import torch

    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_acts = np.asarray(acts[train_idx], dtype=np.float32)
    d_hidden = int(train_acts.shape[1])
    if d_hidden != int(config["d_hidden"]):
        raise ValueError(f"hidden dimension mismatch: data={d_hidden}, manifest={config['d_hidden']}")
    model = BatchTopKSAE(d_hidden, int(config["N"]), int(config["k"])).to(device)
    mean = np.mean(train_acts, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(train_acts, axis=0, dtype=np.float64).astype(np.float32)
    model.set_normalisation(torch.from_numpy(mean).to(device), torch.from_numpy(scale).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), betas=(0.9, 0.99))
    rng = np.random.default_rng(seed)
    checkpoint = Path(config["checkpoint"])
    start = 0

    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        saved_config = saved.get("config", {})
        if saved_config.get("config_hash") != config["config_hash"]:
            raise ValueError(f"checkpoint config hash mismatch: {checkpoint}")
        model.load_state_dict(saved["model"])
        start = int(saved.get("step", 0))
        if not saved.get("final", False):
            optimizer.load_state_dict(saved["optimizer"])
            if "numpy_rng_state" in saved:
                rng.bit_generator.state = saved["numpy_rng_state"]
            if "torch_rng_state" in saved:
                torch.set_rng_state(saved["torch_rng_state"].cpu())
            if torch.cuda.is_available() and "cuda_rng_state" in saved:
                torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), device=device)
        if bool(saved.get("final", False)) and start >= int(config["steps"]):
            model.eval()
            return model

    checkpoint_requested = {"value": False}
    terminate_requested = {"value": False}

    def handle_usr1(*_args) -> None:
        checkpoint_requested["value"] = True

    def handle_term(*_args) -> None:
        checkpoint_requested["value"] = True
        terminate_requested["value"] = True

    previous_usr1 = signal.getsignal(signal.SIGUSR1)
    previous_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGUSR1, handle_usr1)
    signal.signal(signal.SIGTERM, handle_term)

    batch_size = int(config["batch_size"])
    model.train()
    recent_losses: list[float] = []
    try:
        for step in range(start, int(config["steps"])):
            selected = rng.choice(len(train_acts), size=min(batch_size, len(train_acts)), replace=False)
            raw = torch.from_numpy(train_acts[selected]).to(device)
            reconstruction, _, normalized = model(raw)
            reconstruction_loss = (reconstruction - normalized).square().mean()
            auxiliary_loss = model.auxiliary_loss(normalized, reconstruction)
            loss = reconstruction_loss + auxiliary_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100000.0)
            model.project_decoder_grad_and_normalise_()
            optimizer.step()
            model.normalise_decoder_()
            recent_losses.append(float(loss.item()))
            done = step + 1
            if done % 50 == 0:
                print(
                    f"task={config['task_index']} step={done} loss={np.mean(recent_losses[-50:]):.6f} "
                    f"recon={reconstruction_loss.item():.6f} aux={auxiliary_loss.item():.6f}",
                    flush=True,
                )
            if done % checkpoint_every == 0 or checkpoint_requested["value"]:
                save_checkpoint(checkpoint, model, optimizer, done, config, rng, final=False)
                checkpoint_requested["value"] = False
            if terminate_requested["value"]:
                raise SystemExit(143)
    finally:
        signal.signal(signal.SIGUSR1, previous_usr1)
        signal.signal(signal.SIGTERM, previous_term)

    save_checkpoint(checkpoint, model, optimizer, int(config["steps"]), config, rng, final=True)
    model.eval()
    return model


def evaluate(model, acts: np.ndarray, indices: np.ndarray, concepts: np.ndarray, batch_size: int, device: str):
    import torch

    n_features = int(model.N)
    n_concepts = int(concepts.shape[1])
    count = 0
    sse = 0.0
    sst = 0.0
    active_total = 0.0
    fired_rows = np.zeros(n_features, dtype=np.float64)
    valid_count = np.zeros(n_concepts, dtype=np.int64)
    sum_z = np.zeros((n_features, n_concepts), dtype=np.float64)
    sum_z2 = np.zeros((n_features, n_concepts), dtype=np.float64)
    sum_y = np.zeros(n_concepts, dtype=np.float64)
    sum_y2 = np.zeros(n_concepts, dtype=np.float64)
    cross = np.zeros((n_features, n_concepts), dtype=np.float64)

    eval_mean = np.mean(np.asarray(acts[indices], dtype=np.float32), axis=0)
    mean_normalized = model.normalise(torch.from_numpy(eval_mean).to(device)).detach()
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(indices), batch_size):
            batch_indices = indices[lo : lo + batch_size]
            raw = torch.as_tensor(np.asarray(acts[batch_indices]), dtype=torch.float32, device=device)
            label_values = np.asarray(concepts[batch_indices], dtype=np.float32)
            valid = np.isfinite(label_values)
            labels = torch.as_tensor(np.nan_to_num(label_values), dtype=torch.float32, device=device)
            reconstruction, z, normalized = model(raw)
            sse += float((normalized - reconstruction).square().sum().item())
            sst += float((normalized - mean_normalized).square().sum().item())
            positive = z > 0
            active_total += float(positive.sum().item())
            fired_rows += positive.sum(dim=0).cpu().numpy().astype(np.float64)
            valid_tensor = torch.as_tensor(valid, dtype=torch.float32, device=device)
            valid_count += valid.sum(axis=0)
            sum_z += (z.t() @ valid_tensor).cpu().numpy().astype(np.float64)
            sum_z2 += (z.square().t() @ valid_tensor).cpu().numpy().astype(np.float64)
            sum_y += (labels * valid_tensor).sum(dim=0).cpu().numpy().astype(np.float64)
            sum_y2 += (labels.square() * valid_tensor).sum(dim=0).cpu().numpy().astype(np.float64)
            cross += (z.t() @ (labels * valid_tensor)).cpu().numpy().astype(np.float64)
            count += int(len(batch_indices))

    correlations = np.zeros((n_features, n_concepts), dtype=np.float64)
    for concept_index in range(n_concepts):
        correlations[:, concept_index : concept_index + 1] = correlation_from_sufficient_statistics(
            int(valid_count[concept_index]),
            sum_z[:, concept_index],
            sum_z2[:, concept_index],
            sum_y[concept_index : concept_index + 1],
            sum_y2[concept_index : concept_index + 1],
            cross[:, concept_index : concept_index + 1],
        )
    firing_rate = fired_rows / max(count, 1)
    metrics = {
        "n_records": count,
        "recon_R2": float(1.0 - sse / max(sst, 1e-12)),
        "dead_fraction": float(np.mean(fired_rows == 0)),
        "mean_l0": float(active_total / max(count, 1)),
        "mean_firing_rate": float(np.mean(firing_rate)),
    }
    return metrics, correlations, firing_rate.astype(np.float32)


def main() -> None:
    args = parse_args()
    config = typed_config(manifest_row(args.manifest, args.task_index))
    if canonical_config_hash(config) != config["config_hash"]:
        raise ValueError("manifest config hash does not match its protocol fields")
    metrics_path = Path(config["metrics"])
    if metrics_path.exists():
        existing = json.loads(metrics_path.read_text())
        if existing.get("status") == "complete" and existing.get("config_hash") == config["config_hash"]:
            print(json.dumps(existing, indent=2))
            return

    acts = np.load(Path(config["activation_path"]), mmap_mode="r")
    records = read_csv(Path(config["records_path"]))
    if len(records) != len(acts):
        raise ValueError(f"record/activation row mismatch: {len(records)} != {len(acts)}")
    splits = np.asarray([row["split"] for row in records])
    train_idx = np.flatnonzero(splits == "train")
    val_idx = np.flatnonzero(splits == "val")
    test_idx = np.flatnonzero(splits == "test")
    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError("train/validation/test splits must all be non-empty")

    concepts, concept_names, concept_mean, concept_scale = standardized_concepts(
        [row["ecg_id"] for row in records],
        read_csv(args.concepts),
        splits == "train",
        preserve_missing=args.preserve_missing_concepts,
    )
    family_by_concept = {
        row["concept_id"]: row["family"]
        for row in read_csv(args.concept_registry)
        if row.get("main") == "yes"
    }
    if concept_names != [name for name in concept_names if name in family_by_concept]:
        missing = [name for name in concept_names if name not in family_by_concept]
        raise ValueError(f"concept registry mismatch: {missing}")

    model = train(config, acts, train_idx, args.device, args.checkpoint_every)
    semantic_train_idx = train_idx
    if args.complete_case_evaluation:
        complete = np.all(np.isfinite(concepts), axis=1)
        semantic_train_idx = semantic_train_idx[complete[semantic_train_idx]]
        val_idx = val_idx[complete[val_idx]]
        test_idx = test_idx[complete[test_idx]]
        if min(len(semantic_train_idx), len(val_idx), len(test_idx)) == 0:
            raise RuntimeError("complete-case semantic split is empty")
    if args.semantic_train_limit > 0 and len(train_idx) > args.semantic_train_limit:
        semantic_rng = np.random.default_rng(20260714)
        semantic_train_idx = np.sort(
            semantic_rng.choice(train_idx, size=args.semantic_train_limit, replace=False)
        )

    train_metrics, train_corr, _ = evaluate(
        model, acts, semantic_train_idx, concepts, int(config["batch_size"]), args.device
    )
    val_metrics, val_corr, val_firing_rate = evaluate(
        model, acts, val_idx, concepts, int(config["batch_size"]), args.device
    )
    test_metrics, test_corr, _ = evaluate(
        model, acts, test_idx, concepts, int(config["batch_size"]), args.device
    )
    val_rows, val_semantic = selected_concept_metrics(train_corr, val_corr, concept_names)
    test_rows, test_semantic = selected_concept_metrics(train_corr, test_corr, concept_names)
    concept_rows = []
    for split_name, rows in (("validation", val_rows), ("test", test_rows)):
        for row in rows:
            concept_index = concept_names.index(str(row["concept"]))
            concept_rows.append(
                {
                    "model": config["model"],
                    "layer": config["layer"],
                    "relative_depth": config["relative_depth"],
                    "actual_relative_depth": config["actual_relative_depth"],
                    "expansion_E": config["expansion_E"],
                    "sparsity_arm": config["sparsity_arm"],
                    "seed": config["seed"],
                    "split": split_name,
                    "family": family_by_concept[str(row["concept"])],
                    "train_mean": float(concept_mean[concept_index]),
                    "train_scale": float(concept_scale[concept_index]),
                    **row,
                }
            )
    atomic_csv(Path(config["concept_metrics"]), concept_rows)
    firing_path = Path(config["firing_rate"])
    firing_path.parent.mkdir(parents=True, exist_ok=True)
    firing_tmp = firing_path.with_suffix(firing_path.suffix + f".tmp.{os.getpid()}")
    with firing_tmp.open("wb") as handle:
        np.save(handle, val_firing_rate)
    os.replace(firing_tmp, firing_path)

    payload: dict[str, Any] = {
        "status": "complete",
        "config_hash": config["config_hash"],
        "task_index": config["task_index"],
        "model": config["model"],
        "feature_suffix": config["feature_suffix"],
        "layer": config["layer"],
        "relative_depth": config["relative_depth"],
        "actual_relative_depth": config["actual_relative_depth"],
        "n_layers": config["n_layers"],
        "d_hidden": config["d_hidden"],
        "sparsity_arm": config["sparsity_arm"],
        "expansion_E": config["expansion_E"],
        "N": config["N"],
        "k": config["k"],
        "k_over_d": config["k_over_d"],
        "k_over_N": config["k_over_N"],
        "seed": config["seed"],
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "split_counts": {
            "train": int(len(train_idx)),
            "semantic_train": int(len(semantic_train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
            "semantic_complete_case": bool(args.complete_case_evaluation),
        },
        "train_semantic_sample": train_metrics,
        "validation": {**val_metrics, **val_semantic},
        "test": {**test_metrics, **test_semantic},
        "checkpoint": config["checkpoint"],
        "concept_metrics": config["concept_metrics"],
        "firing_rate": config["firing_rate"],
    }
    atomic_json(metrics_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
