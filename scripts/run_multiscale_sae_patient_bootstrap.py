#!/usr/bin/env python
"""Patient-cluster bootstrap for one frozen multi-scale SAE cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PATIENT_BOOTSTRAP_PROTOCOL = "patient_cluster_v2"
BOOTSTRAP_CHECKPOINT_VERSION = 1
BOOTSTRAP_DISTRIBUTION_KEYS = (
    "recon_R2",
    "semantic_alignment",
    "concept_coverage_020",
    "concept_correlation",
)

from benchmark_v1.multiscale_sae import read_csv, standardized_concepts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/training_manifest.csv",
    )
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--bootstrap-chunk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--concepts", type=Path, default=ROOT / "results/manifest/concepts_matrix.csv"
    )
    parser.add_argument("--complete-case-concepts", action="store_true")
    parser.add_argument(
        "--patient-manifest",
        type=Path,
        default=ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v1/manifest.csv",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/multiscale_sae_v1/patient_bootstrap",
    )
    return parser.parse_args()


def manifest_row(path: Path, task_index: int) -> dict[str, str]:
    matches = [row for row in read_csv(path) if int(row["task_index"]) == task_index]
    if len(matches) != 1:
        raise RuntimeError(f"expected one manifest row for task {task_index}, found {len(matches)}")
    return matches[0]


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def selected_features(
    path: Path, split: str, concept_names: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(path) if row["split"] == split]
    by_concept = {row["concept"]: row for row in rows}
    if set(by_concept) != set(concept_names):
        raise RuntimeError(
            f"concept support mismatch in {path}: expected={len(concept_names)}, "
            f"observed={len(by_concept)}"
        )
    indices = np.asarray(
        [int(float(by_concept[name]["selected_feature"])) for name in concept_names],
        dtype=np.int64,
    )
    expected_correlations = np.asarray(
        [float(by_concept[name]["eval_correlation"]) for name in concept_names],
        dtype=np.float64,
    )
    return indices, expected_correlations


def patient_groups(
    records: list[dict[str, str]], eval_indices: np.ndarray, patient_manifest: Path
) -> tuple[np.ndarray, np.ndarray, str, str]:
    patient_by_ecg = {
        str(row["ecg_id"]): str(row["patient_id"])
        for row in read_csv(patient_manifest)
    }
    patient_ids = []
    for index in eval_indices:
        ecg_id = str(records[int(index)]["ecg_id"])
        if ecg_id not in patient_by_ecg:
            raise KeyError(f"patient id missing for ecg_id={ecg_id}")
        patient_ids.append(patient_by_ecg[ecg_id])
    unique, inverse = np.unique(np.asarray(patient_ids, dtype=str), return_inverse=True)
    inverse = inverse.astype(np.int64)
    id_digest = hashlib.sha256("\n".join(unique.tolist()).encode()).hexdigest()
    record_counts = np.bincount(inverse, minlength=len(unique))
    cluster_digest = hashlib.sha256(
        "\n".join(
            f"{patient_id}\t{int(record_count)}"
            for patient_id, record_count in zip(unique, record_counts)
        ).encode()
    ).hexdigest()
    return unique, inverse, id_digest, cluster_digest


def bootstrap_design_hash(
    patient_cluster_hash: str, n_patients: int, samples: int, seed: int
) -> str:
    payload = {
        "algorithm": "numpy.default_rng.multinomial",
        "n_patients": int(n_patients),
        "patient_cluster_hash": patient_cluster_hash,
        "probabilities": "uniform",
        "samples": int(samples),
        "seed": int(seed),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def aggregate_by_patient(
    inverse: np.ndarray,
    sse: np.ndarray,
    sst: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
) -> dict[str, np.ndarray]:
    n_patients = int(inverse.max()) + 1
    n_concepts = z.shape[1]
    stats = {
        "count": np.zeros(n_patients, dtype=np.float64),
        "sse": np.zeros(n_patients, dtype=np.float64),
        "sst": np.zeros(n_patients, dtype=np.float64),
        "sum_z": np.zeros((n_patients, n_concepts), dtype=np.float64),
        "sum_z2": np.zeros((n_patients, n_concepts), dtype=np.float64),
        "sum_y": np.zeros((n_patients, n_concepts), dtype=np.float64),
        "sum_y2": np.zeros((n_patients, n_concepts), dtype=np.float64),
        "cross": np.zeros((n_patients, n_concepts), dtype=np.float64),
    }
    np.add.at(stats["count"], inverse, 1.0)
    np.add.at(stats["sse"], inverse, sse)
    np.add.at(stats["sst"], inverse, sst)
    np.add.at(stats["sum_z"], inverse, z)
    np.add.at(stats["sum_z2"], inverse, np.square(z))
    np.add.at(stats["sum_y"], inverse, y)
    np.add.at(stats["sum_y2"], inverse, np.square(y))
    np.add.at(stats["cross"], inverse, z * y)
    return stats


def summarize_weighted(stats_t: dict[str, Any], weights):
    import torch

    count = weights @ stats_t["count"]
    sse = weights @ stats_t["sse"]
    sst = weights @ stats_t["sst"]
    sum_z = weights @ stats_t["sum_z"]
    sum_z2 = weights @ stats_t["sum_z2"]
    sum_y = weights @ stats_t["sum_y"]
    sum_y2 = weights @ stats_t["sum_y2"]
    cross = weights @ stats_t["cross"]
    covariance = cross - sum_z * sum_y / count[:, None]
    var_z = torch.clamp(sum_z2 - sum_z.square() / count[:, None], min=0.0)
    var_y = torch.clamp(sum_y2 - sum_y.square() / count[:, None], min=0.0)
    denominator = torch.sqrt(var_z * var_y)
    correlation = torch.where(
        denominator > 1e-12,
        covariance / torch.clamp(denominator, min=1e-12),
        torch.zeros_like(covariance),
    ).clamp(-1.0, 1.0)
    absolute = correlation.abs()
    return {
        "recon_R2": 1.0 - sse / torch.clamp(sst, min=1e-12),
        "semantic_alignment": absolute.mean(dim=1),
        "concept_coverage_020": (absolute >= 0.20).to(torch.float64).mean(dim=1),
        "concept_correlation": correlation,
    }


def save_bootstrap_checkpoint(
    path: Path,
    metadata: dict[str, Any],
    completed_samples: int,
    rng: np.random.Generator,
    distributions: dict[str, list[np.ndarray]],
) -> None:
    arrays = {
        key: np.concatenate(parts, axis=0)
        for key, parts in distributions.items()
    }
    atomic_npz(
        path,
        checkpoint_version=np.asarray(BOOTSTRAP_CHECKPOINT_VERSION, dtype=np.int64),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        completed_samples=np.asarray(completed_samples, dtype=np.int64),
        rng_state_json=np.asarray(json.dumps(rng.bit_generator.state, sort_keys=True)),
        **arrays,
    )


def load_bootstrap_checkpoint(
    path: Path, expected_metadata: dict[str, Any]
) -> tuple[int, dict[str, Any], dict[str, list[np.ndarray]]]:
    with np.load(path, allow_pickle=False) as saved:
        version = int(saved["checkpoint_version"])
        metadata = json.loads(str(saved["metadata_json"]))
        completed = int(saved["completed_samples"])
        rng_state = json.loads(str(saved["rng_state_json"]))
        if version != BOOTSTRAP_CHECKPOINT_VERSION:
            raise RuntimeError(f"unsupported bootstrap checkpoint version: {version}")
        if metadata != expected_metadata:
            raise RuntimeError(
                f"bootstrap checkpoint identity mismatch: {path}"
            )
        if not 0 < completed <= int(expected_metadata["bootstrap_samples"]):
            raise RuntimeError(
                f"invalid completed sample count in {path}: {completed}"
            )
        distributions = {}
        for key in BOOTSTRAP_DISTRIBUTION_KEYS:
            values = np.asarray(saved[key])
            if len(values) != completed:
                raise RuntimeError(
                    f"bootstrap checkpoint length mismatch for {key}: "
                    f"{len(values)} != {completed}"
                )
            distributions[key] = [values]
    return completed, rng_state, distributions


def bootstrap(
    stats: dict[str, np.ndarray],
    samples: int,
    seed: int,
    chunk_size: int,
    device: str,
    checkpoint_path: Path | None = None,
    checkpoint_identity: dict[str, Any] | None = None,
    after_checkpoint: Callable[[int], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    import torch

    if samples <= 0 or chunk_size <= 0:
        raise ValueError("bootstrap samples and chunk size must be positive")
    stats_t = {
        key: torch.as_tensor(value, dtype=torch.float64, device=device)
        for key, value in stats.items()
    }
    n_patients = len(stats["count"])
    observed_weights = torch.ones((1, n_patients), dtype=torch.float64, device=device)
    observed_t = summarize_weighted(stats_t, observed_weights)
    observed = {
        key: value.detach().cpu().numpy()[0]
        for key, value in observed_t.items()
    }

    checkpoint_metadata = {
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
        "n_patients": int(n_patients),
        "identity": checkpoint_identity,
    }
    if checkpoint_path is not None and checkpoint_identity is None:
        raise ValueError("checkpoint identity is required when checkpointing")

    rng = np.random.default_rng(seed)
    completed_samples = 0
    distributions: dict[str, list[np.ndarray]] = {
        key: [] for key in BOOTSTRAP_DISTRIBUTION_KEYS
    }
    if checkpoint_path is not None and checkpoint_path.exists():
        completed_samples, rng_state, distributions = load_bootstrap_checkpoint(
            checkpoint_path, checkpoint_metadata
        )
        rng.bit_generator.state = rng_state

    probabilities = np.full(n_patients, 1.0 / n_patients, dtype=np.float64)
    for start in range(completed_samples, samples, chunk_size):
        size = min(chunk_size, samples - start)
        weights_np = rng.multinomial(n_patients, probabilities, size=size).astype(np.float64)
        weights = torch.from_numpy(weights_np).to(device=device)
        values = summarize_weighted(stats_t, weights)
        for key, value in values.items():
            distributions[key].append(value.detach().cpu().numpy().astype(np.float32))
        completed_samples = start + size
        if checkpoint_path is not None:
            save_bootstrap_checkpoint(
                checkpoint_path,
                checkpoint_metadata,
                completed_samples,
                rng,
                distributions,
            )
            if after_checkpoint is not None:
                after_checkpoint(completed_samples)
    return observed, {key: np.concatenate(parts, axis=0) for key, parts in distributions.items()}


def main() -> None:
    args = parse_args()
    row = manifest_row(args.manifest, args.task_index)
    output_dir = args.output_root / f"task_{args.task_index:06d}"
    summary_path = output_dir / f"{args.split}_patient_bootstrap.json"
    distribution_path = output_dir / f"{args.split}_patient_bootstrap.npz"
    progress_path = output_dir / f"{args.split}_patient_bootstrap.progress.npz"
    identity = {
        "task_index": args.task_index,
        "config_hash": row["config_hash"],
        "split": args.split,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "patient_bootstrap_protocol": PATIENT_BOOTSTRAP_PROTOCOL,
    }
    if summary_path.exists() and distribution_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("status") == "complete" and all(
            existing.get(key) == value for key, value in identity.items()
        ):
            print(json.dumps(existing, indent=2))
            return

    checkpoint_path = Path(row["checkpoint"])
    metrics_path = Path(row["metrics"])
    concept_path = Path(row["concept_metrics"])
    for path in (checkpoint_path, metrics_path, concept_path):
        if not path.exists():
            raise FileNotFoundError(path)

    acts = np.load(Path(row["activation_path"]), mmap_mode="r")
    records = read_csv(Path(row["records_path"]))
    if len(records) != len(acts):
        raise ValueError(f"record/activation row mismatch: {len(records)} != {len(acts)}")
    splits = np.asarray([record["split"] for record in records])
    split_value = "val" if args.split == "validation" else "test"
    eval_indices = np.flatnonzero(splits == split_value)
    if len(eval_indices) == 0:
        raise RuntimeError(f"empty evaluation split: {args.split}")
    concepts, concept_names, _, _ = standardized_concepts(
        [record["ecg_id"] for record in records],
        read_csv(args.concepts),
        splits == "train",
        preserve_missing=args.complete_case_concepts,
    )
    if args.complete_case_concepts:
        complete = np.all(np.isfinite(concepts), axis=1)
        eval_indices = eval_indices[complete[eval_indices]]
    feature_indices, expected_correlations = selected_features(
        concept_path, args.split, concept_names
    )
    unique_patients, inverse, patient_hash, patient_cluster_hash = patient_groups(
        records, eval_indices, args.patient_manifest
    )
    design_hash = bootstrap_design_hash(
        patient_cluster_hash,
        len(unique_patients),
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    import torch

    from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE

    saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    config = saved.get("config", {})
    if not saved.get("final", False):
        raise RuntimeError(f"checkpoint is not final: {checkpoint_path}")
    if config.get("config_hash") != row["config_hash"]:
        raise RuntimeError(f"checkpoint/manifest hash mismatch: {checkpoint_path}")
    model = BatchTopKSAE(
        int(config["d_hidden"]), int(config["N"]), int(config["k"])
    ).to(args.device)
    model.load_state_dict(saved["model"])
    model.eval()
    selected_tensor = torch.as_tensor(feature_indices, dtype=torch.long, device=args.device)
    eval_mean = np.mean(np.asarray(acts[eval_indices], dtype=np.float32), axis=0)
    mean_normalized = model.normalise(
        torch.from_numpy(eval_mean).to(device=args.device)
    ).detach()
    all_sse = []
    all_sst = []
    all_z = []
    all_y = []
    with torch.no_grad():
        for start in range(0, len(eval_indices), args.batch_size):
            batch_indices = eval_indices[start : start + args.batch_size]
            raw = torch.as_tensor(
                np.asarray(acts[batch_indices]), dtype=torch.float32, device=args.device
            )
            reconstruction, z, normalized = model(raw)
            all_sse.append(
                (normalized - reconstruction).square().sum(dim=1).cpu().numpy().astype(np.float64)
            )
            all_sst.append(
                (normalized - mean_normalized).square().sum(dim=1).cpu().numpy().astype(np.float64)
            )
            all_z.append(z.index_select(1, selected_tensor).cpu().numpy().astype(np.float64))
            all_y.append(np.asarray(concepts[batch_indices], dtype=np.float64))
    stats = aggregate_by_patient(
        inverse,
        np.concatenate(all_sse),
        np.concatenate(all_sst),
        np.concatenate(all_z),
        np.concatenate(all_y),
    )
    observed, distributions = bootstrap(
        stats,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.bootstrap_chunk,
        args.device,
        checkpoint_path=progress_path,
        checkpoint_identity={
            **identity,
            "patient_cluster_hash": patient_cluster_hash,
            "bootstrap_design_hash": design_hash,
            "n_concepts": len(concept_names),
        },
    )

    metrics = json.loads(metrics_path.read_text())
    expected_metrics = metrics[args.split]
    observed_correlations = np.asarray(observed["concept_correlation"], dtype=np.float64)
    verification = {
        "max_abs_concept_correlation_error": float(
            np.max(np.abs(observed_correlations - expected_correlations))
        ),
        "recon_R2_error": float(
            abs(float(observed["recon_R2"]) - float(expected_metrics["recon_R2"]))
        ),
        "semantic_alignment_error": float(
            abs(
                float(observed["semantic_alignment"])
                - float(expected_metrics["mean_train_selected_abs_correlation"])
            )
        ),
    }
    if verification["max_abs_concept_correlation_error"] > 2e-4:
        raise RuntimeError(f"concept correlation reproduction failed: {verification}")
    if verification["recon_R2_error"] > 2e-5:
        raise RuntimeError(f"reconstruction reproduction failed: {verification}")
    if verification["semantic_alignment_error"] > 2e-4:
        raise RuntimeError(f"semantic reproduction failed: {verification}")

    atomic_npz(
        distribution_path,
        recon_R2=distributions["recon_R2"],
        semantic_alignment=distributions["semantic_alignment"],
        concept_coverage_020=distributions["concept_coverage_020"],
        concept_correlation=distributions["concept_correlation"],
        concept_names=np.asarray(concept_names),
        observed_recon_R2=np.asarray(observed["recon_R2"], dtype=np.float64),
        observed_semantic_alignment=np.asarray(
            observed["semantic_alignment"], dtype=np.float64
        ),
        observed_concept_coverage_020=np.asarray(
            observed["concept_coverage_020"], dtype=np.float64
        ),
        observed_concept_correlation=observed_correlations,
    )
    payload = {
        "status": "complete",
        **identity,
        "model": row["model"],
        "relative_depth": float(row["relative_depth"]),
        "actual_relative_depth": float(row["actual_relative_depth"]),
        "layer": int(row["layer"]),
        "expansion_E": int(row["expansion_E"]),
        "seed": int(row["seed"]),
        "n_records": int(len(eval_indices)),
        "n_patients": int(len(unique_patients)),
        "patient_id_hash": patient_hash,
        "patient_cluster_hash": patient_cluster_hash,
        "bootstrap_design_hash": design_hash,
        "n_concepts": len(concept_names),
        "observed_recon_R2": float(observed["recon_R2"]),
        "observed_semantic_alignment": float(observed["semantic_alignment"]),
        "observed_concept_coverage_020": float(observed["concept_coverage_020"]),
        "verification": verification,
        "distribution": str(distribution_path),
        "progress_checkpoint": str(progress_path),
        "bootstrap_checkpoint_version": BOOTSTRAP_CHECKPOINT_VERSION,
        "resampling_unit": "PTB-XL patient with all of that patient's records retained",
        "batchtopk_resampling": "codes are computed once in frozen original record order and batch size; patient bootstrap reweights per-record sufficient statistics without re-encoding",
        "comparison_rule": "all cells use identical bootstrap patient draws; FM comparisons remain matched on relative expansion E",
        "claim_boundary": "conditional on frozen FM/SAE weights, train-selected features, fixed test normalization reference, and the three preregistered SAE seeds",
    }
    atomic_json(summary_path, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
