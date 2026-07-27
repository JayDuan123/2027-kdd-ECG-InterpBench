#!/usr/bin/env python
"""Evaluate fitted comparison directions on the frozen 12-cell waveform panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE  # noqa: E402
from scripts.method_comparison_common import (  # noqa: E402
    BASE,
    COMMON_K,
    METHODS,
    SEEDS,
    stable_seed,
    write_json,
)
from scripts.method_comparison_models import encode_decode_sae, semi_nmf_transform  # noqa: E402
from scripts.run_waveform_triangle_worker import (  # noqa: E402
    MEASUREMENT,
    preflight,
    rebuild_inputs,
)
from scripts.run_waveform_intervention_worker import (  # noqa: E402
    MODEL_SUFFIX,
    TARGETS,
    infer_pooled,
    load_encoder,
    task_cell,
)


OUT = BASE / "waveform_triangle" / "workers"
RANDOM_GROUPS = 20
WAVEFORM_METHODS = (*METHODS, "sae_existing_8d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=BASE / "manifest.csv")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--model-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def worker_path(base: Path, row: pd.Series) -> Path:
    return (
        base
        / "workers"
        / f"task_{int(row.task_index):03d}_{row.model_suffix}_{row.cohort}_seed{int(row.seed)}"
    )


def selected_components(worker: Path, method: str, target: str) -> np.ndarray:
    frame = pd.read_csv(worker / "selected_directions.csv")
    regime = "existing_sae_energy" if method == "sae_existing_8d" else "common64_energy"
    match = frame[
        frame.regime.eq(regime) & frame.method.eq(method) & frame.target.eq(target)
    ]
    if len(match) != 1:
        raise RuntimeError(f"Expected one selected-direction row for {worker}/{method}/{target}")
    raw = str(match.iloc[0].selected_components)
    return np.asarray([int(value) for value in raw.split("|")], dtype=int)


def random_groups(width: int, selected: np.ndarray, seed: int) -> list[np.ndarray]:
    available = np.asarray([index for index in range(width) if index not in set(selected)])
    rng = np.random.default_rng(seed)
    return [
        np.sort(rng.choice(available, size=len(selected), replace=False)).astype(int)
        for _ in range(RANDOM_GROUPS)
    ]


def contribution(
    codes: np.ndarray,
    decoder: np.ndarray,
    coefficient: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(decoder[selected], dtype=np.float64) @ np.asarray(
        coefficient, dtype=np.float64
    )
    return np.asarray(codes[:, selected] @ weights, dtype=np.float32)


def common_sae_payload(path: Path, values: np.ndarray, device: str, batch: int) -> dict:
    import torch

    saved = torch.load(path, map_location=device, weights_only=False)
    state = saved["model"]
    model = BatchTopKSAE(
        int(state["mu"].numel()), int(saved["rank"]), int(saved["k"])
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    codes, reconstruction = encode_decode_sae(model, values, device, batch)
    decoder = (
        model.W_dec.detach().cpu().numpy().T
        * model.sigma.detach().cpu().numpy()[None, :]
    ).astype(np.float32)
    return {"codes": codes, "decoder": decoder, "reconstruction": reconstruction}


def existing_sae_payload(
    checkpoint: Path,
    pooled: np.ndarray,
    scaler,
    device: str,
    batch: int,
) -> dict:
    import torch

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    config = saved["config"]
    state = saved["model"]
    model = BatchTopKSAE(
        int(state["mu"].numel()), int(config["n_features"]), int(config["k"])
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    codes, reconstruction_raw = encode_decode_sae(model, pooled, device, batch)
    reconstruction = scaler.transform(reconstruction_raw).astype(np.float32)
    decoder = (
        model.W_dec.detach().cpu().numpy().T
        * (model.sigma.detach().cpu().numpy() / scaler.scale_)[None, :]
    ).astype(np.float32)
    return {"codes": codes, "decoder": decoder, "reconstruction": reconstruction}


def method_payload(
    method: str,
    fit_root: Path,
    x: np.ndarray,
    pooled: np.ndarray,
    scaler,
    checkpoint: Path,
    device: str,
    batch: int,
) -> dict:
    if method == "sae_common64":
        return common_sae_payload(fit_root / "sae_common64.pt", x, device, batch)
    if method == "pca64":
        model = joblib.load(fit_root / "pca64.joblib")
        codes = model.transform(x).astype(np.float32)
        return {
            "codes": codes,
            "decoder": np.asarray(model.components_, dtype=np.float32),
            "reconstruction": model.inverse_transform(codes).astype(np.float32),
        }
    if method == "ica64":
        model = joblib.load(fit_root / "ica64.joblib")
        codes = model.transform(x).astype(np.float32)
        return {
            "codes": codes,
            "decoder": np.asarray(model.mixing_.T, dtype=np.float32),
            "reconstruction": model.inverse_transform(codes).astype(np.float32),
        }
    if method == "semi_nmf64":
        with np.load(fit_root / "semi_nmf64.npz", allow_pickle=False) as saved:
            factor = {key: saved[key] for key in saved.files}
        codes = semi_nmf_transform(x, factor, 50, device)
        decoder = np.asarray(factor["decoder"], dtype=np.float32)
        mean = np.asarray(factor["mean"], dtype=np.float32)
        return {
            "codes": codes,
            "decoder": decoder,
            "reconstruction": codes @ decoder + mean[None, :],
        }
    if method == "random_basis64":
        with np.load(fit_root / "random_basis64.npz", allow_pickle=False) as saved:
            basis = np.asarray(saved["basis"], dtype=np.float32)
        mean = np.zeros(x.shape[1], dtype=np.float32)
        codes = (x - mean) @ basis
        return {
            "codes": codes,
            "decoder": basis.T,
            "reconstruction": codes @ basis.T + mean[None, :],
        }
    if method == "sae_existing_8d":
        return existing_sae_payload(checkpoint, pooled, scaler, device, batch)
    raise KeyError(method)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    model, cohort, phenotype = task_cell(args.task_index)
    target = TARGETS[phenotype]
    model_suffix = MODEL_SUFFIX[model]
    reference_audit = preflight(model, cohort, phenotype)
    manifest = pd.read_csv(args.manifest)
    matches = manifest[
        manifest.model_suffix.eq(model_suffix) & manifest.cohort.eq(cohort)
    ].sort_values("seed")
    if set(matches.seed.astype(int)) != set(SEEDS) or len(matches) != len(SEEDS):
        raise RuntimeError(f"Incomplete fit manifest for {model_suffix}/{cohort}")
    missing = [
        str(worker_path(BASE, row))
        for _, row in matches.iterrows()
        if not (worker_path(BASE, row) / "complete.json").exists()
    ]
    audit = {
        **reference_audit,
        "task_index": args.task_index,
        "target": target,
        "fit_workers": len(matches),
        "missing_fit_workers": missing,
        "methods": list(WAVEFORM_METHODS),
    }
    if missing:
        raise RuntimeError(f"Missing fitted comparison workers: {missing}")
    if args.preflight_only:
        print(audit)
        return

    worker_out = args.out / model / cohort / phenotype
    complete = worker_out / "complete.json"
    if complete.exists() and not args.force:
        print(f"already complete: {complete}")
        return
    reference = pd.read_csv(
        ROOT
        / "results"
        / "benchmark_extension_v1"
        / "waveform_interventions"
        / "workers"
        / model
        / cohort
        / phenotype
        / "per_variant_metrics.csv"
    )
    inputs, rows = rebuild_inputs(model, phenotype, reference)
    encoder, model_status = load_encoder(model, args.device)
    pooled = infer_pooled(model, encoder, inputs, args.device, args.model_batch)
    bundle = joblib.load(
        ROOT / "results" / "external_benchmark_v1" / model_suffix / cohort / "frozen_heads.joblib"
    )
    scaler = bundle["scaler"]
    x = scaler.transform(pooled).astype(np.float32)
    head = bundle["heads"][target]["clf"]
    coefficient = np.asarray(head.coef_).reshape(-1).astype(np.float32)
    intercept = float(np.asarray(head.intercept_).reshape(-1)[0])
    raw_head = np.asarray(head.decision_function(x), dtype=np.float32)
    output_rows = []

    for _, manifest_row in matches.iterrows():
        seed = int(manifest_row.seed)
        fit_worker = worker_path(BASE, manifest_row)
        fit_root = fit_worker / "fits"
        for method in WAVEFORM_METHODS:
            reconstruction_applicable = method not in {"sparse_probe", "supervised_cav"}
            if reconstruction_applicable:
                payload = method_payload(
                    method,
                    fit_root,
                    x,
                    pooled,
                    scaler,
                    Path(manifest_row.existing_sae_checkpoint),
                    args.device,
                    args.model_batch,
                )
                selected = selected_components(fit_worker, method, target)
                selected_value = contribution(
                    payload["codes"], payload["decoder"], coefficient, selected
                )
                if method == "sae_existing_8d":
                    result_path = (
                        Path(manifest_row.head_path).parent
                        / "steering"
                        / "cohort_adapted_atom"
                        / f"seed{seed}"
                        / target
                        / "result.json"
                    )
                    result = json.loads(result_path.read_text())
                    groups = [
                        np.asarray(group, dtype=int)
                        for group in result["random_groups"][:RANDOM_GROUPS]
                    ]
                else:
                    groups = random_groups(
                        payload["codes"].shape[1],
                        selected,
                        stable_seed("waveform-component-controls", model_suffix, cohort, seed, target, method),
                    )
                random_values = np.column_stack(
                    [
                        contribution(payload["codes"], payload["decoder"], coefficient, group)
                        for group in groups
                    ]
                ).astype(np.float32)
                reconstructed_head = (
                    np.asarray(payload["reconstruction"] @ coefficient, dtype=np.float32) + intercept
                )
            else:
                with np.load(
                    fit_root / f"supervised_directions_{target}.npz", allow_pickle=False
                ) as saved:
                    direction = np.asarray(saved[method], dtype=np.float32)
                direction /= max(float(np.linalg.norm(direction)), 1e-12)
                selected_value = (x @ direction) * float(direction @ coefficient)
                rng = np.random.default_rng(
                    stable_seed("waveform-direction-controls", model_suffix, cohort, seed, target, method)
                )
                random_values = []
                for _ in range(RANDOM_GROUPS):
                    random_direction = rng.normal(size=x.shape[1]).astype(np.float32)
                    random_direction /= max(float(np.linalg.norm(random_direction)), 1e-12)
                    random_values.append(
                        (x @ random_direction) * float(random_direction @ coefficient)
                    )
                random_values = np.column_stack(random_values).astype(np.float32)
                reconstructed_head = raw_head.copy()
            ablated_head = reconstructed_head - selected_value
            for index, base_row in enumerate(rows.itertuples(index=False)):
                item = {
                    **base_row._asdict(),
                    "model": model,
                    "model_suffix": model_suffix,
                    "cohort": cohort,
                    "phenotype": phenotype,
                    "target": target,
                    "seed": seed,
                    "method": method,
                    "reconstruction_applicable": reconstruction_applicable,
                    "raw_target_head": float(raw_head[index]),
                    "reconstructed_target_head": float(reconstructed_head[index]),
                    "selected_contribution": float(selected_value[index]),
                    "selected_ablated_target_head": float(ablated_head[index]),
                }
                for random_index in range(RANDOM_GROUPS):
                    item[f"random_{random_index:02d}_contribution"] = float(
                        random_values[index, random_index]
                    )
                output_rows.append(item)
            print(
                f"waveform method task={args.task_index} seed={seed} method={method}",
                flush=True,
            )

    output = pd.DataFrame(output_rows)
    expected_rows = len(rows) * len(SEEDS) * len(WAVEFORM_METHODS)
    if len(output) != expected_rows:
        raise RuntimeError(f"Waveform output rows {len(output)} != {expected_rows}")
    worker_out.mkdir(parents=True, exist_ok=True)
    atomic_csv(output, worker_out / "method_triangle_per_variant.csv")
    metadata = {
        "schema_version": 1,
        **audit,
        "status": "complete",
        "variant_rows": len(rows),
        "output_rows": len(output),
        "expected_output_rows": expected_rows,
        "seeds": len(SEEDS),
        "random_groups": RANDOM_GROUPS,
        "model_status": model_status,
        "measurement_column": MEASUREMENT[phenotype],
        "waveforms_written": False,
        "record_level_activations_written": False,
        "data_files_modified": False,
    }
    write_json(complete, metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
