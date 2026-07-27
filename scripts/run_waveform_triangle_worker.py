#!/usr/bin/env python
"""Waveform-to-latent-to-readout triangle worker with matched controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_v2_common import V1, V2, write_json  # noqa: E402
from scripts.run_waveform_intervention_worker import (  # noqa: E402
    MODEL_SUFFIX,
    STRENGTHS,
    TARGETS,
    encode_all,
    infer_pooled,
    landmarks_and_measurements,
    load_encoder,
    load_sae,
    preprocess,
    task_cell,
    transform_waveform,
    variant_name,
)
from scripts.extract_external_model_activations import load_wfdb_12lead  # noqa: E402


EXTERNAL = ROOT / "results" / "external_benchmark_v1"
OUT = V2 / "waveform_triangle" / "workers"
MEASUREMENT = {
    "rr_irregularity": "rr_cv",
    "qrs_duration": "qrs_duration_ms",
    "qt_interval": "qt_interval_ms",
}
SEEDS = (4311, 4312, 4313)
RANDOM_GROUPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--model-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def reference_path(model: str, cohort: str, phenotype: str) -> Path:
    return V1 / "waveform_interventions" / "workers" / model / cohort / phenotype / "per_variant_metrics.csv"


def protocol_for(model: str) -> str:
    return "frozen_atom" if model == "ecg_jepa" else "cohort_adapted_atom"


def result_path(model: str, cohort: str, phenotype: str, seed: int) -> Path:
    suffix = MODEL_SUFFIX[model]
    return (
        EXTERNAL
        / suffix
        / cohort
        / "steering"
        / protocol_for(model)
        / f"seed{seed}"
        / TARGETS[phenotype]
        / "result.json"
    )


def preflight(model: str, cohort: str, phenotype: str) -> dict:
    reference = pd.read_csv(reference_path(model, cohort, phenotype))
    identity = reference[reference.variant.eq("identity")]
    if len(identity) != 256 or reference.record_id.nunique() != 256 or len(reference) != 1280:
        raise RuntimeError(
            f"Unexpected v1 reference shape for {model}/{cohort}/{phenotype}: {reference.shape}"
        )
    results = []
    for seed in SEEDS:
        path = result_path(model, cohort, phenotype, seed)
        payload = json.loads(path.read_text())
        if len(payload["selected_atoms"]["top5"]) != 5:
            raise RuntimeError(f"Invalid selected top5: {path}")
        if len(payload["random_groups"]) < RANDOM_GROUPS:
            raise RuntimeError(f"Missing matched random groups: {path}")
        checkpoint = Path(payload["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        results.append(str(path))
    return {
        "model": model,
        "cohort": cohort,
        "phenotype": phenotype,
        "reference_records": int(reference.record_id.nunique()),
        "reference_rows": len(reference),
        "sae_results": len(results),
        "random_groups_per_seed": RANDOM_GROUPS,
    }


def rebuild_inputs(
    model: str, phenotype: str, reference: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    identities = reference[reference.variant.eq("identity")].copy()
    model_inputs = []
    rows = []
    for item in identities.itertuples():
        wave, fs = load_wfdb_12lead(Path(item.record_path), physical=True)
        landmarks, original_measure = landmarks_and_measurements(wave, fs)
        variants = [("identity", 0, 0.0, wave, original_measure)]
        for direction in (-1, 1):
            for strength in STRENGTHS:
                edited = transform_waveform(wave, landmarks, phenotype, strength, direction)
                rpeaks_hint = None if phenotype == "rr_irregularity" else landmarks["rpeaks"]
                _, edited_measure = landmarks_and_measurements(
                    edited, fs, rpeaks_hint=rpeaks_hint
                )
                variants.append(
                    (
                        variant_name(phenotype, direction),
                        direction,
                        strength,
                        edited,
                        edited_measure,
                    )
                )
        for variant, direction, strength, variant_wave, measurements in variants:
            model_inputs.append(preprocess(model, variant_wave))
            rows.append(
                {
                    "record_id": str(item.record_id),
                    "variant": variant,
                    "direction_sign": int(direction),
                    "strength": float(strength),
                    "measurement_value": float(measurements[MEASUREMENT[phenotype]]),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != len(reference):
        raise RuntimeError(f"Rebuilt {len(frame)} rows but expected {len(reference)}")
    return np.stack(model_inputs), frame


def main() -> None:
    args = parse_args()
    model, cohort, phenotype = task_cell(args.task_index)
    audit = preflight(model, cohort, phenotype)
    if args.preflight_only:
        print(audit)
        return

    worker_out = args.out / model / cohort / phenotype
    complete = worker_out / "complete.json"
    if complete.exists() and not args.force:
        print(f"already complete: {complete}")
        return
    worker_out.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(reference_path(model, cohort, phenotype))
    inputs, rows = rebuild_inputs(model, phenotype, reference)
    encoder, model_status = load_encoder(model, args.device)
    pooled = infer_pooled(model, encoder, inputs, args.device, args.model_batch)
    suffix = MODEL_SUFFIX[model]
    target = TARGETS[phenotype]
    bundle = joblib.load(EXTERNAL / suffix / cohort / "frozen_heads.joblib")
    scaler = bundle["scaler"]
    clf = bundle["heads"][target]["clf"]
    standardized = scaler.transform(pooled)
    raw_head = np.asarray(clf.decision_function(standardized), dtype=float)

    reference_key = ["record_id", "variant", "direction_sign", "strength"]
    reference_head = reference[reference_key + [f"head_{target}"]].copy()
    reference_head["record_id"] = reference_head.record_id.astype(str)
    comparison = rows.merge(reference_head, on=reference_key, validate="one_to_one")
    raw_difference = np.abs(raw_head - comparison[f"head_{target}"].to_numpy(dtype=float))

    output_frames = []
    for seed in SEEDS:
        payload = json.loads(result_path(model, cohort, phenotype, seed).read_text())
        sae = load_sae(Path(payload["checkpoint"]), args.device)
        codes = encode_all(sae, pooled, args.device)
        selected = np.asarray(payload["selected_atoms"]["top5"], dtype=int)
        random_groups = np.asarray(payload["random_groups"][:RANDOM_GROUPS], dtype=int)
        if random_groups.shape != (RANDOM_GROUPS, 5):
            raise RuntimeError(f"Unexpected random group shape: {random_groups.shape}")
        decoder_raw = (
            sae.W_dec.detach().cpu().numpy()
            * sae.sigma.detach().cpu().numpy()[:, None]
        )
        raw_coefficient = np.asarray(clf.coef_).reshape(-1) / scaler.scale_
        latent_gradient = raw_coefficient @ decoder_raw
        bias_raw = (
            sae.b_dec.detach().cpu().numpy() * sae.sigma.detach().cpu().numpy()
            + sae.mu.detach().cpu().numpy()
        )
        head_constant = float(
            np.asarray(clf.intercept_).reshape(-1)[0]
            + (bias_raw - scaler.mean_) @ raw_coefficient
        )
        reconstructed_head = head_constant + codes @ latent_gradient
        selected_contribution = codes[:, selected] @ latent_gradient[selected]
        random_contributions = np.column_stack(
            [codes[:, group] @ latent_gradient[group] for group in random_groups]
        )
        seed_frame = rows.copy()
        seed_frame.insert(0, "model", "ECG-JEPA" if model == "ecg_jepa" else "ECG-FM")
        seed_frame.insert(1, "model_suffix", suffix)
        seed_frame.insert(2, "cohort", cohort)
        seed_frame.insert(3, "phenotype", phenotype)
        seed_frame.insert(4, "target", target)
        seed_frame["seed"] = seed
        seed_frame["raw_target_head"] = raw_head
        seed_frame["sae_reconstructed_target_head"] = reconstructed_head
        seed_frame["selected_top5_contribution"] = selected_contribution
        seed_frame["selected_top5_ablated_target_head"] = (
            reconstructed_head - selected_contribution
        )
        seed_frame["selected_atoms"] = "|".join(map(str, selected.tolist()))
        for index in range(RANDOM_GROUPS):
            seed_frame[f"random_contribution_{index:02d}"] = random_contributions[:, index]
        output_frames.append(seed_frame)

    output = pd.concat(output_frames, ignore_index=True)
    output.to_csv(worker_out / "triangle_per_variant.csv", index=False)
    metadata = {
        "schema_version": 1,
        "task_index": args.task_index,
        **audit,
        "target": target,
        "protocol": protocol_for(model),
        "seeds": list(SEEDS),
        "output_rows": len(output),
        "expected_output_rows": 256 * 5 * len(SEEDS),
        "raw_head_reference_max_abs_difference": float(raw_difference.max()),
        "raw_head_reference_mean_abs_difference": float(raw_difference.mean()),
        "model_status": model_status,
        "waveforms_written": False,
        "status": "complete",
    }
    write_json(complete, metadata)
    print(metadata)


if __name__ == "__main__":
    main()
