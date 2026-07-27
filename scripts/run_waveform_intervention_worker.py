#!/usr/bin/env python
"""Controlled RR/QRS/QT waveform interventions for ECG-JEPA and ECG-FM."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.signal import resample


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "external_benchmark_v1"
MANIFEST = ROOT / "results" / "multicohort" / "track_f_full" / "waveform_concepts_by_record.csv"
OUT = ROOT / "results" / "benchmark_extension_v1" / "waveform_interventions" / "workers"

from benchmark_v1.sae_reconciliation.batchtopk_sae import BatchTopKSAE  # noqa: E402
from scripts.extract_external_model_activations import load_wfdb_12lead  # noqa: E402

MODELS = ("ecg_jepa", "ecg_fm")
MODEL_SUFFIX = {"ecg_jepa": "ecg_jepa_cu118_commons", "ecg_fm": "ecg_fm_cu118_commons"}
COHORTS = ("chapman_f", "ningbo_f")
PHENOTYPES = ("rr_irregularity", "qrs_duration", "qt_interval")
TARGETS = {
    "rr_irregularity": "af_rhythm_native",
    "qrs_duration": "bbb_conduction_native",
    "qt_interval": "qt_interval_native",
}
STRENGTHS = (0.15, 0.30)
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--model-batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def task_cell(index: int) -> tuple[str, str, str]:
    cells = [(model, cohort, phenotype) for model in MODELS for cohort in COHORTS for phenotype in PHENOTYPES]
    if index < 0 or index >= len(cells):
        raise ValueError(f"task-index must be in 0..{len(cells)-1}")
    return cells[index]


def finite_array(values, length: int) -> np.ndarray:
    out = np.full(length, np.nan, dtype=float)
    if values is None:
        return out
    raw = np.asarray(values, dtype=float).reshape(-1)
    out[: min(length, len(raw))] = raw[:length]
    return out


def landmarks_and_measurements(
    wave: np.ndarray, fs: float, rpeaks_hint: np.ndarray | None = None
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    import neurokit2 as nk
    lead = np.nan_to_num(np.asarray(wave[1], dtype=float), nan=0.0)
    cleaned = nk.ecg_clean(lead, sampling_rate=fs)
    if rpeaks_hint is None:
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rpeaks = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=int)
    else:
        rpeaks = np.rint(np.asarray(rpeaks_hint)).astype(int)
        rpeaks = rpeaks[(rpeaks > 0) & (rpeaks < len(cleaned) - 1)]
    if len(rpeaks) < 4:
        raise ValueError("fewer than four R peaks")
    _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
    n = len(rpeaks)
    r_on = finite_array(waves.get("ECG_R_Onsets"), n)
    r_off = finite_array(waves.get("ECG_R_Offsets"), n)
    t_off = finite_array(waves.get("ECG_T_Offsets"), n)
    rr = np.diff(rpeaks) / fs * 1000.0
    qrs = (r_off - r_on) / fs * 1000.0
    qt = (t_off - r_on) / fs * 1000.0
    qrs = qrs[np.isfinite(qrs) & (qrs >= 40) & (qrs <= 220)]
    qt = qt[np.isfinite(qt) & (qt >= 180) & (qt <= 700)]
    if len(qrs) < 2 or len(qt) < 2:
        raise ValueError("insufficient valid QRS/QT delineations")
    measurements = {
        "rr_mean_ms": float(np.mean(rr)),
        "rr_cv": float(np.std(rr) / max(np.mean(rr), 1e-8)),
        "qrs_duration_ms": float(np.median(qrs)),
        "qt_interval_ms": float(np.median(qt)),
    }
    return {"rpeaks": rpeaks.astype(float), "r_on": r_on, "r_off": r_off, "t_off": t_off}, measurements


def warp_from_anchors(wave: np.ndarray, new_positions: list[float], old_positions: list[float]) -> np.ndarray:
    n = wave.shape[1]
    pairs = sorted(zip(new_positions + [0.0, float(n - 1)], old_positions + [0.0, float(n - 1)]))
    kept_new, kept_old = [], []
    for new, old in pairs:
        new = float(np.clip(new, 0, n - 1)); old = float(np.clip(old, 0, n - 1))
        if kept_new and (new <= kept_new[-1] + 0.5 or old <= kept_old[-1] + 0.5):
            continue
        kept_new.append(new); kept_old.append(old)
    if len(kept_new) < 4:
        raise ValueError("insufficient monotonic warp anchors")
    sample = np.arange(n, dtype=float)
    source_coordinate = np.interp(sample, kept_new, kept_old)
    return np.vstack([np.interp(source_coordinate, sample, lead) for lead in wave]).astype(np.float32)


def rr_warp(wave: np.ndarray, landmarks: dict[str, np.ndarray], strength: float, direction: int) -> np.ndarray:
    rpeaks = landmarks["rpeaks"]
    intervals = np.diff(rpeaks)
    if direction > 0:
        pattern = np.sin(np.arange(len(intervals)) * 2.399963229728653)
        pattern -= pattern.mean(); pattern /= max(float(np.std(pattern)), 1e-8)
        target = intervals * np.maximum(0.45, 1.0 + strength * pattern)
    else:
        target = (1.0 - strength) * intervals + strength * np.median(intervals)
    target *= intervals.sum() / max(target.sum(), 1e-8)
    new_r = np.concatenate([[rpeaks[0]], rpeaks[0] + np.cumsum(target)])
    return warp_from_anchors(wave, new_r.tolist(), rpeaks.tolist())


def qrs_warp(wave: np.ndarray, landmarks: dict[str, np.ndarray], strength: float, direction: int) -> np.ndarray:
    factor = 1.0 + direction * strength
    new_positions, old_positions = [], []
    for r, onset, offset in zip(landmarks["rpeaks"], landmarks["r_on"], landmarks["r_off"]):
        if not (np.isfinite(onset) and np.isfinite(offset) and onset < r < offset):
            continue
        new_positions.extend([r - (r - onset) * factor, r, r + (offset - r) * factor])
        old_positions.extend([onset, r, offset])
    return warp_from_anchors(wave, new_positions, old_positions)


def qt_warp(wave: np.ndarray, landmarks: dict[str, np.ndarray], strength: float, direction: int) -> np.ndarray:
    factor = 1.0 + direction * strength
    new_positions, old_positions = [], []
    for onset, t_end in zip(landmarks["r_on"], landmarks["t_off"]):
        if not (np.isfinite(onset) and np.isfinite(t_end) and onset < t_end):
            continue
        new_positions.extend([onset, onset + (t_end - onset) * factor])
        old_positions.extend([onset, t_end])
    return warp_from_anchors(wave, new_positions, old_positions)


def transform_waveform(
    wave: np.ndarray, landmarks: dict[str, np.ndarray], phenotype: str, strength: float, direction: int
) -> np.ndarray:
    if phenotype == "rr_irregularity":
        return rr_warp(wave, landmarks, strength, direction)
    if phenotype == "qrs_duration":
        return qrs_warp(wave, landmarks, strength, direction)
    if phenotype == "qt_interval":
        return qt_warp(wave, landmarks, strength, direction)
    raise ValueError(phenotype)


def variant_name(phenotype: str, direction: int) -> str:
    if phenotype == "rr_irregularity":
        return "increase_irregularity" if direction > 0 else "decrease_irregularity"
    if phenotype == "qrs_duration":
        return "widen" if direction > 0 else "narrow"
    return "lengthen" if direction > 0 else "shorten"


def preprocess(model: str, wave: np.ndarray) -> np.ndarray:
    if model == "ecg_jepa":
        indices = [0, 1, 6, 7, 8, 9, 10, 11]
        result = resample(wave[indices], 2500, axis=1).astype(np.float32)
    elif model == "ecg_fm":
        result = resample(wave, 5000, axis=1).astype(np.float32)
    else:
        raise ValueError(model)
    mean = result.mean(axis=1, keepdims=True); std = result.std(axis=1, keepdims=True)
    return np.nan_to_num((result - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def load_encoder(model: str, device: str):
    if model == "ecg_jepa":
        from benchmark_v1.adapters.ecg_jepa import try_load_encoder
        encoder, status = try_load_encoder(device)
    else:
        from benchmark_v1.adapters.ecg_fm import try_load_model
        encoder, status = try_load_model(device)
    if encoder is None:
        raise RuntimeError(status)
    return encoder, status


def infer_pooled(model_name: str, model, waves: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    import torch
    outputs = []
    with torch.no_grad():
        for lo in range(0, len(waves), batch_size):
            tensor = torch.as_tensor(waves[lo : lo + batch_size], dtype=torch.float32, device=device)
            if model_name == "ecg_jepa":
                pooled = model.representation(tensor)
            else:
                padding = torch.zeros(tensor.shape[0], tensor.shape[-1], dtype=torch.bool, device=device)
                tokens = model.extract_features(source=tensor, padding_mask=padding, mask=False)["x"]
                pooled = tokens.mean(dim=1)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs)


def stable_records(cohort: str, phenotype: str, bundle: dict, sample_size: int) -> list[dict[str, str]]:
    cohort_plain = cohort.replace("_f", "")
    record_ids = np.asarray(bundle["record_ids"]).astype(str)
    split = np.asarray(bundle["split"])
    allowed = set(record_ids[split == "test"])
    candidates = []
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("cohort") != cohort_plain or row.get("record_id") not in allowed:
                continue
            if row.get("status") != "ok" or row.get("interval_pass") != "true":
                continue
            score = int.from_bytes(
                hashlib.sha256(f"waveform-extension-v1:{cohort}:{phenotype}:{row['record_id']}".encode()).digest()[:8],
                "big",
            )
            candidates.append((score, row))
    ranked = [row for _, row in sorted(candidates, key=lambda item: item[0])]
    if len(ranked) < sample_size:
        raise RuntimeError(f"{cohort}/{phenotype}: only {len(ranked)} eligible test records")
    # Reserve deterministic replacements for records whose edited waveform
    # cannot be delineated.  No waveform is written to disk.
    selected = ranked[: min(len(ranked), sample_size * 3)]
    return selected


def load_sae(checkpoint: Path, device: str) -> BatchTopKSAE:
    import torch
    saved = torch.load(checkpoint, map_location=device)
    config = saved["config"]
    d = int(saved["model"]["mu"].numel())
    sae = BatchTopKSAE(d, int(config["n_features"]), int(config["k"])).to(device)
    sae.load_state_dict(saved["model"]); sae.eval()
    return sae


def encode_all(sae: BatchTopKSAE, pooled: np.ndarray, device: str) -> np.ndarray:
    import torch
    with torch.no_grad():
        raw = torch.as_tensor(pooled, dtype=torch.float32, device=device)
        return sae.encode(raw).cpu().numpy().astype(np.float32)


def main() -> None:
    args = parse_args()
    model_name, cohort, phenotype = task_cell(args.task_index)
    model_suffix = MODEL_SUFFIX[model_name]
    worker_out = args.out / model_name / cohort / phenotype
    complete = worker_out / "complete.json"
    if complete.exists() and not args.force:
        print(f"already complete: {complete}")
        return
    worker_out.mkdir(parents=True, exist_ok=True)
    pair_root = BASE / model_suffix / cohort
    bundle = joblib.load(pair_root / "frozen_heads.joblib")
    target = TARGETS[phenotype]
    names = list(bundle["targets"])
    if target not in names:
        raise RuntimeError(f"{target} is unavailable for {model_suffix}/{cohort}")
    records = stable_records(cohort, phenotype, bundle, args.sample_size)

    model_inputs, output_rows = [], []
    failures = []
    for record in records:
        if len(output_rows) // 5 >= args.sample_size:
            break
        record_id = record["record_id"]
        try:
            wave, fs = load_wfdb_12lead(Path(record["record_path"]), physical=True)
            landmarks, original_measure = landmarks_and_measurements(wave, fs)
            variants = [("identity", 0, 0.0, wave, original_measure)]
            for direction in (-1, 1):
                for strength in STRENGTHS:
                    edited = transform_waveform(wave, landmarks, phenotype, strength, direction)
                    rpeaks_hint = None if phenotype == "rr_irregularity" else landmarks["rpeaks"]
                    _, edited_measure = landmarks_and_measurements(edited, fs, rpeaks_hint=rpeaks_hint)
                    variants.append((variant_name(phenotype, direction), direction, strength, edited, edited_measure))
            record_inputs, record_rows = [], []
            for variant, direction, strength, variant_wave, measurement in variants:
                record_inputs.append(preprocess(model_name, variant_wave))
                record_rows.append(
                    {
                        "record_id": record_id, "record_path": record["record_path"],
                        "subject_id": record.get("subject_id", ""), "variant": variant,
                        "direction_sign": direction, "strength": strength, "fs": fs,
                        "rr_mean_ms": measurement["rr_mean_ms"], "rr_cv": measurement["rr_cv"],
                        "qrs_duration_ms": measurement["qrs_duration_ms"],
                        "qt_interval_ms": measurement["qt_interval_ms"],
                    }
                )
            model_inputs.extend(record_inputs)
            output_rows.extend(record_rows)
        except Exception as exc:
            failures.append({"record_id": record_id, "error": f"{type(exc).__name__}: {exc}"})
    successful_records = len(output_rows) // 5
    if successful_records != args.sample_size:
        raise RuntimeError(
            f"Only {successful_records}/{args.sample_size} records produced all five waveform variants"
        )

    encoder, model_status = load_encoder(model_name, args.device)
    pooled = infer_pooled(model_name, encoder, np.stack(model_inputs), args.device, args.model_batch)
    scaler = bundle["scaler"]; heads = bundle["heads"]
    standardized = scaler.transform(pooled)
    for head_name in names:
        output = heads[head_name]["clf"].decision_function(standardized)
        for row, value in zip(output_rows, output):
            row[f"head_{head_name}"] = float(value)

    protocol = "frozen_atom" if model_name == "ecg_jepa" else "cohort_adapted_atom"
    for seed in (4311, 4312, 4313):
        result_path = pair_root / "steering" / protocol / f"seed{seed}" / target / "result.json"
        result = json.loads(result_path.read_text())
        sae = load_sae(Path(result["checkpoint"]), args.device)
        codes = encode_all(sae, pooled, args.device)
        selected = np.asarray(result["selected_atoms"]["top5"], dtype=int)
        decoder_raw = (
            sae.W_dec.detach().cpu().numpy()
            * sae.sigma.detach().cpu().numpy()[:, None]
        )
        target_coefficient_raw = (
            np.asarray(heads[target]["clf"].coef_).reshape(-1) / scaler.scale_
        )
        latent_target_gradient = target_coefficient_raw @ decoder_raw
        activation_sum = codes[:, selected].sum(axis=1)
        contribution = codes[:, selected] @ latent_target_gradient[selected]
        active = (codes[:, selected] > 0).sum(axis=1)
        for row, value, logit_value, count in zip(output_rows, activation_sum, contribution, active):
            row[f"sae_seed{seed}_top5_activation_sum"] = float(value)
            row[f"sae_seed{seed}_top5_logit_contribution"] = float(logit_value)
            row[f"sae_seed{seed}_top5_active"] = int(count)
            row[f"sae_seed{seed}_atoms"] = "|".join(map(str, selected.tolist()))

    frame = pd.DataFrame(output_rows)
    frame.insert(0, "model", "ECG-JEPA" if model_name == "ecg_jepa" else "ECG-FM")
    frame.insert(1, "model_suffix", model_suffix); frame.insert(2, "cohort", cohort)
    frame.insert(3, "phenotype", phenotype); frame.insert(4, "target", target)
    frame.to_csv(worker_out / "per_variant_metrics.csv", index=False)
    pd.DataFrame(failures, columns=["record_id", "error"]).to_csv(worker_out / "failures.csv", index=False)
    metadata = {
        "schema_version": 1, "task_index": args.task_index, "model": model_name,
        "model_suffix": model_suffix, "cohort": cohort, "phenotype": phenotype, "target": target,
        "requested_records": args.sample_size, "successful_records": int(frame.record_id.nunique()),
        "candidate_records_examined": int(frame.record_id.nunique()) + len(failures),
        "failed_records": len(failures), "variants_per_success": 5, "rows": len(frame),
        "protocol": protocol, "sae_seeds": [4311, 4312, 4313], "model_status": model_status,
        "status": "complete",
    }
    complete.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
