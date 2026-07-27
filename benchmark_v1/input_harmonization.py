"""Checkpoint-compatible waveform harmonization for ECG foundation models."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


FULL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
PROTOCOLS = ("native", "lead", "temporal", "joint")
CANONICAL_SAMPLES = 2500

MODEL_INTERFACES = {
    "cardiac_fm": {"leads": 12, "samples": 5000, "flatten": False, "depth": 12},
    "csfm": {"leads": 12, "samples": 2500, "flatten": False, "depth": 6},
    "ecg_fm": {"leads": 12, "samples": 5000, "flatten": False, "depth": 12},
    "ecg_jepa": {"leads": 8, "samples": 2500, "flatten": False, "depth": 12},
    "hubert_ecg": {"leads": 12, "samples": 2500, "flatten": True, "depth": 12},
    "st_mem": {"leads": 12, "samples": 2250, "flatten": False, "depth": 12},
}


def canonical_model_name(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    aliases = {
        "cardiacfm": "cardiac_fm",
        "ecgfm": "ecg_fm",
        "ecgjepa": "ecg_jepa",
        "hubertecg": "hubert_ecg",
        "stmem": "st_mem",
    }
    key = aliases.get(key, key)
    if key not in MODEL_INTERFACES:
        raise ValueError(f"unsupported model: {name}")
    return key


def validate_protocol(protocol: str) -> str:
    value = protocol.strip().lower().replace("_harmonized", "")
    if value not in PROTOCOLS:
        raise ValueError(f"unsupported input protocol: {protocol}")
    return value


def final_layer_for_model(model: str) -> int:
    spec = MODEL_INTERFACES[canonical_model_name(model)]
    return int(spec["depth"]) - 1


def _signal_calibration(header_path: Path) -> dict[str, tuple[float, float]]:
    """Return WFDB ADC gain and baseline by lead."""
    lines = header_path.read_text(encoding="utf-8").splitlines()
    first = lines[0].split()
    n_signals = int(first[1])
    calibration: dict[str, tuple[float, float]] = {}
    pattern = re.compile(r"^([-+0-9.eE]+)(?:\(([-+0-9.eE]+)\))?/")
    for line in lines[1 : 1 + n_signals]:
        fields = line.split()
        if len(fields) < 3:
            continue
        match = pattern.match(fields[2])
        gain = float(match.group(1)) if match else 1.0
        baseline = float(match.group(2) or 0.0) if match else 0.0
        calibration[fields[-1]] = (gain, baseline)
    return calibration


def load_physical_12lead(
    ecg_id: str,
    waveform_index: dict[str, dict[str, Path]] | None = None,
) -> tuple[Any, int]:
    """Load one PTB-XL Challenge-format record in physical units."""
    import numpy as np
    from scipy.io import loadmat

    from benchmark_v1.adapters.ecg_jepa import (
        build_waveform_index,
        parse_header,
        record_name_for_ecg_id,
    )
    from benchmark_v1.config import PTBXL_WAVEFORM_ROOT

    index = waveform_index if waveform_index is not None else build_waveform_index()
    record_name = record_name_for_ecg_id(ecg_id)
    entry = index.get(record_name)
    if entry is None:
        raise FileNotFoundError(f"record {record_name} not found under {PTBXL_WAVEFORM_ROOT}")
    header = parse_header(entry["hea"])
    lead_to_idx = {lead: idx for idx, lead in enumerate(header.leads)}
    missing = [lead for lead in FULL_LEADS if lead not in lead_to_idx]
    if missing:
        raise ValueError(f"record {record_name} is missing leads: {missing}")
    payload = loadmat(entry["mat"])
    if "val" not in payload:
        raise KeyError(f"{entry['mat']} does not contain MATLAB variable 'val'")
    wave = np.asarray(payload["val"], dtype=np.float32)
    if wave.shape[0] != header.n_signals and wave.shape[-1] == header.n_signals:
        wave = wave.T
    wave = wave[[lead_to_idx[lead] for lead in FULL_LEADS]]
    calibration = _signal_calibration(entry["hea"])
    for index_in_wave, lead in enumerate(FULL_LEADS):
        gain, baseline = calibration.get(lead, (1.0, 0.0))
        if not np.isfinite(gain) or abs(gain) < 1e-12:
            raise ValueError(f"invalid ADC gain for {record_name} lead {lead}: {gain}")
        wave[index_in_wave] = (wave[index_in_wave] - baseline) / gain
    return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0), header.sample_rate


def select_independent_leads(wave12: Any) -> Any:
    import numpy as np

    wave = np.asarray(wave12, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != len(FULL_LEADS):
        raise ValueError(f"expected [12, samples], got {wave.shape}")
    index = {lead: i for i, lead in enumerate(FULL_LEADS)}
    return wave[[index[lead] for lead in INDEPENDENT_LEADS]].copy()


def reconstruct_12_leads(wave8: Any) -> Any:
    """Reconstruct dependent limb leads from I and II."""
    import numpy as np

    wave = np.asarray(wave8, dtype=np.float32)
    if wave.ndim != 2 or wave.shape[0] != len(INDEPENDENT_LEADS):
        raise ValueError(f"expected [8, samples], got {wave.shape}")
    lead = {name: wave[i] for i, name in enumerate(INDEPENDENT_LEADS)}
    lead_i, lead_ii = lead["I"], lead["II"]
    values = {
        "I": lead_i,
        "II": lead_ii,
        "III": lead_ii - lead_i,
        "aVR": -(lead_i + lead_ii) / 2.0,
        "aVL": lead_i - lead_ii / 2.0,
        "aVF": lead_ii - lead_i / 2.0,
        **{name: lead[name] for name in INDEPENDENT_LEADS[2:]},
    }
    return np.stack([values[name] for name in FULL_LEADS]).astype(np.float32)


def zscore_per_lead(wave: Any) -> Any:
    import numpy as np

    values = np.nan_to_num(np.asarray(wave, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mean = values.mean(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    values = (values - mean) / np.maximum(scale, 1e-6)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def resample_to(wave: Any, samples: int) -> Any:
    import numpy as np
    from scipy.signal import resample

    values = np.asarray(wave, dtype=np.float32)
    if values.shape[1] == samples:
        return values.copy()
    return resample(values, samples, axis=1).astype(np.float32)


def build_model_input(
    wave12: Any,
    model: str,
    protocol: str,
    source_sample_rate: int = 500,
) -> Any:
    """Create one checkpoint-compatible model input from a common 12-lead record."""
    import numpy as np

    model_key = canonical_model_name(model)
    protocol_key = validate_protocol(protocol)
    spec = MODEL_INTERFACES[model_key]
    values = np.asarray(wave12, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(FULL_LEADS):
        raise ValueError(f"expected [12, samples], got {values.shape}")
    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be positive")

    lead_harmonized = protocol_key in {"lead", "joint"}
    temporal_harmonized = protocol_key in {"temporal", "joint"}
    independent = select_independent_leads(values)
    if int(spec["leads"]) == 8:
        values = independent
    elif lead_harmonized:
        values = reconstruct_12_leads(independent)

    values = zscore_per_lead(values)
    if temporal_harmonized:
        values = resample_to(values, CANONICAL_SAMPLES)

    if model_key == "st_mem":
        if temporal_harmonized:
            values = resample_to(values, int(spec["samples"]))
        else:
            values = resample_to(values, CANONICAL_SAMPLES)
            start = (values.shape[1] - int(spec["samples"])) // 2
            values = values[:, start : start + int(spec["samples"])]
    else:
        values = resample_to(values, int(spec["samples"]))

    expected = (int(spec["leads"]), int(spec["samples"]))
    if values.shape != expected:
        raise RuntimeError(f"{model_key}/{protocol_key} produced {values.shape}, expected {expected}")
    if bool(spec["flatten"]):
        return values.reshape(-1).astype(np.float32)
    return values.astype(np.float32)


def prepare_model_batch(
    ecg_ids: list[str],
    model: str,
    protocol: str,
    waveform_index: dict[str, dict[str, Path]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    from benchmark_v1.adapters.ecg_jepa import build_waveform_index

    model_key = canonical_model_name(model)
    protocol_key = validate_protocol(protocol)
    index = waveform_index if waveform_index is not None else build_waveform_index()
    waves = []
    sample_rates = []
    for ecg_id in ecg_ids:
        wave, sample_rate = load_physical_12lead(ecg_id, index)
        waves.append(build_model_input(wave, model_key, protocol_key, sample_rate))
        sample_rates.append(int(sample_rate))
    spec = MODEL_INTERFACES[model_key]
    if waves:
        batch = np.stack(waves).astype(np.float32)
    elif bool(spec["flatten"]):
        batch = np.empty((0, int(spec["leads"]) * int(spec["samples"])), dtype=np.float32)
    else:
        batch = np.empty((0, int(spec["leads"]), int(spec["samples"])), dtype=np.float32)
    metadata = {
        "model": model_key,
        "protocol": protocol_key,
        "requested_records": len(ecg_ids),
        "found_records": len(waves),
        "source_sample_rates": sorted(set(sample_rates)),
        "canonical_leads": list(INDEPENDENT_LEADS) if protocol_key in {"lead", "joint"} else "native",
        "canonical_samples": CANONICAL_SAMPLES if protocol_key in {"temporal", "joint"} else "native",
        "target_shape_per_record": list(batch.shape[1:]),
        "batch_shape": list(batch.shape),
    }
    return batch, metadata
