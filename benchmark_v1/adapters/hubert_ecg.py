from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from benchmark_v1.adapters.ecg_jepa import build_waveform_index
from benchmark_v1.adapters.ecg_fm import ECG_FM_LEADS
from benchmark_v1.config import WORKSPACE


HUBERT_ECG_ROOT = WORKSPACE / "HuBERT-ECG"
HUBERT_ECG_CHECKPOINT = HUBERT_ECG_ROOT / "checkpoints" / "hubert-ecg-base"
HUBERT_ECG_LEADS = ECG_FM_LEADS
HUBERT_ECG_TARGET_SAMPLES = 2500
HUBERT_ECG_DEPTH = 12


def dependency_status() -> dict[str, bool]:
    return {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "hubert_ecg": importlib.util.find_spec("hubert_ecg") is not None,
    }


def missing_dependencies() -> list[str]:
    return [name for name, ok in dependency_status().items() if not ok]


def parse_layer_spec(spec: str, depth: int = HUBERT_ECG_DEPTH) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(depth))
    if spec in {"last", "-1"}:
        return [depth - 1]
    layers = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if any(layer < 0 or layer >= depth for layer in layers):
        raise ValueError(f"layer spec {spec!r} outside depth {depth}")
    return layers


def load_flattened_waveform(ecg_id: str, waveform_index: dict[str, dict[str, Path]] | None = None) -> Any:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for HuBERT-ECG waveform loading: {', '.join(missing)}")

    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample

    from benchmark_v1.adapters.ecg_jepa import parse_header, record_name_for_ecg_id
    from benchmark_v1.config import PTBXL_WAVEFORM_ROOT

    index = waveform_index if waveform_index is not None else build_waveform_index()
    record_name = record_name_for_ecg_id(ecg_id)
    entry = index.get(record_name)
    if entry is None:
        raise FileNotFoundError(f"record {record_name} not found under {PTBXL_WAVEFORM_ROOT}")
    header = parse_header(entry["hea"])
    lead_to_idx = {lead: idx for idx, lead in enumerate(header.leads)}
    indices = [lead_to_idx[lead] for lead in HUBERT_ECG_LEADS]
    mat = loadmat(entry["mat"])
    wave = np.asarray(mat["val"], dtype=np.float32)
    if wave.shape[0] != header.n_signals and wave.shape[-1] == header.n_signals:
        wave = wave.T
    wave = wave[indices]
    if wave.shape[1] != HUBERT_ECG_TARGET_SAMPLES:
        wave = resample(wave, HUBERT_ECG_TARGET_SAMPLES, axis=1).astype(np.float32)
    mean = wave.mean(axis=1, keepdims=True)
    std = wave.std(axis=1, keepdims=True)
    wave = (wave - mean) / np.maximum(std, 1e-6)
    return wave.reshape(-1).astype(np.float32)


def prepare_inputs(ecg_ids: list[str]) -> tuple[Any | None, dict[str, Any]]:
    missing = missing_dependencies()
    meta = {
        "requested_records": len(ecg_ids),
        "missing_dependencies": missing,
        "target_shape_per_record": [len(HUBERT_ECG_LEADS) * HUBERT_ECG_TARGET_SAMPLES],
        "lead_shape_before_flatten": [len(HUBERT_ECG_LEADS), HUBERT_ECG_TARGET_SAMPLES],
    }
    if missing:
        return None, meta

    import numpy as np

    index = build_waveform_index()
    waves = [load_flattened_waveform(ecg_id, index) for ecg_id in ecg_ids]
    batch = np.stack(waves, axis=0) if waves else np.empty((0, len(HUBERT_ECG_LEADS) * HUBERT_ECG_TARGET_SAMPLES))
    meta["batch_shape"] = list(batch.shape)
    meta["found_records"] = len(waves)
    return batch, meta


def try_load_model(device: str = "cpu") -> tuple[Any | None, str]:
    if not HUBERT_ECG_CHECKPOINT.exists():
        return None, f"checkpoint missing: {HUBERT_ECG_CHECKPOINT}"
    missing = missing_dependencies()
    if missing:
        return None, f"missing dependencies: {', '.join(missing)}"
    try:
        import torch
        import hubert_ecg  # noqa: F401
        from transformers import AutoModel

        model = AutoModel.from_pretrained(str(HUBERT_ECG_CHECKPOINT), trust_remote_code=True)
        model.to(torch.device(device))
        model.eval()
        return model, (
            f"loaded, depth={model.config.num_hidden_layers}, "
            f"hidden={model.config.hidden_size}, device={device}"
        )
    except Exception as exc:  # pragma: no cover - optional runtime.
        return None, f"load failed: {type(exc).__name__}: {exc}"


def extract_activations(batch: Any, layers: list[int], device: str = "cpu") -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for HuBERT-ECG activation extraction: {', '.join(missing)}")

    import torch

    model, status = try_load_model(device=device)
    if model is None:
        raise RuntimeError(status)
    depth = model.config.num_hidden_layers
    if max(layers, default=-1) >= depth:
        raise ValueError(f"requested layer outside encoder depth {depth}")

    with torch.no_grad():
        input_values = torch.as_tensor(batch, dtype=torch.float32, device=device)
        out = model(input_values, output_hidden_states=True, return_dict=True)
        hidden_states = out.hidden_states
        tokens = out.last_hidden_state.detach().cpu().numpy().astype("float32")
        pooled = tokens.mean(axis=1).astype("float32")
        captured = {
            layer_idx: hidden_states[layer_idx + 1].detach().cpu().numpy().astype("float32")
            for layer_idx in layers
        }

    return {
        "pooled": pooled,
        "layers": captured,
        "tokens": tokens,
        "model_status": status,
        "depth": depth,
    }
