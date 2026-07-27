from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from benchmark_v1.adapters.ecg_jepa import (
    build_waveform_index,
    load_reduced_waveform,
    parse_layer_spec,
    prepare_smoke_inputs,
)
from benchmark_v1.config import WORKSPACE


ECG_FM_ROOT = WORKSPACE / "ECG-FM"
FAIRSEQ_SIGNALS_ROOT = WORKSPACE / "fairseq-signals"
ECG_FM_CHECKPOINT = ECG_FM_ROOT / "ckpts" / "mimic_iv_ecg_physionet_pretrained.pt"
ECG_FM_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_FM_TARGET_SAMPLES = 5000
ECG_FM_DEPTH = 12


def dependency_status() -> dict[str, bool]:
    return {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "omegaconf": importlib.util.find_spec("omegaconf") is not None,
        "hydra": importlib.util.find_spec("hydra") is not None,
    }


def missing_dependencies() -> list[str]:
    return [name for name, ok in dependency_status().items() if not ok]


def load_12lead_waveform(ecg_id: str, waveform_index: dict[str, dict[str, Path]] | None = None) -> Any:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for ECG-FM waveform loading: {', '.join(missing)}")

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
    indices = [lead_to_idx[lead] for lead in ECG_FM_LEADS]
    mat = loadmat(entry["mat"])
    wave = np.asarray(mat["val"], dtype=np.float32)
    if wave.shape[0] != header.n_signals and wave.shape[-1] == header.n_signals:
        wave = wave.T
    wave = wave[indices]
    if wave.shape[1] != ECG_FM_TARGET_SAMPLES:
        wave = resample(wave, ECG_FM_TARGET_SAMPLES, axis=1).astype(np.float32)
    mean = wave.mean(axis=1, keepdims=True)
    std = wave.std(axis=1, keepdims=True)
    wave = (wave - mean) / np.maximum(std, 1e-6)
    return wave.astype(np.float32)


def prepare_inputs(ecg_ids: list[str]) -> tuple[Any | None, dict[str, Any]]:
    missing = missing_dependencies()
    meta = {
        "requested_records": len(ecg_ids),
        "missing_dependencies": missing,
        "target_shape_per_record": [len(ECG_FM_LEADS), ECG_FM_TARGET_SAMPLES],
    }
    if missing:
        return None, meta

    import numpy as np

    index = build_waveform_index()
    waves = [load_12lead_waveform(ecg_id, index) for ecg_id in ecg_ids]
    batch = np.stack(waves, axis=0) if waves else np.empty((0, len(ECG_FM_LEADS), ECG_FM_TARGET_SAMPLES))
    meta["batch_shape"] = list(batch.shape)
    meta["found_records"] = len(waves)
    return batch, meta


def try_load_model(device: str = "cpu") -> tuple[Any | None, str]:
    if not ECG_FM_CHECKPOINT.exists():
        return None, f"checkpoint missing: {ECG_FM_CHECKPOINT}"
    if not FAIRSEQ_SIGNALS_ROOT.exists():
        return None, f"fairseq-signals missing: {FAIRSEQ_SIGNALS_ROOT}"
    sys.path.insert(0, str(FAIRSEQ_SIGNALS_ROOT))
    try:
        import fairseq_signals.models.wav2vec2  # noqa: F401
        from fairseq_signals.models import build_model_from_checkpoint

        model = build_model_from_checkpoint(str(ECG_FM_CHECKPOINT))
        model.to(device)
        model.eval()
        return model, f"loaded, depth={len(model.encoder.layers)}, embed_dim={model.encoder.embed_dim}, device={device}"
    except Exception as exc:  # pragma: no cover - optional runtime.
        return None, f"load failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            sys.path.remove(str(FAIRSEQ_SIGNALS_ROOT))
        except ValueError:
            pass


def extract_activations(batch: Any, layers: list[int], device: str = "cpu") -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for ECG-FM activation extraction: {', '.join(missing)}")

    import torch

    model, status = try_load_model(device=device)
    if model is None:
        raise RuntimeError(status)
    blocks = model.encoder.layers
    if max(layers, default=-1) >= len(blocks):
        raise ValueError(f"requested layer outside encoder depth {len(blocks)}")

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            x = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = x.transpose(0, 1).detach().cpu().numpy().astype("float32")

        return hook

    for layer_idx in layers:
        handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            source = torch.as_tensor(batch, dtype=torch.float32, device=device)
            padding_mask = torch.zeros(source.shape[0], source.shape[-1], dtype=torch.bool, device=device)
            out = model.extract_features(source=source, padding_mask=padding_mask, mask=False)
            tokens = out["x"].detach().cpu().numpy().astype("float32")
            pooled = tokens.mean(axis=1).astype("float32")
    finally:
        for handle in handles:
            handle.remove()

    return {
        "pooled": pooled,
        "layers": captured,
        "tokens": tokens,
        "model_status": status,
        "depth": len(blocks),
    }
