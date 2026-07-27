from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from benchmark_v1.adapters.ecg_fm import load_12lead_waveform
from benchmark_v1.adapters.ecg_jepa import build_waveform_index
from benchmark_v1.config import WORKSPACE


ST_MEM_ROOT = WORKSPACE / "ST-MEM"
ST_MEM_CHECKPOINT = ST_MEM_ROOT / "checkpoints" / "encoder.pth"
ST_MEM_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ST_MEM_TARGET_SAMPLES = 2250
ST_MEM_DEPTH = 12


def dependency_status() -> dict[str, bool]:
    return {
        "numpy": importlib.util.find_spec("numpy") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "einops": importlib.util.find_spec("einops") is not None,
    }


def missing_dependencies() -> list[str]:
    return [name for name, ok in dependency_status().items() if not ok]


def load_waveform(ecg_id: str, waveform_index: dict[str, dict[str, Path]] | None = None) -> Any:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for ST-MEM waveform loading: {', '.join(missing)}")

    import numpy as np
    from scipy.signal import resample

    wave = load_12lead_waveform(ecg_id, waveform_index)
    wave = resample(wave, 2500, axis=1).astype(np.float32)
    start = (wave.shape[1] - ST_MEM_TARGET_SAMPLES) // 2
    wave = wave[:, start : start + ST_MEM_TARGET_SAMPLES]
    return wave.astype(np.float32)


def prepare_inputs(ecg_ids: list[str]) -> tuple[Any | None, dict[str, Any]]:
    missing = missing_dependencies()
    meta = {
        "requested_records": len(ecg_ids),
        "missing_dependencies": missing,
        "target_shape_per_record": [len(ST_MEM_LEADS), ST_MEM_TARGET_SAMPLES],
    }
    if missing:
        return None, meta

    import numpy as np

    index = build_waveform_index()
    waves = [load_waveform(ecg_id, index) for ecg_id in ecg_ids]
    batch = np.stack(waves, axis=0) if waves else np.empty((0, len(ST_MEM_LEADS), ST_MEM_TARGET_SAMPLES))
    meta["batch_shape"] = list(batch.shape)
    meta["found_records"] = len(waves)
    return batch, meta


def try_load_model(device: str = "cpu") -> tuple[Any | None, str]:
    if not ST_MEM_CHECKPOINT.exists():
        return None, f"checkpoint missing: {ST_MEM_CHECKPOINT}"
    sys.path.insert(0, str(ST_MEM_ROOT))
    try:
        import torch
        from models.encoder.st_mem_vit import st_mem_vit_base

        checkpoint = torch.load(str(ST_MEM_CHECKPOINT), map_location="cpu")
        model = st_mem_vit_base(num_leads=len(ST_MEM_LEADS))
        model.load_state_dict(checkpoint["model"], strict=True)
        model.to(device)
        model.eval()
        return model, f"loaded, epoch={checkpoint.get('epoch')}, depth={model.depth}, width={model.width}, device={device}"
    except Exception as exc:  # pragma: no cover - optional runtime.
        return None, f"load failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            sys.path.remove(str(ST_MEM_ROOT))
        except ValueError:
            pass


def parse_layer_spec(spec: str, depth: int = ST_MEM_DEPTH) -> list[int]:
    if spec == "all":
        return list(range(depth))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0 or layer >= depth:
            raise ValueError(f"layer {layer} is outside valid range 0..{depth - 1}")
        layers.append(layer)
    return sorted(set(layers))


def extract_activations(batch: Any, layers: list[int], device: str = "cpu") -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for ST-MEM activation extraction: {', '.join(missing)}")

    import torch

    model, status = try_load_model(device=device)
    if model is None:
        raise RuntimeError(status)
    if max(layers, default=-1) >= model.depth:
        raise ValueError(f"requested layer outside transformer depth {model.depth}")

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer_idx] = output.detach().cpu().numpy().astype("float32")

        return hook

    for layer_idx in layers:
        block = getattr(model, f"block{layer_idx}")
        handles.append(block.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            source = torch.as_tensor(batch, dtype=torch.float32, device=device)
            pooled = model.forward_encoding(source).detach().cpu().numpy().astype("float32")
    finally:
        for handle in handles:
            handle.remove()

    return {
        "pooled": pooled,
        "layers": captured,
        "model_status": status,
        "depth": model.depth,
    }
