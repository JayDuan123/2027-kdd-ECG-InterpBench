from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from benchmark_v1.adapters.ecg_fm import load_12lead_waveform
from benchmark_v1.adapters.ecg_jepa import build_waveform_index
from benchmark_v1.config import WORKSPACE


CSFM_ROOT = WORKSPACE / "Cardiac-Sensing-FM"
CSFM_CHECKPOINT = CSFM_ROOT / "pretrained" / "CSFM_tiny.pth"
CSFM_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CSFM_CHANNELS = list(range(12))
CSFM_TARGET_SAMPLES = 2500
CSFM_DEPTH = 6


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
        raise RuntimeError(f"missing dependencies for CSFM waveform loading: {', '.join(missing)}")

    import numpy as np
    from scipy.signal import resample

    wave = load_12lead_waveform(ecg_id, waveform_index)
    if wave.shape[1] != CSFM_TARGET_SAMPLES:
        wave = resample(wave, CSFM_TARGET_SAMPLES, axis=1).astype(np.float32)
    return wave.astype(np.float32)


def prepare_inputs(ecg_ids: list[str]) -> tuple[Any | None, dict[str, Any]]:
    missing = missing_dependencies()
    meta = {
        "requested_records": len(ecg_ids),
        "missing_dependencies": missing,
        "target_shape_per_record": [len(CSFM_LEADS), CSFM_TARGET_SAMPLES],
    }
    if missing:
        return None, meta

    import numpy as np

    index = build_waveform_index()
    waves = [load_waveform(ecg_id, index) for ecg_id in ecg_ids]
    batch = np.stack(waves, axis=0) if waves else np.empty((0, len(CSFM_LEADS), CSFM_TARGET_SAMPLES))
    meta["batch_shape"] = list(batch.shape)
    meta["found_records"] = len(waves)
    return batch, meta


def try_load_model(device: str = "cpu") -> tuple[Any | None, str]:
    if not CSFM_CHECKPOINT.exists():
        return None, f"checkpoint missing: {CSFM_CHECKPOINT}"
    sys.path.insert(0, str(CSFM_ROOT))
    try:
        import torch
        from network.model import CSFM_model

        model = CSFM_model("Tiny")
        checkpoint = torch.load(str(CSFM_CHECKPOINT), map_location="cpu")
        state = {
            k.replace("encoder.", ""): v
            for k, v in checkpoint.items()
            if k.startswith("encoder.") and "mlp_head" not in k
        }
        model.load_state_dict(state, strict=False)
        model.mlp_head = torch.nn.Identity()
        model.to(device)
        model.eval()
        return model, f"loaded, depth={len(model.transformer.layers)}, embed_dim={model.encoder_dim}, device={device}"
    except Exception as exc:  # pragma: no cover - optional runtime.
        return None, f"load failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            sys.path.remove(str(CSFM_ROOT))
        except ValueError:
            pass


def parse_layer_spec(spec: str, depth: int = CSFM_DEPTH) -> list[int]:
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
        raise RuntimeError(f"missing dependencies for CSFM activation extraction: {', '.join(missing)}")

    import torch

    model, status = try_load_model(device=device)
    if model is None:
        raise RuntimeError(status)
    blocks = model.transformer.layers
    if max(layers, default=-1) >= len(blocks):
        raise ValueError(f"requested layer outside transformer depth {len(blocks)}")

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            # Transformer.forward does: x = ff(x) + x. Capture the full post-block tensor.
            block_output = output + inputs[0]
            captured[layer_idx] = block_output.detach().cpu().numpy().astype("float32")

        return hook

    for layer_idx in layers:
        handles.append(blocks[layer_idx][1].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            source = torch.as_tensor(batch, dtype=torch.float32, device=device)
            channel = torch.as_tensor(CSFM_CHANNELS, dtype=torch.long, device=device)
            pooled = model(source, channel, task="cls").detach().cpu().numpy().astype("float32")
    finally:
        for handle in handles:
            handle.remove()

    return {
        "pooled": pooled,
        "layers": captured,
        "model_status": status,
        "depth": len(blocks),
    }
