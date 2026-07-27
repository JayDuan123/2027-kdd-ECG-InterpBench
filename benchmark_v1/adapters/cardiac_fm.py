from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmark_v1.adapters.ecg_fm import (
    ECG_FM_CHECKPOINT,
    ECG_FM_DEPTH,
    ECG_FM_LEADS,
    ECG_FM_TARGET_SAMPLES,
    FAIRSEQ_SIGNALS_ROOT,
    dependency_status,
    load_12lead_waveform,
    parse_layer_spec,
)
from benchmark_v1.adapters.ecg_jepa import build_waveform_index
from benchmark_v1.config import WORKSPACE


CARDIAC_FM_ROOT = WORKSPACE / "CARDIAC-FM"
CARDIAC_FM_STAGE1_CHECKPOINT = CARDIAC_FM_ROOT / "checkpoints" / "hf" / "model_epoch_8.pth"
CARDIAC_FM_LEADS = ECG_FM_LEADS
CARDIAC_FM_TARGET_SAMPLES = ECG_FM_TARGET_SAMPLES
CARDIAC_FM_DEPTH = ECG_FM_DEPTH


def missing_dependencies() -> list[str]:
    return [name for name, ok in dependency_status().items() if not ok]


def prepare_inputs(ecg_ids: list[str]) -> tuple[Any | None, dict[str, Any]]:
    missing = missing_dependencies()
    meta = {
        "requested_records": len(ecg_ids),
        "missing_dependencies": missing,
        "target_shape_per_record": [len(CARDIAC_FM_LEADS), CARDIAC_FM_TARGET_SAMPLES],
    }
    if missing:
        return None, meta

    import numpy as np

    index = build_waveform_index()
    waves = [load_12lead_waveform(ecg_id, index) for ecg_id in ecg_ids]
    batch = np.stack(waves, axis=0) if waves else np.empty((0, len(CARDIAC_FM_LEADS), CARDIAC_FM_TARGET_SAMPLES))
    meta["batch_shape"] = list(batch.shape)
    meta["found_records"] = len(waves)
    return batch, meta


def _load_stage1_into_ecg_fm(model: Any) -> tuple[int, int]:
    import torch
    import torch.nn as nn

    state = torch.load(str(CARDIAC_FM_STAGE1_CHECKPOINT), map_location="cpu")
    encoder_state = {
        key.replace("module.ecg_encoder.", ""): value
        for key, value in state.items()
        if key.startswith("module.ecg_encoder.")
    }
    projection_state = {
        key.replace("module.ecg_projection.", ""): value
        for key, value in state.items()
        if key.startswith("module.ecg_projection.")
    }
    model.load_state_dict(encoder_state, strict=True)
    projection = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.1), nn.Linear(768, 512))
    projection.load_state_dict(projection_state, strict=True)
    projection.eval()
    model.ecg_projection_multi = projection
    return len(encoder_state), len(projection_state)


def try_load_model(device: str = "cpu") -> tuple[Any | None, str]:
    if not ECG_FM_CHECKPOINT.exists():
        return None, f"base ECG-FM checkpoint missing: {ECG_FM_CHECKPOINT}"
    if not CARDIAC_FM_STAGE1_CHECKPOINT.exists():
        return None, f"CARDIAC-FM stage1 checkpoint missing: {CARDIAC_FM_STAGE1_CHECKPOINT}"
    if not FAIRSEQ_SIGNALS_ROOT.exists():
        return None, f"fairseq-signals missing: {FAIRSEQ_SIGNALS_ROOT}"
    sys.path.insert(0, str(FAIRSEQ_SIGNALS_ROOT))
    try:
        import fairseq_signals.models.wav2vec2  # noqa: F401
        from fairseq_signals.models import build_model_from_checkpoint

        model = build_model_from_checkpoint(str(ECG_FM_CHECKPOINT))
        enc_keys, proj_keys = _load_stage1_into_ecg_fm(model)
        model.to(device)
        model.ecg_projection_multi.to(device)
        model.eval()
        return model, (
            f"loaded CARDIAC-FM ECG branch, depth={len(model.encoder.layers)}, "
            f"embed_dim={model.encoder.embed_dim}, encoder_keys={enc_keys}, "
            f"projection_keys={proj_keys}, device={device}"
        )
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
        raise RuntimeError(f"missing dependencies for CARDIAC-FM activation extraction: {', '.join(missing)}")

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
            pooled_768 = tokens.mean(axis=1)
            projected = model.ecg_projection_multi(torch.as_tensor(pooled_768, dtype=torch.float32, device=device))
            pooled = projected.detach().cpu().numpy().astype("float32")
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
