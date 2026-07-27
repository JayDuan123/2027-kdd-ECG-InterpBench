from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path
from typing import Any

from benchmark_v1.config import ROOT, WORKSPACE, PTBXL_WAVEFORM_ROOT


ECG_JEPA_ROOT = WORKSPACE / "ECG_JEPA"
ECG_JEPA_CHECKPOINT = ECG_JEPA_ROOT / "weights" / "multiblock_epoch100.pth"
DEFAULT_SPLIT_CSV = ROOT / "results" / "manifest" / "split.csv"

PTBXL_HEADER_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_JEPA_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
ECG_JEPA_TARGET_SAMPLES = 2500
ECG_JEPA_DEPTH = 12


@dataclass(frozen=True)
class HeaderInfo:
    record_name: str
    n_signals: int
    sample_rate: int
    n_samples: int
    leads: list[str]


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    detail: str


def record_name_for_ecg_id(ecg_id: str | int) -> str:
    return f"HR{int(ecg_id):05d}"


def dependency_status() -> list[DependencyStatus]:
    deps = [
        ("numpy", "numpy"),
        ("scipy.io", "scipy.io"),
        ("scipy.signal", "scipy.signal"),
        ("torch", "torch"),
    ]
    out: list[DependencyStatus] = []
    for display, module_name in deps:
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            out.append(DependencyStatus(display, False, "not importable"))
        else:
            out.append(DependencyStatus(display, True, "importable"))
    return out


def missing_dependencies() -> list[str]:
    return [dep.name for dep in dependency_status() if not dep.available]


def build_waveform_index(root: Path = PTBXL_WAVEFORM_ROOT) -> dict[str, dict[str, Path]]:
    index: dict[str, dict[str, Path]] = {}
    for header in root.glob("g*/HR*.hea"):
        record_name = header.stem
        mat = header.with_suffix(".mat")
        index[record_name] = {"hea": header, "mat": mat}
    return index


def parse_header(path: Path) -> HeaderInfo:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty header: {path}")
    first = lines[0].split()
    if len(first) < 4:
        raise ValueError(f"invalid header first line: {path}")
    record_name = first[0]
    n_signals = int(first[1])
    sample_rate = int(float(first[2]))
    n_samples = int(first[3])
    leads: list[str] = []
    for line in lines[1 : 1 + n_signals]:
        parts = line.split()
        if parts:
            leads.append(parts[-1])
    return HeaderInfo(record_name, n_signals, sample_rate, n_samples, leads)


def reduced_lead_indices(header: HeaderInfo) -> list[int]:
    lead_to_idx = {lead: idx for idx, lead in enumerate(header.leads)}
    missing = [lead for lead in ECG_JEPA_LEADS if lead not in lead_to_idx]
    if missing:
        raise ValueError(f"header {header.record_name} is missing ECG-JEPA leads: {missing}")
    return [lead_to_idx[lead] for lead in ECG_JEPA_LEADS]


def read_split_ids(split: str, limit: int, split_csv: Path = DEFAULT_SPLIT_CSV, offset: int = 0) -> list[str]:
    ids: list[str] = []
    seen = 0
    with split_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("split") != split:
                continue
            if seen < offset:
                seen += 1
                continue
            ecg_id = row.get("ecg_id")
            if not ecg_id:
                continue
            ids.append(ecg_id)
            if len(ids) >= limit:
                break
    return ids


def load_reduced_waveform(ecg_id: str, waveform_index: dict[str, dict[str, Path]] | None = None) -> Any:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for waveform loading: {', '.join(missing)}")

    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample

    index = waveform_index if waveform_index is not None else build_waveform_index()
    record_name = record_name_for_ecg_id(ecg_id)
    entry = index.get(record_name)
    if entry is None:
        raise FileNotFoundError(f"record {record_name} not found under {PTBXL_WAVEFORM_ROOT}")
    header = parse_header(entry["hea"])
    indices = reduced_lead_indices(header)
    mat = loadmat(entry["mat"])
    if "val" not in mat:
        raise KeyError(f"{entry['mat']} does not contain MATLAB variable 'val'")
    wave = np.asarray(mat["val"], dtype=np.float32)
    if wave.shape[0] != header.n_signals and wave.shape[-1] == header.n_signals:
        wave = wave.T
    wave = wave[indices]
    if wave.shape[1] != ECG_JEPA_TARGET_SAMPLES:
        wave = resample(wave, ECG_JEPA_TARGET_SAMPLES, axis=1).astype(np.float32)
    mean = wave.mean(axis=1, keepdims=True)
    std = wave.std(axis=1, keepdims=True)
    wave = (wave - mean) / np.maximum(std, 1e-6)
    return wave.astype(np.float32)


def prepare_smoke_inputs(ecg_ids: list[str]) -> tuple[Any | None, dict[str, Any]]:
    missing = missing_dependencies()
    index = build_waveform_index()
    records = []
    for ecg_id in ecg_ids:
        record_name = record_name_for_ecg_id(ecg_id)
        entry = index.get(record_name)
        if entry is None:
            records.append({"ecg_id": ecg_id, "record_name": record_name, "status": "missing"})
            continue
        header = parse_header(entry["hea"])
        records.append(
            {
                "ecg_id": ecg_id,
                "record_name": record_name,
                "status": "found",
                "sample_rate": header.sample_rate,
                "n_samples": header.n_samples,
                "input_leads": ",".join(ECG_JEPA_LEADS),
            }
        )

    meta: dict[str, Any] = {
        "requested_records": len(ecg_ids),
        "found_records": sum(1 for item in records if item["status"] == "found"),
        "missing_dependencies": missing,
        "records": records,
        "target_shape_per_record": [len(ECG_JEPA_LEADS), ECG_JEPA_TARGET_SAMPLES],
    }
    if missing:
        return None, meta

    import numpy as np

    waves = [load_reduced_waveform(ecg_id, index) for ecg_id in ecg_ids]
    batch = np.stack(waves, axis=0) if waves else np.empty((0, len(ECG_JEPA_LEADS), ECG_JEPA_TARGET_SAMPLES))
    meta["batch_shape"] = list(batch.shape)
    return batch, meta


def try_load_encoder(device: str = "cpu") -> tuple[Any | None, str]:
    if importlib.util.find_spec("torch") is None:
        return None, "torch not importable"
    if not ECG_JEPA_CHECKPOINT.exists():
        return None, f"checkpoint missing: {ECG_JEPA_CHECKPOINT}"
    if not (ECG_JEPA_ROOT / "models.py").exists():
        return None, f"models.py missing: {ECG_JEPA_ROOT / 'models.py'}"

    sys.path.insert(0, str(ECG_JEPA_ROOT))
    try:
        from models import load_encoder

        loaded = load_encoder(str(ECG_JEPA_CHECKPOINT), leads=list(range(len(ECG_JEPA_LEADS))))
        if isinstance(loaded, tuple):
            encoder, embed_dim = loaded
        else:
            encoder, embed_dim = loaded, "unknown"
        if hasattr(encoder, "to"):
            encoder.to(device)
        if hasattr(encoder, "eval"):
            encoder.eval()
        return encoder, f"loaded, embed_dim={embed_dim}, device={device}"
    except Exception as exc:  # pragma: no cover - depends on optional model runtime.
        return None, f"load failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            sys.path.remove(str(ECG_JEPA_ROOT))
        except ValueError:
            pass


def parse_layer_spec(spec: str, depth: int = ECG_JEPA_DEPTH) -> list[int]:
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


def extract_encoder_activations(batch: Any, layers: list[int], device: str = "cpu") -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError(f"missing dependencies for activation extraction: {', '.join(missing)}")

    import torch

    encoder, status = try_load_encoder(device=device)
    if encoder is None:
        raise RuntimeError(status)
    blocks = encoder.encoder_blocks.blocks
    if max(layers, default=-1) >= len(blocks):
        raise ValueError(f"requested layer outside encoder depth {len(blocks)}")

    captured: dict[int, Any] = {}
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            captured[layer_idx] = output.detach().cpu().numpy().astype("float32")

        return hook

    for layer_idx in layers:
        handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.no_grad():
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=device)
            pooled = encoder.representation(tensor).detach().cpu().numpy().astype("float32")
    finally:
        for handle in handles:
            handle.remove()

    return {
        "pooled": pooled,
        "layers": captured,
        "model_status": status,
        "depth": len(blocks),
    }
