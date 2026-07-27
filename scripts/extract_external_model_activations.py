#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.config import ROOT  # noqa: E402


DEFAULT_MANIFEST = ROOT / "results" / "multicohort" / "track_f_full" / "waveform_concepts_by_record.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "activations_external"

MODEL_ALIASES = {
    "CSFM": "csfm",
    "ECG-JEPA": "ecg_jepa",
    "ECG-FM": "ecg_fm",
    "CARDIAC-FM": "cardiac_fm",
    "HuBERT-ECG": "hubert_ecg",
    "ST-MEM": "st_mem",
    "csfm_cu118_commons": "csfm",
    "ecg_jepa_cu118_commons": "ecg_jepa",
    "ecg_fm_cu118_commons": "ecg_fm",
    "cardiac_fm_cu118_commons": "cardiac_fm",
    "hubert_ecg_cu118_commons": "hubert_ecg",
    "st_mem_cu118_commons": "st_mem",
}

MODEL_SUFFIX = {
    "csfm": "csfm_cu118_commons",
    "ecg_jepa": "ecg_jepa_cu118_commons",
    "ecg_fm": "ecg_fm_cu118_commons",
    "cardiac_fm": "cardiac_fm_cu118_commons",
    "hubert_ecg": "hubert_ecg_cu118_commons",
    "st_mem": "st_mem_cu118_commons",
}

COHORT_ALIASES = {
    "MIMIC-F": "mimic",
    "Chapman-F": "chapman",
    "CPSC-F": "cpsc",
    "Ningbo-F": "ningbo",
}


def canonical_model(name: str) -> str:
    try:
        return MODEL_ALIASES[name]
    except KeyError as exc:
        raise ValueError(f"unsupported model {name!r}; choices={sorted(set(MODEL_ALIASES))}") from exc


def canonical_cohort(name: str) -> str:
    return COHORT_ALIASES.get(name, name).lower().replace("_f", "").replace("-f", "")


def model_spec(model: str) -> dict[str, Any]:
    if model == "csfm":
        from benchmark_v1.adapters import csfm as adapter

        return {
            "display": "CSFM",
            "depth": adapter.CSFM_DEPTH,
            "layers": adapter.parse_layer_spec,
            "extract": adapter.extract_activations,
            "preprocess": preprocess_csfm,
            "target_shape": [len(adapter.CSFM_LEADS), adapter.CSFM_TARGET_SAMPLES],
        }
    if model == "ecg_jepa":
        from benchmark_v1.adapters import ecg_jepa as adapter

        return {
            "display": "ECG-JEPA",
            "depth": adapter.ECG_JEPA_DEPTH,
            "layers": adapter.parse_layer_spec,
            "extract": adapter.extract_encoder_activations,
            "preprocess": preprocess_ecg_jepa,
            "target_shape": [len(adapter.ECG_JEPA_LEADS), adapter.ECG_JEPA_TARGET_SAMPLES],
        }
    if model == "ecg_fm":
        from benchmark_v1.adapters import ecg_fm as adapter

        return {
            "display": "ECG-FM",
            "depth": adapter.ECG_FM_DEPTH,
            "layers": adapter.parse_layer_spec,
            "extract": adapter.extract_activations,
            "preprocess": preprocess_ecg_fm,
            "target_shape": [len(adapter.ECG_FM_LEADS), adapter.ECG_FM_TARGET_SAMPLES],
        }
    if model == "cardiac_fm":
        from benchmark_v1.adapters import cardiac_fm as adapter

        return {
            "display": "CARDIAC-FM",
            "depth": adapter.CARDIAC_FM_DEPTH,
            "layers": adapter.parse_layer_spec,
            "extract": adapter.extract_activations,
            "preprocess": preprocess_cardiac_fm,
            "target_shape": [len(adapter.CARDIAC_FM_LEADS), adapter.CARDIAC_FM_TARGET_SAMPLES],
        }
    if model == "hubert_ecg":
        from benchmark_v1.adapters import hubert_ecg as adapter

        return {
            "display": "HuBERT-ECG",
            "depth": adapter.HUBERT_ECG_DEPTH,
            "layers": adapter.parse_layer_spec,
            "extract": adapter.extract_activations,
            "preprocess": preprocess_hubert_ecg,
            "target_shape": [len(adapter.HUBERT_ECG_LEADS) * adapter.HUBERT_ECG_TARGET_SAMPLES],
            "lead_shape_before_flatten": [len(adapter.HUBERT_ECG_LEADS), adapter.HUBERT_ECG_TARGET_SAMPLES],
        }
    if model == "st_mem":
        from benchmark_v1.adapters import st_mem as adapter

        return {
            "display": "ST-MEM",
            "depth": adapter.ST_MEM_DEPTH,
            "layers": adapter.parse_layer_spec,
            "extract": adapter.extract_activations,
            "preprocess": preprocess_st_mem,
            "target_shape": [len(adapter.ST_MEM_LEADS), adapter.ST_MEM_TARGET_SAMPLES],
        }
    raise ValueError(f"unsupported canonical model {model!r}")


def read_manifest(path: Path, cohort: str, offset: int, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    skipped = 0
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if canonical_cohort(row.get("cohort", "")) != cohort:
                continue
            if row.get("status") != "ok":
                continue
            if skipped < offset:
                skipped += 1
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def write_record_ids(path: Path, rows: list[dict[str, str]], cohort: str) -> None:
    fields = ["ecg_id", "record_name", "cohort", "record_path", "subject_id", "study_id_or_record_key"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record_id = row.get("record_id") or Path(row.get("record_path", "")).name
            writer.writerow(
                {
                    "ecg_id": f"{cohort}:{record_id}",
                    "record_name": record_id,
                    "cohort": cohort,
                    "record_path": row.get("record_path", ""),
                    "subject_id": row.get("subject_id", ""),
                    "study_id_or_record_key": row.get("study_id_or_record_key", ""),
                }
            )


def import_runtime() -> tuple[Any, Any, Any]:
    import numpy as np
    from scipy.signal import resample
    import wfdb

    return np, resample, wfdb


def load_wfdb_12lead(base: Path, physical: bool = True) -> tuple[Any, int]:
    np, _resample, wfdb = import_runtime()
    if physical:
        signals, fields = wfdb.rdsamp(str(base))
        sig_names = fields.get("sig_name", [])
        fs = int(float(fields.get("fs", 0)))
    else:
        record = wfdb.rdrecord(str(base), physical=False)
        signals = record.d_signal
        sig_names = record.sig_name
        fs = int(float(record.fs))
    lead_to_idx = {str(lead).strip(): idx for idx, lead in enumerate(sig_names)}
    required = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    missing = [lead for lead in required if lead not in lead_to_idx]
    if missing:
        raise ValueError(f"{base} missing required 12-lead channels: {missing}")
    wave = np.asarray(signals, dtype=np.float32).T
    wave = wave[[lead_to_idx[lead] for lead in required]]
    return wave, fs


def zscore_resample_12lead(base: Path, target_samples: int) -> Any:
    np, resample, _wfdb = import_runtime()
    wave, _fs = load_wfdb_12lead(base)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    if wave.shape[1] != target_samples:
        wave = resample(wave, target_samples, axis=1).astype(np.float32)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    mean = wave.mean(axis=1, keepdims=True)
    std = wave.std(axis=1, keepdims=True)
    wave = (wave - mean) / np.maximum(std, 1e-6)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    return wave.astype(np.float32)


def preprocess_ecg_fm(base: Path) -> Any:
    from benchmark_v1.adapters.ecg_fm import ECG_FM_TARGET_SAMPLES

    return zscore_resample_12lead(base, ECG_FM_TARGET_SAMPLES)


def preprocess_cardiac_fm(base: Path) -> Any:
    from benchmark_v1.adapters.cardiac_fm import CARDIAC_FM_TARGET_SAMPLES

    return zscore_resample_12lead(base, CARDIAC_FM_TARGET_SAMPLES)


def preprocess_csfm(base: Path) -> Any:
    np, resample, _wfdb = import_runtime()
    from benchmark_v1.adapters.csfm import CSFM_TARGET_SAMPLES

    wave, _fs = load_wfdb_12lead(base, physical=False)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    if wave.shape[1] != CSFM_TARGET_SAMPLES:
        wave = resample(wave, CSFM_TARGET_SAMPLES, axis=1).astype(np.float32)
    return np.asarray(wave, dtype=np.float32)


def preprocess_ecg_jepa(base: Path) -> Any:
    np, resample, _wfdb = import_runtime()
    from benchmark_v1.adapters.ecg_jepa import ECG_JEPA_LEADS, ECG_JEPA_TARGET_SAMPLES, PTBXL_HEADER_LEADS

    wave, _fs = load_wfdb_12lead(base)
    lead_to_idx = {lead: idx for idx, lead in enumerate(PTBXL_HEADER_LEADS)}
    wave = wave[[lead_to_idx[lead] for lead in ECG_JEPA_LEADS]]
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    if wave.shape[1] != ECG_JEPA_TARGET_SAMPLES:
        wave = resample(wave, ECG_JEPA_TARGET_SAMPLES, axis=1).astype(np.float32)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    mean = wave.mean(axis=1, keepdims=True)
    std = wave.std(axis=1, keepdims=True)
    wave = (wave - mean) / np.maximum(std, 1e-6)
    return np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def preprocess_hubert_ecg(base: Path) -> Any:
    from benchmark_v1.adapters.hubert_ecg import HUBERT_ECG_TARGET_SAMPLES

    wave = zscore_resample_12lead(base, HUBERT_ECG_TARGET_SAMPLES)
    return wave.reshape(-1).astype("float32")


def preprocess_st_mem(base: Path) -> Any:
    np, resample, _wfdb = import_runtime()
    from benchmark_v1.adapters.st_mem import ST_MEM_TARGET_SAMPLES

    wave, _fs = load_wfdb_12lead(base, physical=False)
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    wave = resample(wave, 2500, axis=1).astype(np.float32)
    start = (wave.shape[1] - ST_MEM_TARGET_SAMPLES) // 2
    return wave[:, start : start + ST_MEM_TARGET_SAMPLES].astype(np.float32)


def build_batch(rows: list[dict[str, str]], preprocess: Callable[[Path], Any]) -> tuple[Any | None, list[dict[str, str]], list[dict[str, str]]]:
    np, _resample, _wfdb = import_runtime()
    waves = []
    ok_rows: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    for row in rows:
        record_path = row.get("record_path", "")
        try:
            waves.append(preprocess(Path(record_path)))
            ok_rows.append(row)
        except Exception as exc:
            failures.append(
                {
                    "record_id": row.get("record_id", ""),
                    "record_path": record_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not waves:
        return None, ok_rows, failures
    return np.stack(waves, axis=0), ok_rows, failures


def write_failures(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["record_id", "record_path", "error"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract model activations for external ECG cohorts from Track F manifest paths.")
    parser.add_argument("--model", required=True, help="Model name or activation suffix.")
    parser.add_argument("--cohort", required=True, help="External cohort name, e.g. MIMIC-F, Chapman-F, CPSC-F, Ningbo-F.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--shard-name", default="")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument(
        "--pool-layer-activations",
        action="store_true",
        help="Save token-mean [batch, hidden] layer arrays instead of full token tensors.",
    )
    parser.add_argument("--skip-model-load", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = canonical_model(args.model)
    cohort = canonical_cohort(args.cohort)
    suffix = MODEL_SUFFIX[model]
    spec = model_spec(model)
    shard_name = args.shard_name or f"{cohort}_offset{args.offset:06d}_n{args.limit:04d}"
    run_out_dir = args.out_dir / suffix / f"{cohort}_f" / shard_name
    run_out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest, cohort, args.offset, args.limit)
    batch, ok_rows, failures = build_batch(rows, spec["preprocess"])
    write_record_ids(run_out_dir / "record_ids.csv", ok_rows, f"{cohort}_f")
    write_failures(run_out_dir / "failed_records.csv", failures)

    layers = [] if args.layers == "pooled" else spec["layers"](args.layers, depth=spec["depth"])
    payload = {
        "model": spec["display"],
        "model_suffix": suffix,
        "cohort": f"{cohort}_f",
        "offset": args.offset,
        "limit": args.limit,
        "shard_name": shard_name,
        "selected_records": len(rows),
        "loaded_records": len(ok_rows),
        "failed_records": len(failures),
        "target_shape_per_record": spec["target_shape"],
        "lead_shape_before_flatten": spec.get("lead_shape_before_flatten", ""),
        "batch_shape": None if batch is None else list(batch.shape),
        "layers": layers,
    }
    (run_out_dir / "inputs_shape.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    activation_files: list[str] = []
    if args.save_activations and batch is not None and not args.skip_model_load:
        import numpy as np

        activations = spec["extract"](batch, layers, device=args.device)
        np.save(run_out_dir / "pooled.npy", activations["pooled"])
        activation_files.append("pooled.npy")
        layer_shapes = {}
        for layer_idx, values in sorted(activations["layers"].items()):
            if args.pool_layer_activations and values.ndim == 3:
                values = values.mean(axis=1, dtype=np.float32)
            filename = f"layer_{layer_idx:02d}.npy"
            np.save(run_out_dir / filename, values)
            activation_files.append(filename)
            layer_shapes[str(layer_idx)] = list(values.shape)
        activation_meta = {
            **payload,
            "device": args.device,
            "input_shape": list(batch.shape),
            "pooled_shape": list(activations["pooled"].shape),
            "model_status": activations.get("model_status", ""),
            "layer_shapes": layer_shapes,
            "layer_aggregation": "token_mean" if args.pool_layer_activations else "unpooled_tokens",
            "pooled_file": "pooled.npy",
            "layer_file_template": "layer_{layer:02d}.npy",
        }
        (run_out_dir / "activation_metadata.json").write_text(
            json.dumps(activation_meta, indent=2) + "\n",
            encoding="utf-8",
        )
        activation_files.append("activation_metadata.json")

    report = [
        f"# External Activation Shard: {spec['display']}",
        "",
        f"- cohort: {cohort}_f",
        f"- offset: {args.offset}",
        f"- limit: {args.limit}",
        f"- requested rows: {len(rows)}",
        f"- loaded records: {len(ok_rows)}",
        f"- failed records: {len(failures)}",
        f"- layers: {','.join(str(x) for x in layers)}",
        f"- device: {args.device}",
        f"- batch shape: {payload['batch_shape']}",
        f"- activation files: {', '.join(activation_files) if activation_files else 'none'}",
        "",
    ]
    (run_out_dir / "adapter_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {run_out_dir / 'adapter_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
