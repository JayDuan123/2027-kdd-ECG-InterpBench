#!/usr/bin/env python
"""Fit train-only ridge probes to external-cohort layer activations.

Only waveform-derived continuous ECG measurements are probe concepts. Native
and ICD diagnoses remain downstream tasks and never enter this concept matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ACT_ROOT = ROOT / "results/activations_external_full_v1/layer_atlas"
OUT_ROOT = ROOT / "results/external_benchmark_v1/layer_probe"
MANIFESTS = {
    "chapman_f": ROOT / "results/activations_external_full_v1/plan_chapman_cpsc/layer_sample_manifest.csv",
    "cpsc_f": ROOT / "results/activations_external_full_v1/plan_chapman_cpsc/layer_sample_manifest.csv",
    "ningbo_f": ROOT / "results/activations_external_full_v1/plan_ningbo/layer_sample_manifest.csv",
    "mimic_f": ROOT / "results/activations_external_full_v1/plan_mimic_100k/mimic_layer_manifest.csv",
}
COHORT_VALUES = {"chapman_f": "chapman", "cpsc_f": "cpsc", "ningbo_f": "ningbo", "mimic_f": "mimic"}
BASE_CONCEPTS = (
    "rr_mean_ms", "qrs_duration_ms", "pr_interval_ms", "qt_like_ms",
    "r_amp_global_mv", "st_amp_global_mv", "t_amp_global_mv",
)
FAMILIES = {
    "rr_mean_ms": "rate_rhythm", "heart_rate_bpm": "rate_rhythm",
    "qrs_duration_ms": "interval", "pr_interval_ms": "interval",
    "qt_like_ms": "interval", "qtc_bazett_ms": "interval",
    "r_amp_global_mv": "amplitude", "st_amp_global_mv": "st_t",
    "t_amp_global_mv": "st_t",
}
ALPHAS = (0.1, 1.0, 10.0, 100.0)
VAL_R2_MIN = 0.04
CONTROL_MARGIN_MIN = 0.01
PEAK_GAP_MIN = 0.002


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-suffix", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--activation-root", type=Path, default=ACT_ROOT)
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--seed", type=int, default=20260712)
    return p.parse_args()


def split_for(group_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"external-head-v1:{group_id}".encode()).digest()[:8], "big") % 10
    return "train" if value < 7 else "val" if value < 8 else "test"


def robust_scale(y: np.ndarray) -> tuple[float, float]:
    median = float(np.nanmedian(y))
    q25, q75 = np.nanpercentile(y, [25, 75])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.nanstd(y))
    return median, max(scale, 1e-8)


def read_record_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_layer_index(index_dir: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray, np.ndarray]:
    shards = pd.read_csv(index_dir / "shards.csv")
    if shards.empty:
        raise RuntimeError(f"No layer shards indexed in {index_dir}")
    layer_names = sorted({name for value in shards.layer_files.fillna("") for name in str(value).split("|") if name})
    if not layer_names:
        raise RuntimeError(f"No layer files indexed in {index_dir}")
    rows: list[dict[str, str]] = []
    shard_dirs: list[str] = []
    shard_sizes: list[int] = []
    for shard in shards.itertuples(index=False):
        record_file = Path(shard.record_ids_file)
        if not record_file.is_absolute():
            record_file = ROOT / record_file
        current = read_record_rows(record_file)
        rows.extend(current)
        pooled_file = Path(shard.pooled_file)
        if not pooled_file.is_absolute():
            pooled_file = ROOT / pooled_file
        shard_dirs.append(str(pooled_file.parent))
        shard_sizes.append(len(current))
    return rows, layer_names, np.asarray(shard_dirs, dtype="U512"), np.asarray(shard_sizes, dtype=int)


def load_layer(layer_name: str, shard_dirs: np.ndarray, shard_sizes: np.ndarray) -> np.ndarray:
    arrays = []
    for directory, expected in zip(shard_dirs, shard_sizes):
        path = Path(str(directory)) / layer_name
        values = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if len(values) != int(expected):
            raise RuntimeError(f"{path}: {len(values)} rows != {expected} IDs")
        arrays.append(values)
    result = np.concatenate(arrays)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid pooled layer matrix {layer_name}: shape={result.shape}")
    return result


def concept_frame(manifest_path: Path, cohort: str, record_names: np.ndarray) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path, low_memory=False)
    frame = frame[frame.cohort.astype(str).str.lower().eq(COHORT_VALUES[cohort])].copy()
    frame["record_id"] = frame.record_id.astype(str)
    frame = frame.drop_duplicates("record_id").set_index("record_id")
    missing = [name for name in record_names if name not in frame.index]
    if missing:
        raise RuntimeError(f"Missing measurements for {len(missing)} records; first={missing[:3]}")
    aligned = frame.loc[record_names].copy()
    for concept in BASE_CONCEPTS:
        aligned[concept] = pd.to_numeric(aligned[concept], errors="coerce")
    rr = aligned["rr_mean_ms"].to_numpy(dtype=float)
    qt = aligned["qt_like_ms"].to_numpy(dtype=float)
    aligned["heart_rate_bpm"] = np.divide(60000.0, rr, out=np.full(len(rr), np.nan), where=rr > 0)
    aligned["qtc_bazett_ms"] = np.divide(qt, np.sqrt(rr / 1000.0), out=np.full(len(rr), np.nan), where=(rr > 0) & np.isfinite(qt))
    return aligned


def main() -> None:
    args = parse_args()
    cohort = args.cohort.lower().replace("-", "_")
    if cohort not in MANIFESTS:
        raise ValueError(f"Unsupported cohort {cohort}")
    index_dir = args.activation_root / args.model_suffix / cohort
    record_rows, layer_names, shard_dirs, shard_sizes = load_layer_index(index_dir)
    record_names = np.asarray([row["record_name"] for row in record_rows], dtype="U64")
    subject_ids = np.asarray([row.get("subject_id", "") or row["record_name"] for row in record_rows], dtype="U64")
    groups = subject_ids if cohort == "mimic_f" else record_names
    splits = np.asarray([split_for(group) for group in groups], dtype="U5")
    concepts = concept_frame(MANIFESTS[cohort], cohort, record_names)
    concept_names = list(BASE_CONCEPTS) + ["heart_rate_bpm", "qtc_bazett_ms"]
    y_matrix = concepts[concept_names].to_numpy(dtype=np.float64)
    train = splits == "train"; val = splits == "val"; test = splits == "test"
    rng = np.random.default_rng(args.seed)
    score_rows: list[dict[str, object]] = []

    for layer_name in layer_names:
        x = load_layer(layer_name, shard_dirs, shard_sizes)
        scaler = StandardScaler().fit(x[train])
        x_train = scaler.transform(x[train]); x_val = scaler.transform(x[val]); x_test = scaler.transform(x[test])
        train_positions = np.flatnonzero(train); val_positions = np.flatnonzero(val); test_positions = np.flatnonzero(test)
        for concept_idx, concept in enumerate(concept_names):
            y = y_matrix[:, concept_idx]
            tr_valid = np.isfinite(y[train]); va_valid = np.isfinite(y[val]); te_valid = np.isfinite(y[test])
            if tr_valid.sum() < 100 or va_valid.sum() < 20 or te_valid.sum() < 20:
                continue
            median, scale = robust_scale(y[train_positions[tr_valid]])
            ytr = (y[train_positions[tr_valid]] - median) / scale
            yva = (y[val_positions[va_valid]] - median) / scale
            yte = (y[test_positions[te_valid]] - median) / scale
            best = None
            for alpha in ALPHAS:
                probe = Ridge(alpha=alpha, solver="lsqr").fit(x_train[tr_valid], ytr)
                pred = probe.predict(x_val[va_valid]); score = float(r2_score(yva, pred))
                if best is None or score > best[0]:
                    best = (score, alpha, probe)
            assert best is not None
            shuffled = ytr.copy(); rng.shuffle(shuffled)
            gaussian = rng.normal(size=len(ytr))
            shuffled_probe = Ridge(alpha=best[1], solver="lsqr").fit(x_train[tr_valid], shuffled)
            gaussian_probe = Ridge(alpha=best[1], solver="lsqr").fit(x_train[tr_valid], gaussian)
            score_rows.append({
                "feature": Path(layer_name).stem,
                "concept_id": concept,
                "family": FAMILIES[concept],
                "alpha": best[1],
                "n_train": int(tr_valid.sum()), "n_val": int(va_valid.sum()), "n_test": int(te_valid.sum()),
                "val_r2": best[0],
                "test_r2": float(r2_score(yte, best[2].predict(x_test[te_valid]))),
                "val_r2_shuffled": float(r2_score(yva, shuffled_probe.predict(x_val[va_valid]))),
                "val_r2_gaussian": float(r2_score(yva, gaussian_probe.predict(x_val[va_valid]))),
            })

    scores = pd.DataFrame(score_rows)
    peaks = []
    for concept, part in scores.groupby("concept_id", sort=False):
        ordered = part.sort_values("val_r2", ascending=False).reset_index(drop=True)
        peak = ordered.iloc[0]; second = float(ordered.iloc[1].val_r2) if len(ordered) > 1 else np.nan
        shuffled_margin = float(peak.val_r2 - peak.val_r2_shuffled)
        gaussian_margin = float(peak.val_r2 - peak.val_r2_gaussian)
        peak_gap = float(peak.val_r2 - second) if np.isfinite(second) else np.nan
        strict = bool(
            peak.val_r2 >= VAL_R2_MIN
            and shuffled_margin >= CONTROL_MARGIN_MIN
            and gaussian_margin >= CONTROL_MARGIN_MIN
            and (not np.isfinite(peak_gap) or peak_gap >= PEAK_GAP_MIN)
        )
        peaks.append({
            "concept_id": concept, "family": peak.family, "peak_feature": peak.feature,
            "peak_val_r2": float(peak.val_r2), "second_best_val_r2": second, "peak_gap": peak_gap,
            "test_r2_at_peak": float(peak.test_r2),
            "val_r2_shuffled_at_peak": float(peak.val_r2_shuffled),
            "val_r2_gaussian_at_peak": float(peak.val_r2_gaussian),
            "shuffled_margin": shuffled_margin, "gaussian_margin": gaussian_margin,
            "strict_encoded": strict,
        })
    peaks_frame = pd.DataFrame(peaks)
    out = args.out_root / args.model_suffix / cohort
    out.mkdir(parents=True, exist_ok=True)
    for path, frame in ((out / "probe_scores.csv", scores), (out / "strict_probe_peaks.csv", peaks_frame)):
        tmp = path.with_suffix(f".csv.tmp.{os.getpid()}"); frame.to_csv(tmp, index=False); tmp.replace(path)
    report = {
        "schema_version": 1, "model_suffix": args.model_suffix, "cohort": cohort,
        "records": len(record_names), "split_unit": "patient" if cohort == "mimic_f" else "record",
        "layers": len(layer_names), "concepts_requested": len(concept_names),
        "concepts_scored": int(peaks_frame.concept_id.nunique()),
        "strict_encoded_count": int(peaks_frame.strict_encoded.sum()),
        "strict_gate": {"val_r2_min": VAL_R2_MIN, "control_margin_min": CONTROL_MARGIN_MIN, "peak_gap_min": PEAK_GAP_MIN},
        "measurement_provenance": "waveform-derived NeuroKit2/global summaries; not vendor-equivalent",
        "diagnostic_labels_used_as_concepts": False,
    }
    tmp = out / f"probe_report.json.tmp.{os.getpid()}"; tmp.write_text(json.dumps(report, indent=2) + "\n"); tmp.replace(out / "probe_report.json")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
