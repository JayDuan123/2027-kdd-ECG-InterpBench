#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUT_DIR = ROOT / "results" / "multicohort" / "external_sae"
SAE_CELLS = ROOT / "results" / "sae_extension" / "six_model_sae_audit" / "l0clamp_reclassified" / "sae_l0clamp_reclassified_cells.csv"


EXTERNAL_COHORTS = ["MIMIC-F", "Chapman-F", "CPSC-F", "Ningbo-F"]


MODEL_SUFFIX = {
    "CARDIAC-FM": "cardiac_fm_cu118_commons",
    "CSFM": "csfm_cu118_commons",
    "ECG-FM": "ecg_fm_cu118_commons",
    "ECG-JEPA": "ecg_jepa_cu118_commons",
    "HuBERT-ECG": "hubert_ecg_cu118_commons",
    "ST-MEM": "st_mem_cu118_commons",
}


RECON_FIELDS = [
    "external_cohort",
    "model",
    "concept",
    "task",
    "layer",
    "analysis",
    "ptbxl_sae_status",
    "sae_checkpoint",
    "external_activation_status",
    "external_activation_path",
    "recon_gate_status",
    "recon_gate_pass",
    "external_recon_r2",
    "required_recon_r2_floor",
    "n_external_activation_rows",
    "reason",
]


STEERING_FIELDS = [
    "external_cohort",
    "model",
    "concept",
    "task",
    "layer",
    "status",
    "recon_gate_status",
    "steering_claim_allowed",
    "reason",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def find_checkpoint(cell: dict[str, str]) -> tuple[str, str]:
    result_csv = cell.get("result_csv", "")
    if not result_csv:
        return "missing_result_csv", ""
    result_path = ROOT / result_csv if not result_csv.startswith("/") else Path(result_csv)
    checkpoint_dir = result_path.parent / "checkpoints"
    if not checkpoint_dir.exists():
        return "missing_checkpoint_dir", ""
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return "missing_checkpoint", ""
    return "available", str(checkpoints[0])


def activation_candidates(model: str, cohort: str) -> list[Path]:
    suffix = MODEL_SUFFIX.get(model, "")
    cohort_slug = cohort.lower().replace("-", "_")
    return [
        ROOT / "results" / "activations_external_v2" / suffix / cohort_slug,
        ROOT / "results" / "activations_external" / suffix / cohort_slug,
        ROOT / "results" / "activations_external" / f"{suffix}_{cohort_slug}",
        ROOT / "results" / "activations" / f"{suffix}_{cohort_slug}",
        ROOT / "results" / "activations" / suffix / cohort_slug,
    ]


def find_external_activation(model: str, cohort: str) -> tuple[str, str, str]:
    for path in activation_candidates(model, cohort):
        if not path.exists():
            continue
        records = path / "records.csv"
        if records.exists():
            n_rows = max(sum(1 for _ in records.open()) - 1, 0)
            return "available", str(path), str(n_rows)
        npy_files = list(path.rglob("*.npy"))
        if npy_files:
            return "available_no_record_index", str(path), ""
        return "present_but_no_activation_files", str(path), ""
    return "missing_external_activation_cache", "", "0"


def load_sae(checkpoint: Path, device: str = "cpu"):
    import torch

    from benchmark_v1.sae_extension.topk_sae import TopKSAE

    saved = torch.load(checkpoint, map_location=device)
    meta = saved["meta"]
    sae = TopKSAE(
        d=int(meta["d"]),
        n_features=int(meta["n_features"]),
        k=int(meta["k"]),
    ).to(device)
    sae.load_state_dict(saved["sae"])
    sae.eval()
    return sae


def read_index_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def shard_dir_from_row(row: dict[str, str]) -> Path:
    metadata = row.get("activation_metadata", "")
    if metadata:
        return ROOT / Path(metadata).parent
    return ROOT / row["shard_name"]


def load_external_layer_acts(index_dir: Path, layer: int, max_shards: int = 0):
    import numpy as np
    import torch

    shard_rows = read_index_rows(index_dir / "shards.csv")
    if max_shards > 0:
        shard_rows = shard_rows[:max_shards]
    arrays = []
    n_records = 0
    for shard in shard_rows:
        shard_dir = ROOT / Path(shard["activation_metadata"]).parent
        layer_path = shard_dir / f"layer_{int(layer):02d}.npy"
        if not layer_path.exists():
            continue
        x = np.asarray(np.load(layer_path, mmap_mode="r"), dtype=np.float32)
        if x.ndim == 3:
            x = x.mean(axis=1)
        elif x.ndim != 2:
            raise ValueError(f"unsupported activation shape {x.shape} in {layer_path}")
        arrays.append(x)
        n_records += int(x.shape[0])
    if not arrays:
        raise FileNotFoundError(f"no layer_{int(layer):02d}.npy activations found under {index_dir}")
    return torch.as_tensor(np.concatenate(arrays, axis=0), dtype=torch.float32), n_records


def evaluate_external_recon(
    checkpoint: str,
    activation_path: str,
    layer: str,
    device: str = "cpu",
    max_shards: int = 0,
    sae_cache: dict[str, object] | None = None,
    acts_cache: dict[tuple[str, int, int], tuple[object, int]] | None = None,
) -> tuple[str, str, str]:
    from benchmark_v1.sae_extension.train_sae import _batched_recon_r2

    if sae_cache is not None and checkpoint in sae_cache:
        sae = sae_cache[checkpoint]
    else:
        sae = load_sae(Path(checkpoint), device=device)
        if sae_cache is not None:
            sae_cache[checkpoint] = sae

    layer_int = int(float(layer))
    acts_key = (activation_path, layer_int, int(max_shards))
    if acts_cache is not None and acts_key in acts_cache:
        acts, n_records = acts_cache[acts_key]
    else:
        acts, n_records = load_external_layer_acts(Path(activation_path), layer_int, max_shards=max_shards)
        if acts_cache is not None:
            acts_cache[acts_key] = (acts, n_records)
    r2 = _batched_recon_r2(sae, acts)
    return f"{r2:.8g}", str(n_records), "ok"


def eligible_cells(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    seen = set()
    for row in rows:
        if row.get("analysis") != "main_recon_0.90":
            continue
        if row.get("matched_tier") != "in_band":
            continue
        if row.get("steering_status") != "completed":
            continue
        key = (row.get("model"), row.get("concept"), row.get("task"), row.get("layer"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def build_rows(evaluate_recon: bool = True, device: str = "cpu", max_shards: int = 0) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cells = eligible_cells(read_csv(SAE_CELLS))
    recon_rows: list[dict[str, str]] = []
    steering_rows: list[dict[str, str]] = []
    sae_cache: dict[str, object] = {}
    acts_cache: dict[tuple[str, int, int], tuple[object, int]] = {}
    for cell in cells:
        checkpoint_status, checkpoint = find_checkpoint(cell)
        for cohort in EXTERNAL_COHORTS:
            activation_status, activation_path, n_rows = find_external_activation(cell["model"], cohort)
            r2 = ""
            if checkpoint_status != "available":
                recon_status = "fail_missing_ptbxl_sae_checkpoint"
                pass_flag = "false"
                reason = checkpoint_status
            elif activation_status != "available":
                recon_status = "fail_missing_external_activation_cache"
                pass_flag = "false"
                reason = activation_status
            else:
                if evaluate_recon:
                    try:
                        r2, n_eval, eval_reason = evaluate_external_recon(
                            checkpoint=checkpoint,
                            activation_path=activation_path,
                            layer=cell["layer"],
                            device=device,
                            max_shards=max_shards,
                            sae_cache=sae_cache,
                            acts_cache=acts_cache,
                        )
                        n_rows = n_eval
                        r2_float = float(r2)
                        if not math.isfinite(r2_float):
                            recon_status = "fail_external_recon_nonfinite"
                            pass_flag = "false"
                            reason = "nonfinite external reconstruction R2"
                        elif r2_float >= 0.90:
                            recon_status = "pass_external_recon_gate"
                            pass_flag = "true"
                            reason = eval_reason
                        else:
                            recon_status = "fail_external_recon_below_floor"
                            pass_flag = "false"
                            reason = eval_reason
                    except Exception as exc:
                        r2 = ""
                        recon_status = "fail_recon_eval_error"
                        pass_flag = "false"
                        reason = f"{type(exc).__name__}: {exc}"
                else:
                    r2 = ""
                    recon_status = "pending_recon_eval"
                    pass_flag = "false"
                    reason = "external activation cache exists, but reconstruction evaluation was disabled"
            recon_rows.append(
                {
                    "external_cohort": cohort,
                    "model": cell["model"],
                    "concept": cell["concept"],
                    "task": cell["task"],
                    "layer": cell["layer"],
                    "analysis": cell["analysis"],
                    "ptbxl_sae_status": checkpoint_status,
                    "sae_checkpoint": checkpoint,
                    "external_activation_status": activation_status,
                    "external_activation_path": activation_path,
                    "recon_gate_status": recon_status,
                    "recon_gate_pass": pass_flag,
                    "external_recon_r2": r2,
                    "required_recon_r2_floor": "0.90",
                    "n_external_activation_rows": n_rows,
                    "reason": reason,
                }
            )
            steering_rows.append(
                {
                    "external_cohort": cohort,
                    "model": cell["model"],
                    "concept": cell["concept"],
                    "task": cell["task"],
                    "layer": cell["layer"],
                    "status": "skipped_no_recon_gate_pass",
                    "recon_gate_status": recon_status,
                    "steering_claim_allowed": "false",
                    "reason": "External SAE steering is disallowed unless external SAE reconstruction gate passes.",
                }
            )
    return recon_rows, steering_rows


def render_report(recon_rows: list[dict[str, str]], steering_rows: list[dict[str, str]]) -> str:
    total = len(recon_rows)
    pass_count = sum(row["recon_gate_pass"] == "true" for row in recon_rows)
    activation_missing = sum(row["external_activation_status"] == "missing_external_activation_cache" for row in recon_rows)
    checkpoint_missing = sum(row["ptbxl_sae_status"] != "available" for row in recon_rows)
    eval_error = sum(row["recon_gate_status"] == "fail_recon_eval_error" for row in recon_rows)
    nonfinite = sum(row["recon_gate_status"] == "fail_external_recon_nonfinite" for row in recon_rows)
    below_floor = sum(row["recon_gate_status"] == "fail_external_recon_below_floor" for row in recon_rows)
    pending = sum(row["recon_gate_status"] == "pending_recon_eval" for row in recon_rows)
    by_cohort = {}
    for row in recon_rows:
        key = row["external_cohort"]
        stats = by_cohort.setdefault(key, {"rows": 0, "pass": 0, "missing_activation": 0, "nonfinite": 0})
        stats["rows"] += 1
        stats["pass"] += int(row["recon_gate_pass"] == "true")
        stats["missing_activation"] += int(row["external_activation_status"] == "missing_external_activation_cache")
        stats["nonfinite"] += int(row["recon_gate_status"] == "fail_external_recon_nonfinite")
    lines = [
        "# External SAE Reconstruction Gate",
        "",
        "This gate enforces the multi-cohort SAE rule: external steering may be interpreted only after SAE reconstruction fidelity passes on the external activation distribution.",
        "",
        f"- eligible PTB-XL SAE cells expanded over external cohorts: {total}",
        f"- recon-gate passes: {pass_count}",
        f"- rows missing external activation cache: {activation_missing}",
        f"- rows missing PTB-XL SAE checkpoint: {checkpoint_missing}",
        f"- rows below reconstruction floor: {below_floor}",
        f"- rows with nonfinite external reconstruction R2: {nonfinite}",
        f"- rows with reconstruction-eval errors: {eval_error}",
        f"- rows pending reconstruction eval: {pending}",
        "",
        "| External cohort | Rows | Recon passes | Missing external activation cache | Nonfinite recon R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cohort, stats in sorted(by_cohort.items()):
        lines.append(
            f"| {cohort} | {stats['rows']} | {stats['pass']} | "
            f"{stats['missing_activation']} | {stats['nonfinite']} |"
        )
    lines.extend(
        [
            "",
            "## Result",
            "",
            "External SAE steering claims are allowed only for rows with `recon_gate_pass=true`.",
            "",
            "Rows failing because of missing activation cache or reconstruction-eval errors are gate/infrastructure failures, not negative steering results.",
            "",
            "## Steering Audit",
            "",
            f"- steering rows emitted: {len(steering_rows)}",
            "- all rows are marked `skipped_no_recon_gate_pass`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external SAE reconstruction gate and steering-skip audit.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-evaluate-recon", action="store_true", help="Only check cache availability; do not compute external SAE reconstruction R2.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-shards-per-row", type=int, default=0)
    args = parser.parse_args()
    recon_rows, steering_rows = build_rows(
        evaluate_recon=not args.no_evaluate_recon,
        device=args.device,
        max_shards=args.max_shards_per_row,
    )
    write_csv_atomic(args.out_dir / "external_sae_recon_gate.csv", recon_rows, RECON_FIELDS)
    write_csv_atomic(args.out_dir / "external_sae_steering_audit.csv", steering_rows, STEERING_FIELDS)
    report = render_report(recon_rows, steering_rows)
    report_path = args.out_dir / "external_sae_recon_gate_report.md"
    tmp = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp.write_text(report)
    os.replace(tmp, report_path)
    print(f"wrote: {args.out_dir / 'external_sae_recon_gate.csv'}")
    print(f"wrote: {args.out_dir / 'external_sae_steering_audit.csv'}")
    print(f"wrote: {report_path}")
    print(f"recon_rows={len(recon_rows)} recon_passes={sum(row['recon_gate_pass'] == 'true' for row in recon_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
