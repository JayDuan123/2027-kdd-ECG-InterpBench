#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.config import ROOT  # noqa: E402
from scripts.extract_external_model_activations import (  # noqa: E402
    COHORT_ALIASES,
    MODEL_ALIASES,
    MODEL_SUFFIX,
    canonical_cohort,
    canonical_model,
)


DEFAULT_GATE = ROOT / "results" / "multicohort" / "external_sae" / "external_sae_recon_gate.csv"
DEFAULT_MANIFEST = ROOT / "results" / "multicohort" / "track_f_full" / "waveform_concepts_by_record.csv"
DEFAULT_OUT_DIR = ROOT / "results" / "activations_external" / "external_sae_plan"
DEFAULT_ACTIVATION_OUT_DIR = ROOT / "results" / "activations_external"
PYTHON = "/rhf/allocations/wq8/yd68/venvs/ecg_fm_cu118/bin/python"
SITE_PACKAGES = {
    "csfm": "/rhf/allocations/wq8/yd68/venvs/csfm_cu118/lib/python3.10/site-packages",
    "ecg_jepa": "/rhf/allocations/wq8/yd68/venvs/ecg_jepa_cu118/lib/python3.10/site-packages",
    "st_mem": "/rhf/allocations/wq8/yd68/venvs/st_mem_cu118/lib/python3.10/site-packages",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def needed_layers(gate_rows: list[dict[str, str]]) -> dict[tuple[str, str], set[int]]:
    out: dict[tuple[str, str], set[int]] = {}
    for row in gate_rows:
        model = canonical_model(row["model"])
        cohort = canonical_cohort(row["external_cohort"])
        if row.get("ptbxl_sae_status") != "available":
            continue
        try:
            layer = int(row["layer"])
        except ValueError:
            continue
        out.setdefault((model, cohort), set()).add(layer)
    return out


def available_counts(manifest: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            cohort = canonical_cohort(row.get("cohort", ""))
            counts[cohort] = counts.get(cohort, 0) + 1
    return counts


def command(
    model: str,
    cohort: str,
    offset: int,
    limit: int,
    layers: str,
    activation_out_dir: Path,
    manifest: Path,
    device: str,
) -> str:
    suffix = MODEL_SUFFIX[model]
    shard = f"{cohort}_f_offset{offset:06d}_n{limit:04d}"
    env_prefix = "MPLCONFIGDIR=/tmp/mplconfig-benchmark "
    pythonpath_parts = []
    if model in SITE_PACKAGES:
        pythonpath_parts.append(SITE_PACKAGES[model])
    if model in {"ecg_fm", "cardiac_fm"}:
        pythonpath_parts.append("/rhf/allocations/wq8/yd68/fairseq-signals")
    if pythonpath_parts:
        env_prefix += f"PYTHONPATH={':'.join(pythonpath_parts)}:${{PYTHONPATH:-}} "
    return (
        env_prefix
        + f"{PYTHON} scripts/extract_external_model_activations.py "
        f"--model {suffix} --cohort {cohort}_f "
        f"--offset {offset} --limit {limit} "
        f"--manifest {manifest} "
        f"--out-dir {activation_out_dir} "
        f"--shard-name {shard} "
        f"--save-activations --layers {layers} --device {device}"
    )


def index_command(model: str, cohort: str, activation_out_dir: Path) -> str:
    suffix = MODEL_SUFFIX[model]
    index_dir = activation_out_dir / suffix / f"{cohort}_f"
    return (
        f"{PYTHON} scripts/build_activation_index.py "
        f"--model {suffix} --activation-dir {index_dir} --out-dir {index_dir}"
    )


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan external activation extraction only for SAE-gate eligible model/cohort/layer cells.")
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--activation-out-dir", type=Path, default=DEFAULT_ACTIVATION_OUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", default="", help="Optional comma list using gate model names or suffixes.")
    parser.add_argument("--cohorts", default="", help="Optional comma list, e.g. MIMIC-F,Chapman-F.")
    parser.add_argument("--max-records-per-cohort", type=int, default=0, help="Optional cap for smoke planning.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_filter = {canonical_model(item) for item in parse_csv_set(args.models)} if args.models else set()
    cohort_filter = {canonical_cohort(item) for item in parse_csv_set(args.cohorts)} if args.cohorts else set()

    gate_rows = read_csv(args.gate)
    layer_map = needed_layers(gate_rows)
    counts = available_counts(args.manifest)

    extraction_commands: list[str] = []
    index_commands: list[str] = []
    summary_rows: list[dict[str, str]] = []
    for (model, cohort), layers in sorted(layer_map.items()):
        if model_filter and model not in model_filter:
            continue
        if cohort_filter and cohort not in cohort_filter:
            continue
        total = counts.get(cohort, 0)
        if args.max_records_per_cohort > 0:
            total = min(total, args.max_records_per_cohort)
        layers_spec = ",".join(str(layer) for layer in sorted(layers))
        n_shards = math.ceil(total / args.batch_size) if total else 0
        for shard_idx in range(n_shards):
            offset = shard_idx * args.batch_size
            limit = min(args.batch_size, total - offset)
            extraction_commands.append(
                command(
                    model=model,
                    cohort=cohort,
                    offset=offset,
                    limit=limit,
                    layers=layers_spec,
                    activation_out_dir=args.activation_out_dir,
                    manifest=args.manifest,
                    device=args.device,
                )
            )
        if n_shards:
            index_commands.append(index_command(model, cohort, args.activation_out_dir))
        summary_rows.append(
            {
                "model": model,
                "model_suffix": MODEL_SUFFIX[model],
                "cohort": f"{cohort}_f",
                "records": str(total),
                "batch_size": str(args.batch_size),
                "shards": str(n_shards),
                "layers": layers_spec,
                "activation_dir": str(args.activation_out_dir / MODEL_SUFFIX[model] / f"{cohort}_f"),
            }
        )

    commands_path = args.out_dir / "all_commands.txt"
    index_path = args.out_dir / "index_commands.txt"
    summary_path = args.out_dir / "plan_summary.csv"
    report_path = args.out_dir / "plan_report.md"
    commands_path.write_text("\n".join(extraction_commands) + ("\n" if extraction_commands else ""), encoding="utf-8")
    index_path.write_text("\n".join(index_commands) + ("\n" if index_commands else ""), encoding="utf-8")
    with summary_path.open("w", newline="") as f:
        fields = ["model", "model_suffix", "cohort", "records", "batch_size", "shards", "layers", "activation_dir"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# External Activation Extraction Plan",
        "",
        "This plan is derived from `external_sae_recon_gate.csv`; it extracts only the model/cohort/layer combinations needed before rerunning the external SAE reconstruction gate.",
        "",
        f"- gate: `{args.gate}`",
        f"- manifest: `{args.manifest}`",
        f"- batch size: {args.batch_size}",
        f"- device: {args.device}",
        f"- extraction commands: {len(extraction_commands)}",
        f"- index commands: {len(index_commands)}",
        f"- max records per cohort: {args.max_records_per_cohort if args.max_records_per_cohort else 'none'}",
        "",
        "| Model | Cohort | Records | Shards | Layers |",
        "|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(f"| {row['model_suffix']} | {row['cohort']} | {row['records']} | {row['shards']} | {row['layers']} |")
    lines.extend(
        [
            "",
            "After extraction finishes, run every line in `index_commands.txt`, then rerun:",
            "",
            "```bash",
            "python scripts/build_external_sae_recon_gate.py",
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {commands_path}")
    print(f"wrote {index_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {report_path}")
    print(f"commands={len(extraction_commands)} index_commands={len(index_commands)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
