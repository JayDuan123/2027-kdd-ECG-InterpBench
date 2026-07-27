#!/usr/bin/env python
"""Build checkpointable full-cohort pooled and sampled-layer extraction plans."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "results/multicohort/track_f_full/waveform_concepts_by_record.csv"
DEFAULT_OUT = ROOT / "results/activations_external_full_v1"
MODELS = ("csfm", "cardiac_fm", "ecg_fm", "ecg_jepa", "hubert_ecg", "st_mem")


def score(cohort: str, record_id: str) -> int:
    digest = hashlib.sha256(f"external-layer-atlas-v1:{cohort}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cohorts", default="chapman,cpsc")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--pooled-batch", type=int, default=128)
    p.add_argument("--layer-batch", type=int, default=16)
    p.add_argument("--layer-sample", type=int, default=4096)
    return p.parse_args()


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    from scripts.plan_external_activation_extraction import command, index_command

    a = parse_args()
    cohorts = tuple(item.strip() for item in a.cohorts.split(",") if item.strip())
    plan = a.out / ("plan_" + "_".join(cohorts))
    plan.mkdir(parents=True, exist_ok=True)
    rows_by_cohort = {cohort: [] for cohort in cohorts}
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            cohort = row.get("cohort", "")
            if cohort in rows_by_cohort and row.get("status") == "ok":
                rows_by_cohort[cohort].append(row)

    full_rows = [row for cohort in cohorts for row in rows_by_cohort[cohort]]
    full_manifest = plan / "full_manifest.csv"
    write_manifest(full_manifest, fieldnames, full_rows)
    sampled_rows = []
    for cohort in cohorts:
        ranked = sorted(rows_by_cohort[cohort], key=lambda r: score(cohort, r.get("record_id", "")))
        sampled_rows.extend(ranked[: min(a.layer_sample, len(ranked))])
    layer_manifest = plan / "layer_sample_manifest.csv"
    write_manifest(layer_manifest, fieldnames, sampled_rows)

    pooled_root = a.out / "pooled"
    layer_root = a.out / "layer_atlas"
    pooled_commands = []
    layer_commands = []
    pooled_indexes = []
    layer_indexes = []
    summary = []
    for model in MODELS:
        for cohort in cohorts:
            total = len(rows_by_cohort[cohort])
            layer_total = min(a.layer_sample, total)
            model_pooled = []
            for offset in range(0, total, a.pooled_batch):
                model_pooled.append(command(model, cohort, offset, min(a.pooled_batch, total - offset), "pooled", pooled_root, full_manifest, "cuda"))
            model_layers = []
            for offset in range(0, layer_total, a.layer_batch):
                cmd = command(model, cohort, offset, min(a.layer_batch, layer_total - offset), "all", layer_root, layer_manifest, "cuda")
                model_layers.append(cmd + " --pool-layer-activations")
            pooled_commands.extend(model_pooled)
            layer_commands.extend(model_layers)
            pooled_indexes.append(index_command(model, cohort, pooled_root))
            layer_indexes.append(index_command(model, cohort, layer_root))
            summary.append({
                "model": model, "cohort": cohort, "records": total,
                "pooled_batch": a.pooled_batch, "pooled_shards": math.ceil(total / a.pooled_batch),
                "layer_records": layer_total, "layer_batch": a.layer_batch,
                "layer_shards": math.ceil(layer_total / a.layer_batch),
            })

    def write_lines(name: str, lines: list[str]) -> None:
        (plan / name).write_text("\n".join(lines) + ("\n" if lines else ""))

    write_lines("pooled_commands.txt", pooled_commands)
    write_lines("layer_commands.txt", layer_commands)
    write_lines("pooled_index_commands.txt", pooled_indexes)
    write_lines("layer_index_commands.txt", layer_indexes)
    # One first shard per model/cohort validates the production batch size.
    smoke = []
    cursor = 0
    for model in MODELS:
        for cohort in cohorts:
            smoke.append(pooled_commands[cursor])
            cursor += math.ceil(len(rows_by_cohort[cohort]) / a.pooled_batch)
    write_lines("pooled_smoke_commands.txt", smoke)
    with (plan / "plan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    (plan / "plan_report.md").write_text(
        "# Full External Benchmark Plan\n\n"
        f"- Cohorts: {', '.join(cohorts)}\n"
        f"- Full pooled commands: {len(pooled_commands)}\n"
        f"- Sampled all-layer commands: {len(layer_commands)}\n"
        f"- Layer sample per cohort: {a.layer_sample}\n"
        "- Full token tensors are not persisted; layer arrays use token-mean pooling.\n"
        "- Every shard is restartable and completion is keyed by activation_metadata.json.\n"
    )
    print(f"plan={plan} pooled={len(pooled_commands)} layers={len(layer_commands)} smoke={len(smoke)}")


if __name__ == "__main__":
    main()
