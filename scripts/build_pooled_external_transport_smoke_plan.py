#!/usr/bin/env python
"""Create a deterministic six-model pooled-activation transport smoke plan."""
from __future__ import annotations

import csv
import hashlib
import heapq
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "results/multicohort/track_f_full/waveform_concepts_by_record.csv"
OUT = ROOT / "results/activations_external_pooled_smoke"
PLAN = OUT / "plan"
COHORTS = ("chapman", "cpsc", "ningbo", "mimic")
MODELS = (
    "csfm",
    "cardiac_fm",
    "ecg_fm",
    "ecg_jepa",
    "hubert_ecg",
    "st_mem",
)
SAMPLE_SIZE = 512
BATCH_SIZE = 32


def score(cohort: str, record_id: str) -> int:
    digest = hashlib.sha256(f"pooled-transport-v1:{cohort}:{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> None:
    from scripts.plan_external_activation_extraction import command, index_command

    PLAN.mkdir(parents=True, exist_ok=True)
    heaps: dict[str, list[tuple[int, int, dict[str, str]]]] = {cohort: [] for cohort in COHORTS}
    counter = 0
    with SOURCE.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            cohort = row.get("cohort", "")
            if cohort not in heaps or row.get("status") != "ok":
                continue
            value = score(cohort, row.get("record_id", ""))
            item = (-value, counter, row)
            counter += 1
            heap = heaps[cohort]
            if len(heap) < SAMPLE_SIZE:
                heapq.heappush(heap, item)
            elif value < -heap[0][0]:
                heapq.heapreplace(heap, item)
    sampled = []
    for cohort in COHORTS:
        if len(heaps[cohort]) != SAMPLE_SIZE:
            raise RuntimeError(f"{cohort}: expected {SAMPLE_SIZE} records, got {len(heaps[cohort])}")
        rows = [item[2] for item in sorted(heaps[cohort], key=lambda item: -item[0])]
        sampled.extend(rows)
    manifest = PLAN / "sample_manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)

    commands = []
    indexes = []
    summary = []
    for model in MODELS:
        for cohort in COHORTS:
            for offset in range(0, SAMPLE_SIZE, BATCH_SIZE):
                commands.append(
                    command(
                        model=model,
                        cohort=cohort,
                        offset=offset,
                        limit=min(BATCH_SIZE, SAMPLE_SIZE - offset),
                        layers="0",
                        activation_out_dir=OUT,
                        manifest=manifest,
                        device="cuda",
                    )
                )
            indexes.append(index_command(model, cohort, OUT))
            summary.append(
                {
                    "model": model,
                    "cohort": f"{cohort}_f",
                    "records": SAMPLE_SIZE,
                    "batch_size": BATCH_SIZE,
                    "shards": SAMPLE_SIZE // BATCH_SIZE,
                    "layers": "0",
                }
            )
    (PLAN / "all_commands.txt").write_text("\n".join(commands) + "\n")
    (PLAN / "index_commands.txt").write_text("\n".join(indexes) + "\n")
    with (PLAN / "plan_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# Pooled External SAE Transport Smoke Plan",
        "",
        f"- Cohorts: {', '.join(COHORTS)}",
        f"- Models: {', '.join(MODELS)}",
        f"- Deterministic records per cohort: {SAMPLE_SIZE}",
        f"- Batch size: {BATCH_SIZE}",
        f"- GPU shard commands: {len(commands)}",
        f"- Index commands: {len(indexes)}",
        "- Pooled activations are primary; layer 0 is saved only as an extraction integrity check.",
        "- CSFM/ST-MEM use digital-signal preprocessing; other models use their anchor z-score preprocessing.",
    ]
    (PLAN / "plan_report.md").write_text("\n".join(lines) + "\n")
    print(f"commands={len(commands)} indexes={len(indexes)} sampled={len(sampled)}")


if __name__ == "__main__":
    main()
