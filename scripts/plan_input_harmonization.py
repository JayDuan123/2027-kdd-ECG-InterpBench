#!/usr/bin/env python3
"""Build reproducible smoke or full extraction command manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV  # noqa: E402
from benchmark_v1.input_harmonization import MODEL_INTERFACES, PROTOCOLS, canonical_model_name  # noqa: E402


PYTHON = {
    "cardiac_fm": ROOT.parent / "venvs/ecg_fm_cu118/bin/python",
    "csfm": ROOT.parent / "venvs/csfm_cu118/bin/python",
    "ecg_fm": ROOT.parent / "venvs/ecg_fm_cu118/bin/python",
    "ecg_jepa": ROOT.parent / "venvs/ecg_jepa_cu118/bin/python",
    "hubert_ecg": ROOT.parent / "venvs/ecg_fm_cu118/bin/python",
    "st_mem": ROOT.parent / "venvs/st_mem_cu118/bin/python",
}
CHUNK = {
    "cardiac_fm": 32,
    "csfm": 16,
    "ecg_fm": 32,
    "ecg_jepa": 32,
    "hubert_ecg": 16,
    "st_mem": 16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_INTERFACES))
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def split_counts(path: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", "")
            if split in counts:
                counts[split] += 1
    return counts


def command(model: str, protocol: str, split: str, offset: int, limit: int, smoke: bool) -> str:
    output_root = ROOT / "results/input_harmonization_v1"
    if smoke:
        output_root /= "smoke"
    output = output_root / "activations" / protocol / model
    shard = f"{split}_offset{offset:06d}_n{limit:04d}"
    words = [
        "env",
        "MPLCONFIGDIR=/tmp/mplconfig-input-harmonization",
    ]
    if model in {"cardiac_fm", "ecg_fm"}:
        words.append(f"PYTHONPATH={ROOT.parent / 'fairseq-signals'}")
    words.extend(
        [
            str(PYTHON[model]),
            "scripts/extract_harmonized_ptbxl_activations.py",
            "--model",
            model,
            "--protocol",
            protocol,
            "--split",
            split,
            "--offset",
            str(offset),
            "--limit",
            str(limit),
            "--split-csv",
            str(DEFAULT_SPLIT_CSV),
            "--out-dir",
            str(output),
            "--shard-name",
            shard,
            "--device",
            "cuda",
        ]
    )
    return shlex.join(words)


def main() -> None:
    args = parse_args()
    counts = split_counts(args.split_csv)
    rows: list[dict[str, object]] = []
    if args.mode == "smoke":
        models = sorted(MODEL_INTERFACES)
        for model in models:
            for protocol in PROTOCOLS:
                rows.append(
                    {
                        "task_index": len(rows),
                        "model": model,
                        "protocol": protocol,
                        "split": "test",
                        "offset": 0,
                        "limit": 2,
                        "command": command(model, protocol, "test", 0, 2, True),
                    }
                )
    else:
        if not args.model:
            raise ValueError("--model is required in full mode")
        model = canonical_model_name(args.model)
        chunk = CHUNK[model]
        for protocol in PROTOCOLS[1:]:
            for split in ("train", "val", "test"):
                for offset in range(0, counts[split], chunk):
                    limit = min(chunk, counts[split] - offset)
                    rows.append(
                        {
                            "task_index": len(rows),
                            "model": model,
                            "protocol": protocol,
                            "split": split,
                            "offset": offset,
                            "limit": limit,
                            "command": command(model, protocol, split, offset, limit, False),
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    command_path = args.output.with_suffix(".commands.txt")
    command_path.write_text("\n".join(str(row["command"]) for row in rows) + "\n", encoding="utf-8")
    report = {
        "mode": args.mode,
        "model": args.model or "all",
        "tasks": len(rows),
        "split_counts": counts,
        "manifest": str(args.output),
        "commands": str(command_path),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
