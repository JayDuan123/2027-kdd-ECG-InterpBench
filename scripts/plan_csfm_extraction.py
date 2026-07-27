#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark_v1.adapters.ecg_jepa import DEFAULT_SPLIT_CSV
from benchmark_v1.config import ROOT


DEFAULT_OUT_DIR = ROOT / "results" / "activations" / "csfm_commons_plan"
CSFM_PYTHON = "/rhf/allocations/wq8/yd68/venvs/csfm_cu118/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write CSFM activation extraction shard commands.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--activation-out-dir", default="results/activations/csfm_complete_commons")
    return parser.parse_args()


def split_counts() -> dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with DEFAULT_SPLIT_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            split = row.get("split")
            if split in counts:
                counts[split] += 1
    return counts


def command(split: str, offset: int, limit: int, layers: str, activation_out_dir: str, device: str) -> str:
    shard = f"{split}_offset{offset:06d}_n{limit:04d}"
    return (
        "MPLCONFIGDIR=/tmp/mplconfig-benchmark "
        f"{CSFM_PYTHON} scripts/extract_csfm_activations.py "
        f"--split {split} --offset {offset} --limit {limit} "
        f"--save-activations --layers {layers} "
        f"--device {device} "
        f"--out-dir {activation_out_dir} --shard-name {shard}"
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    counts = split_counts()
    target_splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    lines: list[str] = []
    report = ["# CSFM Activation Extraction Plan", ""]
    report.append(f"- batch size: {args.batch_size}")
    report.append(f"- layers: {args.layers}")
    report.append(f"- device: {args.device}")
    report.append(f"- activation out dir: {args.activation_out_dir}")
    report.append("")
    for split in target_splits:
        total = counts[split]
        n_shards = math.ceil(total / args.batch_size) if total else 0
        report.append(f"## {split}")
        report.append("")
        report.append(f"- records: {total}")
        report.append(f"- shards: {n_shards}")
        report.append("")
        for shard_idx in range(n_shards):
            offset = shard_idx * args.batch_size
            limit = min(args.batch_size, total - offset)
            lines.append(command(split, offset, limit, args.layers, args.activation_out_dir, args.device))
    commands_path = args.out_dir / f"{args.split}_commands.txt"
    report_path = args.out_dir / f"{args.split}_plan.md"
    commands_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"wrote {commands_path}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
