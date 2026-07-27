#!/usr/bin/env python
"""Run every joint-steering group/seed cell with bounded CPU parallelism."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "results/sae_reconciliation/steering_benchmark_multimodel_v2_expanded"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_joint_steering_manifest.py"),
            "--base",
            str(args.base),
        ],
        check=True,
    )
    groups = pd.read_csv(args.base / "joint_steering/joint_steering_manifest.csv")
    tasks = [(int(index), seed) for index in groups.group_index for seed in (4311, 4312, 4313)]

    def run(task: tuple[int, int]) -> tuple[int, int]:
        group_index, seed = task
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_joint_steering_task.py"),
                "--base",
                str(args.base),
                "--group-index",
                str(group_index),
                "--seed",
                str(seed),
                "--n-random",
                "20",
            ],
            check=True,
        )
        return task

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            group_index, seed = future.result()
            print(f"completed group={group_index} seed={seed}", flush=True)
    print(f"joint steering tasks completed={len(tasks)}", flush=True)


if __name__ == "__main__":
    main()
