#!/usr/bin/env python
"""Build the recon-matched top-k SAE group-clamp task manifest.

The primary manifest contains only mathematically available groups. Requested
groups larger than the selected SAE dictionary are retained in a separate
coverage table as ``k_unavailable`` instead of being silently truncated.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_k_grid(spec: str) -> list[int]:
    values = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if not values or values[0] <= 0:
        raise ValueError("--k-grid must contain positive integers")
    return values


def candidate_key(candidate: str) -> tuple[str, str, int]:
    concept, rest = candidate.split("->", 1)
    task, layer = rest.rsplit("@L", 1)
    return concept, task, int(layer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected",
        type=Path,
        default=Path(
            "results/sae_extension/six_model_sae_audit/phase0_selected_operating_points.csv"
        ),
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path(
            "results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv"
        ),
    )
    parser.add_argument("--k-grid", default="1,2,5,10")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/sae_extension/six_model_sae_audit/topk_group_steering_v2"
        ),
    )
    args = parser.parse_args()

    selected = pd.read_csv(args.selected)
    cells = pd.read_csv(args.cells)
    k_grid = parse_k_grid(args.k_grid)
    lookup: dict[tuple[str, str, str, int], int] = {}
    for cell_index, row in cells.iterrows():
        concept, task, layer = candidate_key(str(row["candidate"]))
        lookup[(str(row["model"]), concept, task, layer)] = int(cell_index)

    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        model = str(row["model"])
        concept = str(row["concept"])
        task = str(row["task"])
        layer = int(row["layer"])
        key = (model, concept, task, layer)
        if key not in lookup:
            raise KeyError(f"selected operating point is not in low-coupling cells: {key}")
        n_capacity = int(row["N_capacity"])
        source_csv = Path(str(row["source_csv"]))
        checkpoint_dir = source_csv.parent / "checkpoints"
        checkpoint = checkpoint_dir / (
            f"N{n_capacity}_k0{int(row['k0'])}_seed4311.pt"
        )
        for group_k in k_grid:
            available = group_k <= n_capacity
            rows.append(
                {
                    "cell_index": lookup[key],
                    "model": model,
                    "concept": concept,
                    "task": task,
                    "layer": layer,
                    "N_capacity": n_capacity,
                    "l0_target": int(row["l0_target"]),
                    "recon_R2": float(row["recon_R2"]),
                    "matched_tier": str(row["matched_tier"]),
                    "group_k": group_k,
                    "group_status": "eligible" if available else "k_unavailable",
                    "environment": "csfm" if model == "CSFM" else "transformer",
                    "checkpoint_dir": str(checkpoint_dir),
                    "checkpoint": str(checkpoint),
                    "checkpoint_exists": checkpoint.exists(),
                    "source_csv": str(source_csv),
                }
            )

    coverage = pd.DataFrame(rows).sort_values(
        ["group_status", "model", "cell_index", "group_k"]
    )
    eligible = coverage[
        (coverage["group_status"] == "eligible")
        & coverage["checkpoint_exists"]
        & (coverage["matched_tier"] == "in_band")
    ].copy()
    eligible.insert(0, "task_index", range(len(eligible)))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(args.out_dir / "topk_v2_coverage.csv", index=False)
    eligible.to_csv(args.out_dir / "topk_v2_manifest.csv", index=False)
    missing = coverage[
        (coverage["group_status"] == "eligible") & ~coverage["checkpoint_exists"]
    ]
    missing.to_csv(args.out_dir / "topk_v2_missing_checkpoints.csv", index=False)

    print(f"selected operating points: {len(selected)}")
    print(f"eligible tasks: {len(eligible)}")
    print(f"k-unavailable combinations: {(coverage['group_status'] == 'k_unavailable').sum()}")
    print(f"missing eligible checkpoints: {len(missing)}")
    print(args.out_dir / "topk_v2_manifest.csv")


if __name__ == "__main__":
    main()
