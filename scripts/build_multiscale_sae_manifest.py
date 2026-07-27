#!/usr/bin/env python
"""Build the frozen source-cohort multi-scale SAE training manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.multiscale_sae import build_manifest_rows


def parse_ints(spec: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in spec.split(",") if value.strip())


def parse_floats(spec: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in spec.split(",") if value.strip())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "results/multiscale_sae_v1")
    parser.add_argument("--version", default="multiscale_sae_v1")
    parser.add_argument("--benchmark-role", default="primary_source_atlas")
    parser.add_argument("--expansions", default="1,4,8,16,32")
    parser.add_argument("--seeds", default="4311,4312,4313")
    parser.add_argument("--depths", default="0,0.25,0.5,0.75,1")
    parser.add_argument(
        "--sparsity-arm",
        choices=("fixed_k_over_d", "fixed_k_over_n"),
        default="fixed_k_over_d",
    )
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    rows = build_manifest_rows(
        ROOT,
        args.out,
        expansions=parse_ints(args.expansions),
        seeds=parse_ints(args.seeds),
        depths=parse_floats(args.depths),
        sparsity_arm=args.sparsity_arm,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    if not rows:
        raise RuntimeError("manifest would be empty")
    expansions = parse_ints(args.expansions)
    seeds = parse_ints(args.seeds)
    depths = parse_floats(args.depths)
    models = sorted({str(row["model"]) for row in rows})
    expected_cells = len(models) * len(depths) * len(expansions) * len(seeds)
    cell_keys = [
        (
            str(row["model"]),
            float(row["relative_depth"]),
            int(row["expansion_E"]),
            int(row["seed"]),
        )
        for row in rows
    ]
    if len(rows) != expected_cells or len(set(cell_keys)) != expected_cells:
        raise RuntimeError(
            f"manifest is not a complete matched-scale grid: rows={len(rows)}, "
            f"unique={len(set(cell_keys))}, expected={expected_cells}"
        )
    for depth in depths:
        for expansion in expansions:
            for seed in seeds:
                block_models = {
                    str(row["model"])
                    for row in rows
                    if float(row["relative_depth"]) == depth
                    and int(row["expansion_E"]) == expansion
                    and int(row["seed"]) == seed
                }
                if block_models != set(models):
                    raise RuntimeError(
                        "unmatched model support at "
                        f"depth={depth}, E={expansion}, seed={seed}: {sorted(block_models)}"
                    )
    write_csv(args.out / "training_manifest.csv", rows)

    layers = sorted({(str(row["model"]), int(row["layer"])) for row in rows})
    hidden_dimensions = sorted({int(row["d_hidden"]) for row in rows})
    widths_by_expansion = {
        str(expansion): sorted(
            {int(row["N"]) for row in rows if int(row["expansion_E"]) == expansion}
        )
        for expansion in expansions
    }
    budgets_by_expansion = {
        str(expansion): sorted(
            {int(row["k"]) for row in rows if int(row["expansion_E"]) == expansion}
        )
        for expansion in expansions
    }
    protocol = {
        "version": args.version,
        "benchmark_role": args.benchmark_role,
        "benchmark_object": "ECG foundation-model representations",
        "measurement_instrument": "BatchTopK sparse autoencoder",
        "source_cohort": "PTB-XL",
        "models": models,
        "relative_depths": list(parse_floats(args.depths)),
        "expansion_E": list(parse_ints(args.expansions)),
        "seeds": list(parse_ints(args.seeds)),
        "sparsity_arm": args.sparsity_arm,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "n_model_layer_cells": len(layers),
        "n_training_cells": len(rows),
        "scale_definition": "relative expansion E=N/d_FM; identical E values are used for every FM",
        "layer_hidden_dimensions": hidden_dimensions,
        "absolute_widths_by_E": widths_by_expansion,
        "active_budgets_by_E": budgets_by_expansion,
        "exact_absolute_scale_matching": all(
            len(widths_by_expansion[str(expansion)]) == 1
            and len(budgets_by_expansion[str(expansion)]) == 1
            for expansion in expansions
        ),
        "comparison_rule": "compare FMs only within common (relative depth, E, seed) blocks",
        "selection_rule": "No per-model best-scale ranking and no test-set selection; primary summaries integrate the same frozen E grid for every FM.",
    }
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    lines = [
        "# Multi-Scale SAE Protocol v1",
        "",
        "- Benchmark object: frozen ECG foundation-model layer representations.",
        "- Measurement instrument: one BatchTopK SAE per model, layer, scale, and seed.",
        "- Source cohort: PTB-XL with existing patient-level train/validation/test splits.",
        f"- Models: {', '.join(models)}.",
        f"- Standardized relative depths: {args.depths}.",
        f"- Expansion ratios `N/d`: {args.expansions}.",
        f"- Layer hidden dimensions: {','.join(map(str, hidden_dimensions))}.",
        f"- Exact absolute width/budget matching across FMs: {protocol['exact_absolute_scale_matching']}.",
        f"- Primary sparsity arm: `{args.sparsity_arm}`.",
        f"- Seeds: {args.seeds}.",
        f"- Training budget: {args.steps} steps, batch size {args.batch_size}, Adam lr {args.learning_rate:g}.",
        "- Clinical alignment: select the strongest SAE feature per concept on train and evaluate that frozen feature on validation/test.",
        "- Scale matching: every FM is evaluated at the same relative expansion `E=N/d_FM` within each standardized depth and seed block.",
        "- Primary model comparison uses the complete common layer-scale surface; per-model best-scale comparisons are prohibited.",
        f"- Expected training cells: {len(rows)}.",
    ]
    (args.out / "protocol.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
