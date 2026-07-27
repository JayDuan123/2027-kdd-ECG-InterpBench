#!/usr/bin/env python
"""Summarise Phase 0 recon-band SAE curves."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/sae_extension/six_model_sae_audit/phase0_recon_grid")
    parser.add_argument("--out", default="results/sae_extension/six_model_sae_audit/phase0_recon_curves.csv")
    parser.add_argument("--summary", default="results/sae_extension/six_model_sae_audit/phase0_recon_summary.md")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("**/sae_recon_curve.csv")):
        if path.stat().st_size <= 1:
            continue
        rel_parts = path.relative_to(root).parts
        cell_dir = next((part for part in rel_parts if part.startswith("cell_")), path.parent.name)
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                row["cell_dir"] = cell_dir
                row["source_csv"] = str(path)
                rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        Path(args.summary).write_text("# Phase 0 Recon Summary\n\nNo recon curve rows found.\n", encoding="utf-8")
        print(out)
        print(args.summary)
        return

    df = pd.DataFrame(rows)
    numeric_cols = [
        "layer",
        "recon_target",
        "E",
        "N_capacity",
        "k",
        "l0_target",
        "l0_actual",
        "recon_R2",
        "dead_frac",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.to_csv(out, index=False)

    lines = ["# Phase 0 Recon Summary", ""]
    lines.append(f"- recon rows: {len(df)}")
    lines.append(f"- cells with recon curves: {df[['model', 'concept', 'task', 'layer']].drop_duplicates().shape[0]}")
    lines.append("")
    lines.append("## Tier Counts By Model")
    lines.append("")
    lines.append("| model | in_band | relaxed_band | no_matched_point | missing_artifact |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for model, part in sorted(df.groupby("model"), key=lambda item: item[0]):
        counts = part["matched_tier"].value_counts()
        lines.append(
            f"| {model} | {int(counts.get('in_band', 0))} | "
            f"{int(counts.get('relaxed_band', 0))} | {int(counts.get('no_matched_point', 0))} | "
            f"{int(counts.get('missing_artifact', 0))} |"
        )
    lines.append("")
    lines.append("## Best In-Band Point Per Model")
    lines.append("")
    lines.append("| model | cell | E | N | L0 | recon_R2 | dead_frac |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for model, part in sorted(df[df["matched_tier"] == "in_band"].groupby("model"), key=lambda item: item[0]):
        best = part.sort_values(["N_capacity", "dead_frac", "l0_actual"]).iloc[0]
        cell = f"{best['concept']} -> {best['task']} @ L{int(best['layer'])}"
        e_value = f"{float(best['E']):.3g}"
        lines.append(
            f"| {model} | {cell} | {e_value} | {int(best['N_capacity'])} | "
            f"{best['l0_actual']:.1f} | {best['recon_R2']:.3f} | {best['dead_frac']:.3f} |"
        )
    summary = Path(args.summary)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(summary)


if __name__ == "__main__":
    main()
