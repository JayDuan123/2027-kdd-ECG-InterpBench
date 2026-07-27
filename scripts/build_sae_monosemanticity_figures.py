#!/usr/bin/env python
"""Build monosemanticity taxonomy figures and comparison tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "results/analysis/model_comparison/sae_monosemanticity"


COLORS = {
    "mono": "#2ca25f",
    "entangled": "#de2d26",
    "dead": "#756bb1",
    "inactive": "#9e9e9e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--ratio", type=float, default=3.0)
    return parser.parse_args()


def svg_text(x: float, y: float, text: str, size: int = 11, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{text}</text>'
    )


def build_stacked_svg(summary: pd.DataFrame, out_path: Path, ratio: float) -> None:
    data = summary[summary["ratio"].astype(float).eq(ratio)].copy()
    data = data[data["concept_set"].isin(["FULL", "ORTHO", "ORTHO_METADATA"])]
    models = ["CARDIAC-FM", "CSFM", "ECG-FM", "ECG-JEPA", "HuBERT-ECG", "ST-MEM"]
    sets = ["FULL", "ORTHO", "ORTHO_METADATA"]
    width = 980
    left = 210
    top = 80
    bar_w = 680
    bar_h = 16
    row_gap = 24
    group_gap = 18
    height = top + len(models) * (len(sets) * row_gap + group_gap) + 80
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(20, 30, f"SAE monosemanticity taxonomy, ratio={ratio:g}", 18, weight="bold"),
        svg_text(20, 52, "Stacked fractions per model/concept set; observational enrichment only, no steering.", 12),
    ]
    legend_x = left
    for label, key in [("monosemantic", "mono"), ("entangled", "entangled"), ("dead", "dead"), ("inactive", "inactive")]:
        lines.append(f'<rect x="{legend_x}" y="42" width="14" height="14" fill="{COLORS[key]}"/>')
        lines.append(svg_text(legend_x + 18, 54, label, 11))
        legend_x += 120
    y = top
    for model in models:
        model_rows = data[data["model"].eq(model)]
        lines.append(svg_text(20, y + 15, model, 12, weight="bold"))
        if model_rows.empty:
            lines.append(svg_text(left, y + 15, "missing recon-band operating point", 11))
            y += len(sets) * row_gap + group_gap
            continue
        for concept_set in sets:
            row = model_rows[model_rows["concept_set"].eq(concept_set)]
            lines.append(svg_text(105, y + 13, concept_set, 10))
            if row.empty:
                lines.append(svg_text(left, y + 13, "not reported", 10))
                y += row_gap
                continue
            r = row.iloc[0]
            vals = [
                ("mono", float(r["mono_frac_mean"])),
                ("entangled", float(r["entangled_frac_mean"])),
                ("dead", float(r["dead_frac_mean"])),
                ("inactive", float(r["inactive_frac_mean"])),
            ]
            x = left
            for key, value in vals:
                seg_w = max(0.0, value) * bar_w
                if seg_w > 0:
                    lines.append(
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{seg_w:.1f}" height="{bar_h}" fill="{COLORS[key]}"/>'
                    )
                x += seg_w
            lines.append(f'<rect x="{left:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#333" stroke-width="0.4"/>')
            mono_pct = 100.0 * float(r["mono_frac_mean"])
            sd_pct = 100.0 * float(r["mono_frac_sd"])
            n_runs = int(r["n_runs"])
            in_band = int(r.get("in_band_runs", n_runs))
            lines.append(svg_text(left + bar_w + 10, y + 12, f"mono {mono_pct:.1f}% +/- {sd_pct:.1f}; n={n_runs}, in-band={in_band}", 10))
            y += row_gap
        y += group_gap
    lines.extend(
        [
            svg_text(20, height - 36, "Note: ST-MEM lacks a primary R2 in [0.90, 0.92] SAE operating point in current artifacts.", 11),
            svg_text(20, height - 18, "FULL clinical concept mono fraction is zero for all evaluated recon-band ECG FM dictionaries.", 11, weight="bold"),
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def build_comparison(summary: pd.DataFrame, out_dir: Path, ratio: float) -> None:
    full = summary[(summary["ratio"].astype(float).eq(ratio)) & (summary["concept_set"].eq("FULL"))]
    evaluated = full["model"].nunique()
    max_full_mono = float(full["mono_frac_mean"].max()) if not full.empty else float("nan")
    mean_full_mono = float(full["mono_frac_mean"].mean()) if not full.empty else float("nan")
    rows = [
        {
            "source": "EEG-SAE comparator",
            "concept_domain": "EEG FM concepts",
            "separable_or_monosemantic_fraction": "substantial in reported EEG-SAE Fig 3",
            "interpretation": "SAE-friendly comparator; exact numeric value not present in local pasted spec",
        },
        {
            "source": "ECG-FM benchmark",
            "concept_domain": "FULL 49 ECG clinical measurement concepts",
            "separable_or_monosemantic_fraction": f"mean={mean_full_mono:.3f}, max={max_full_mono:.3f}, evaluated_models={evaluated}",
            "interpretation": "SAE-hostile/distributed: no evaluated recon-band ECG FM dictionary has FULL-set monosemantic features at ratio 3",
        },
    ]
    comp = pd.DataFrame(rows)
    comp.to_csv(out_dir / "monosemanticity_eeg_sae_comparison.csv", index=False)
    md = [
        "# EEG-SAE Comparison Note",
        "",
        "The local pasted spec does not provide the numeric EEG-SAE Fig. 3 separable fraction, so this comparison is qualitative on the EEG side and numeric on the ECG side.",
        "",
        "| source | concept_domain | separable_or_monosemantic_fraction | interpretation |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        md.append(
            f"| {row['source']} | {row['concept_domain']} | {row['separable_or_monosemantic_fraction']} | {row['interpretation']} |"
        )
    md.append("")
    (out_dir / "monosemanticity_eeg_sae_comparison.md").write_text("\n".join(md))


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.in_dir / "monosemanticity_summary.csv")
    build_stacked_svg(summary, args.in_dir / "monosemanticity_ratio3_stacked.svg", args.ratio)
    build_comparison(summary, args.in_dir, args.ratio)
    print(f"wrote {args.in_dir / 'monosemanticity_ratio3_stacked.svg'}")
    print(f"wrote {args.in_dir / 'monosemanticity_eeg_sae_comparison.csv'}")
    print(f"wrote {args.in_dir / 'monosemanticity_eeg_sae_comparison.md'}")


if __name__ == "__main__":
    main()
