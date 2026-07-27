#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import os
import textwrap
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "results" / "multicohort" / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ffloat(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def svg(width: int, height: int, body: list[str]) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">',
            '<style>',
            "text{font-family:Arial,Helvetica,sans-serif;fill:#1f2933}",
            ".title{font-size:28px;font-weight:700}",
            ".subtitle{font-size:16px;fill:#52616b}",
            ".label{font-size:15px;font-weight:700}",
            ".small{font-size:13px;fill:#52616b}",
            ".tiny{font-size:11px;fill:#52616b}",
            ".axis{stroke:#9aa5b1;stroke-width:1}",
            ".arrow{stroke:#52616b;stroke-width:2;fill:none;marker-end:url(#arrow)}",
            "</style>",
            "<defs>",
            '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L0,6 L9,3 z" fill="#52616b" />',
            "</marker>",
            "</defs>",
            *body,
            "</svg>",
        ]
    )


def text(x: float, y: float, content: str, cls: str = "small", anchor: str = "start") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}">{esc(content)}</text>'


def wrapped_text(x: float, y: float, content: str, width_chars: int, cls: str = "small", line_h: int = 16) -> list[str]:
    lines = textwrap.wrap(content, width=width_chars, break_long_words=False)
    return [text(x, y + i * line_h, line, cls=cls) for i, line in enumerate(lines)]


def box(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    fill: str,
    stroke: str = "#334e68",
) -> list[str]:
    out = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="7" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>',
        text(x + 14, y + 25, title, cls="label"),
    ]
    yy = y + 48
    for line in lines:
        out.extend(wrapped_text(x + 14, yy, line, max(18, int(w / 8.2)), cls="small"))
        yy += 34
    return out


def arrow(x1: float, y1: float, x2: float, y2: float) -> str:
    return f'<path d="M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}" class="arrow"/>'


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def make_mc1(root: Path, outdir: Path) -> Path:
    body: list[str] = [
        text(40, 42, "Figure MC-1. Measurement harmonization flow", cls="title"),
        text(40, 70, "External cohorts are gated robustness checks; PTB-XL/PTB-XL+ remains the full benchmark anchor.", cls="subtitle"),
    ]
    body.extend(
        box(
            40,
            120,
            250,
            150,
            "Anchor",
            [
                "PTB-XL waveforms",
                "PTB-XL+ measurement concepts",
                "Full Probe, LEACE, Closure, SAE benchmark",
            ],
            "#eef7ff",
        )
    )
    body.extend(
        box(
            345,
            100,
            250,
            120,
            "MIMIC-V",
            ["Vendor interval, rate, axis only", "QTc-like and axis-difference QRS-T angle"],
            "#f1f8e9",
        )
    )
    body.extend(
        box(
            650,
            100,
            235,
            120,
            "G1/G2/G4",
            ["Vendor audit and crosswalk", "ICD-linked task feasibility"],
            "#fff8e1",
        )
    )
    body.extend(
        box(
            940,
            100,
            330,
            120,
            "Track V closure",
            ["Restricted interval/rate/axis transfer", "No ST/amplitude claim from vendor measurements"],
            "#e8f5e9",
        )
    )
    body.extend(
        box(
            345,
            280,
            250,
            130,
            "Track F cohorts",
            ["MIMIC-F, Chapman-F, CPSC-F, Ningbo-F", "Common waveform-derived concepts"],
            "#f3e5f5",
        )
    )
    body.extend(
        box(
            650,
            280,
            235,
            130,
            "G3/G4",
            ["Waveform extractor gate", "Family and task feasibility gates"],
            "#fff8e1",
        )
    )
    body.extend(
        box(
            940,
            280,
            330,
            130,
            "Track F closure",
            ["Family-specific robustness checks", "Only gate-passed rows enter claims"],
            "#f3e5f5",
        )
    )
    body.extend(
        box(
            345,
            470,
            250,
            120,
            "External activations",
            ["Model/cohort activation caches", "Current available chunk: CSFM non-MIMIC"],
            "#eeeeee",
        )
    )
    body.extend(
        box(
            650,
            470,
            235,
            120,
            "SAE recon gate",
            ["PTB-XL SAE on external activations", "Steering only after recon pass"],
            "#fff8e1",
        )
    )
    body.extend(
        box(
            940,
            470,
            330,
            120,
            "Current SAE result",
            ["0/108 recon-gate passes", "33 CSFM rows below floor; 75 caches missing"],
            "#ffebee",
        )
    )
    body.extend(
        [
            arrow(290, 190, 345, 160),
            arrow(595, 160, 650, 160),
            arrow(885, 160, 940, 160),
            arrow(290, 215, 345, 345),
            arrow(595, 345, 650, 345),
            arrow(885, 345, 940, 345),
            arrow(290, 245, 345, 530),
            arrow(595, 530, 650, 530),
            arrow(885, 530, 940, 530),
        ]
    )
    out = outdir / "figure_mc1_measurement_harmonization_flow.svg"
    write_atomic(out, svg(1320, 640, body))
    return out


def make_mc2(root: Path, outdir: Path) -> Path:
    mimic_rows = [
        r
        for r in read_csv(root / "results" / "multicohort" / "mimic_v_closure" / "mimic_v_closure_summary.csv")
        if r["task_scope"] == "primary"
    ]
    trackf_rows = [
        r
        for r in read_csv(root / "results" / "multicohort" / "track_f_closure" / "closure_transfer_track_f.csv")
        if r["status"] == "ok" and r["task_scope"].startswith("primary")
    ]

    bars: list[dict[str, str | float]] = []
    for r in mimic_rows:
        bars.append(
            {
                "label": f"MIMIC-V {r['task'].replace('_icd', '')}",
                "gain": ffloat(r["bcommon_minus_brand_auroc"]),
                "kind": "Track V primary",
            }
        )
    for r in trackf_rows:
        cohort = r["cohort"].capitalize()
        task = r["task"].replace("_native", "").replace("_icd", "")
        bars.append(
            {
                "label": f"{cohort}-F {task}",
                "gain": ffloat(r["closure_gain_vs_brand_auroc"]),
                "kind": "Track F primary",
            }
        )
    bars.sort(key=lambda r: float(r["gain"]), reverse=True)

    width = 1300
    left = 330
    right = 1180
    top = 120
    row_h = 42
    height = top + row_h * len(bars) + 95
    max_abs = max(0.27, max(abs(float(r["gain"])) for r in bars) if bars else 0.27)
    scale = (right - left) / max_abs
    colors = {"Track V primary": "#2f80ed", "Track F primary": "#27ae60"}
    body = [
        text(40, 42, "Figure MC-2. External closure transfer", cls="title"),
        text(40, 70, "Bars show AUROC gain over matched random baselines for gate-passed primary rows.", cls="subtitle"),
        text(left, 104, "0", cls="tiny", anchor="middle"),
        f'<line x1="{left}" y1="108" x2="{right}" y2="108" class="axis"/>',
    ]
    for tick in [0.05, 0.10, 0.15, 0.20, 0.25]:
        x = left + tick * scale
        body.append(f'<line x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="{height-70}" stroke="#d9e2ec" stroke-width="1"/>')
        body.append(text(x, 104, f"{tick:.2f}", cls="tiny", anchor="middle"))
    for i, row in enumerate(bars):
        y = top + i * row_h
        gain = float(row["gain"])
        label = str(row["label"])
        kind = str(row["kind"])
        color = colors[kind]
        bar_w = gain * scale
        body.append(text(40, y + 21, label, cls="small"))
        body.append(
            f'<rect x="{left:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="25" rx="4" fill="{color}" opacity="0.88"/>'
        )
        body.append(text(left + bar_w + 8, y + 18, f"+{gain:.3f}", cls="small"))
    legend_y = height - 42
    body.append(f'<rect x="40" y="{legend_y-16}" width="16" height="16" fill="#2f80ed"/>')
    body.append(text(64, legend_y - 3, "MIMIC-V primary ICD-linked Track V", cls="small"))
    body.append(f'<rect x="360" y="{legend_y-16}" width="16" height="16" fill="#27ae60"/>')
    body.append(text(384, legend_y - 3, "Track F primary waveform-derived rows", cls="small"))

    out = outdir / "figure_mc2_external_closure_transfer.svg"
    write_atomic(out, svg(width, height, body))
    return out


def make_mc3(root: Path, outdir: Path) -> Path:
    rows = read_csv(root / "results" / "multicohort" / "external_sae" / "external_sae_recon_gate.csv")
    by: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by[(r["model"], r["external_cohort"])].append(r)

    models = sorted({m for m, _ in by})
    cohorts = ["MIMIC-F", "Chapman-F", "CPSC-F", "Ningbo-F"]
    cell_w = 170
    cell_h = 68
    x0 = 220
    y0 = 125
    width = x0 + cell_w * len(cohorts) + 70
    height = y0 + cell_h * len(models) + 120

    def status_for(group: list[dict[str, str]]) -> tuple[str, str, str]:
        if any(r["recon_gate_pass"] == "true" for r in group):
            return "pass", "#2e7d32", "pass"
        if any(r["external_activation_status"] == "available" for r in group):
            vals = [ffloat(r["external_recon_r2"], -1.0) for r in group if r["external_recon_r2"]]
            label = f"below floor\nR2 {min(vals):.3f}-{max(vals):.3f}" if vals else "below floor"
            return "below", "#f9d65c", label
        return "missing", "#d9e2ec", "missing cache"

    body = [
        text(40, 42, "Figure MC-3. External SAE reconstruction gate", cls="title"),
        text(40, 70, "Steering is evaluated only if the external activation distribution passes SAE reconstruction fidelity.", cls="subtitle"),
    ]
    for j, cohort in enumerate(cohorts):
        body.append(text(x0 + j * cell_w + cell_w / 2, y0 - 24, cohort, cls="label", anchor="middle"))
    for i, model in enumerate(models):
        y = y0 + i * cell_h
        body.append(text(40, y + 38, model, cls="label"))
        for j, cohort in enumerate(cohorts):
            x = x0 + j * cell_w
            group = by.get((model, cohort), [])
            _, color, label = status_for(group)
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-12:.1f}" height="{cell_h-12:.1f}" rx="5" '
                f'fill="{color}" stroke="#627d98" stroke-width="1"/>'
            )
            lines = label.split("\n")
            for k, line in enumerate(lines):
                body.append(text(x + (cell_w - 12) / 2, y + 25 + k * 16, line, cls="small", anchor="middle"))
    legend_y = height - 55
    for x, color, label in [
        (40, "#2e7d32", "recon pass"),
        (210, "#f9d65c", "available but below 0.90 floor"),
        (520, "#d9e2ec", "missing activation cache"),
    ]:
        body.append(f'<rect x="{x}" y="{legend_y-16}" width="16" height="16" fill="{color}" stroke="#627d98"/>')
        body.append(text(x + 24, legend_y - 3, label, cls="small"))
    body.append(text(40, height - 18, "Current result: CSFM-only external recon below floor; steering and six-model transfer claims remain disallowed.", cls="subtitle"))

    out = outdir / "figure_mc3_external_sae_recon_gate.svg"
    write_atomic(out, svg(width, height, body))
    return out


def make_report(paths: list[Path], outdir: Path) -> Path:
    lines = [
        "# Multi-Cohort Figures",
        "",
        "Auto-generated from current multi-cohort CSV artifacts. These figures are robustness/gate summaries, not leaderboard figures.",
        "",
        "| Figure | File | Interpretation discipline |",
        "|---|---|---|",
        f"| MC-1 | `{paths[0].name}` | Shows gated paths from PTB-XL/PTB-XL+ anchor to MIMIC-V, Track F, and external SAE. |",
        f"| MC-2 | `{paths[1].name}` | Shows only gate-passed primary closure rows as AUROC gain over random baselines. |",
        f"| MC-3 | `{paths[2].name}` | Shows external SAE reconstruction status before any steering claim. |",
        "",
        "Current SAE claim discipline: observed external SAE reconstruction failures are CSFM-specific (33 available CSFM rows below floor; 75 rows missing activation caches). No external SAE steering failure/success claim and no six-model SAE transfer generalization are allowed.",
        "",
    ]
    out = outdir / "multicohort_figures_report.md"
    write_atomic(out, "\n".join(lines))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready SVG figures for the multi-cohort section.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    paths = [
        make_mc1(ROOT, args.outdir),
        make_mc2(ROOT, args.outdir),
        make_mc3(ROOT, args.outdir),
    ]
    report = make_report(paths, args.outdir)
    for path in paths + [report]:
        print(f"wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
