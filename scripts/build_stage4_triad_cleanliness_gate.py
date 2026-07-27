#!/usr/bin/env python
"""Build the Stage IV orthogonal-triad cleanliness gate.

Stage IV is a positive-control steering stage. Before any positive-control
steering run, the locked triad must be audited for hidden st_amp-like failure
modes:

GATE-A: no strong out-of-group connected damage.
GATE-B: not distributed/nonselective in existing anchor SAE diagnostics.

This script materializes that pre-steering gate from existing LEACE coupling and
SAE summary artifacts. It intentionally does not launch any steering jobs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORTHO_DIR = ROOT / "results" / "analysis" / "model_comparison" / "orthogonal_concepts"
SAE_DIR = ROOT / "results" / "sae_extension"
SIX_MODEL_DIR = SAE_DIR / "six_model_sae_audit"

TRIAD = ["hr_ventricular", "qrs_axis_front", "qrs_duration"]
GATE_A_CONN_THRESHOLD = 0.20
GATE_B_A_GEO_THRESHOLD = 0.50
GATE_B_WBI_EXCESS_TOLERANCE = 0.25


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def to_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def fmt(value: Any, ndigits: int = 8) -> str:
    value = to_float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{ndigits}g}"


def best_row(rows: list[dict[str, str]], key: str) -> dict[str, str] | None:
    finite = [row for row in rows if math.isfinite(to_float(row.get(key)))]
    if not finite:
        return None
    return max(finite, key=lambda row: to_float(row.get(key)))


def source_outgroup_rows(coupling: list[dict[str, str]], concept: str) -> list[dict[str, str]]:
    return [
        row
        for row in coupling
        if row.get("source_concept") == concept
        and row.get("target_concept") != concept
        and row.get("target_concept") not in TRIAD
    ]


def bidirectional_outgroup_rows(coupling: list[dict[str, str]], concept: str) -> list[dict[str, str]]:
    out = []
    for row in coupling:
        source = row.get("source_concept")
        target = row.get("target_concept")
        if source == concept and target != concept and target not in TRIAD:
            out.append(row)
        elif target == concept and source != concept and source not in TRIAD:
            out.append(row)
    return out


def candidate_sort_key(row: dict[str, str]) -> tuple[int, float, float]:
    """Prefer main old robustness, then main l0clamp, then other diagnostics."""
    source = row.get("_b_source", "")
    source_rank = {
        "csfm_main_robustness_E8": 0,
        "l0clamp_main_recon_0.90": 1,
        "l0clamp_sensitivity_recon_0.95": 2,
        "csfm_fixed_interpretation": 3,
    }.get(source, 9)
    recon = to_float(row.get("recon_R2") or row.get("recon_r2"))
    a_geo = to_float(row.get("A_geo_cav") or row.get("A_geo_cav_mean"))
    return (source_rank, -recon if math.isfinite(recon) else 0.0, -a_geo if math.isfinite(a_geo) else 0.0)


def load_b_gate_candidates() -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {concept: [] for concept in TRIAD}

    # Historical CSFM robustness table: this is the exact style referenced by
    # the st_amp special-case gate (E=8 A_geo/WBI/random WBI).
    for row in read_csv(SAE_DIR / "csfm_sae_main_robustness_summary.csv"):
        concept = row.get("concept")
        if concept in out and row.get("E") == "8":
            enriched = dict(row)
            enriched["_b_source"] = "csfm_main_robustness_E8"
            enriched["_model"] = "CSFM"
            enriched["_task"] = row.get("task", "")
            enriched["_analysis"] = "robustness_E8"
            out[concept].append(enriched)

    # Six-model l0clamp summaries provide existing Stage-IV-style diagnostics
    # for some concepts. These are used only as gate evidence, never to select a
    # new concept after a steering outcome.
    for path, source in [
        (
            SIX_MODEL_DIR / "l0clamp_reclassified" / "sae_l0clamp_reclassified_cells.csv",
            "l0clamp_main_or_sensitivity",
        ),
        (SIX_MODEL_DIR / "l0clamp_summary" / "sae_l0clamp_combined_results.csv", "l0clamp_main_recon_0.90"),
        (
            SIX_MODEL_DIR
            / "l0clamp_sensitivity95_summary"
            / "sae_l0clamp_combined_results.csv",
            "l0clamp_sensitivity_recon_0.95",
        ),
    ]:
        for row in read_csv(path):
            concept = row.get("concept")
            if concept not in out:
                continue
            enriched = dict(row)
            analysis = row.get("analysis", "")
            if source == "l0clamp_main_or_sensitivity":
                if analysis == "main_recon_0.90":
                    enriched["_b_source"] = "l0clamp_main_recon_0.90"
                elif analysis == "sensitivity_recon_0.95":
                    enriched["_b_source"] = "l0clamp_sensitivity_recon_0.95"
                else:
                    enriched["_b_source"] = source
            else:
                enriched["_b_source"] = source
            enriched["_model"] = row.get("model", "")
            enriched["_task"] = row.get("task", "")
            enriched["_analysis"] = analysis or row.get("recon_target", "")
            out[concept].append(enriched)

    return out


def select_b_gate_candidate(candidates: list[dict[str, str]]) -> dict[str, str] | None:
    if not candidates:
        return None
    return sorted(candidates, key=candidate_sort_key)[0]


def b_gate_fields(candidate: dict[str, str] | None) -> dict[str, Any]:
    if candidate is None:
        return {
            "A_geo": "",
            "WBI": "",
            "random_WBI": "",
            "wbi_minus_random_WBI": "",
            "passes_B": "NA",
            "B_gate_reason": "missing_anchor_sae_gate_metrics",
            "B_metric_source": "",
            "B_model": "",
            "B_task": "",
            "B_analysis": "",
            "B_recon_R2": "",
        }

    a_geo = to_float(candidate.get("A_geo_cav") or candidate.get("A_geo_cav_mean"))
    wbi = to_float(candidate.get("wbi") or candidate.get("wbi_median"))
    random_wbi = to_float(candidate.get("random_wbi_mean") or candidate.get("random_wbi_seed_mean_mean"))
    recon = to_float(candidate.get("recon_R2") or candidate.get("recon_r2"))
    wbi_excess = wbi - random_wbi

    a_pass = math.isfinite(a_geo) and a_geo >= GATE_B_A_GEO_THRESHOLD
    wbi_pass = (
        math.isfinite(wbi)
        and math.isfinite(random_wbi)
        and wbi_excess <= GATE_B_WBI_EXCESS_TOLERANCE
    )
    passes_b = a_pass and wbi_pass
    reasons = []
    if not a_pass:
        reasons.append("A_geo_below_0.50_or_missing")
    if not wbi_pass:
        reasons.append("WBI_exceeds_random_by_more_than_0.25_or_missing")
    return {
        "A_geo": fmt(a_geo),
        "WBI": fmt(wbi),
        "random_WBI": fmt(random_wbi),
        "wbi_minus_random_WBI": fmt(wbi_excess),
        "passes_B": str(passes_b),
        "B_gate_reason": "pass" if passes_b else "|".join(reasons),
        "B_metric_source": candidate.get("_b_source", ""),
        "B_model": candidate.get("_model", ""),
        "B_task": candidate.get("_task", ""),
        "B_analysis": candidate.get("_analysis", ""),
        "B_recon_R2": fmt(recon),
    }


def role_from_gates(passes_a: bool, passes_b: str, max_out: float) -> str:
    if not passes_a:
        return "moved_to_case_study_out_of_group_collinearity"
    if passes_b == "True":
        return "positive_control"
    if passes_b == "NA":
        return "pending_anchor_sae_gate"
    return "moved_to_case_study_distributed_or_nonselective"


def build_gate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    coupling = read_csv(args.orthogonal_dir / "selection_coupling_residual_matrix.csv")
    b_candidates = load_b_gate_candidates()
    rows: list[dict[str, Any]] = []
    for concept in TRIAD:
        source_best = best_row(source_outgroup_rows(coupling, concept), "target_r2_drop")
        bidir_best = best_row(bidirectional_outgroup_rows(coupling, concept), "target_r2_drop")

        max_out = to_float(source_best.get("target_r2_drop") if source_best else "")
        max_bidir = to_float(bidir_best.get("target_r2_drop") if bidir_best else "")
        passes_a = math.isfinite(max_out) and max_out < GATE_A_CONN_THRESHOLD

        b_fields = b_gate_fields(select_b_gate_candidate(b_candidates.get(concept, [])))
        role = role_from_gates(passes_a, b_fields["passes_B"], max_out)

        rows.append(
            {
                "concept": concept,
                "max_outgroup_conn_damage": fmt(max_out),
                "max_outgroup_pair": (
                    f"{source_best.get('source_concept')}->{source_best.get('target_concept')}"
                    if source_best
                    else ""
                ),
                "max_outgroup_abs_spearman": fmt(
                    source_best.get("source_target_abs_spearman_r") if source_best else ""
                ),
                "max_outgroup_model": source_best.get("model", "") if source_best else "",
                "max_outgroup_task": source_best.get("source_task", "") if source_best else "",
                "max_outgroup_layer": source_best.get("source_layer", "") if source_best else "",
                "max_bidirectional_outgroup_conn_damage": fmt(max_bidir),
                "max_bidirectional_outgroup_pair": (
                    f"{bidir_best.get('source_concept')}->{bidir_best.get('target_concept')}"
                    if bidir_best
                    else ""
                ),
                "passes_A": str(passes_a),
                "A_gate_reason": "pass" if passes_a else "outgroup_connected_damage_ge_0.20",
                **b_fields,
                "role": role,
            }
        )
    return rows


def markdown_report(rows: list[dict[str, Any]]) -> str:
    clean = [row for row in rows if row["role"] == "positive_control"]
    moved = [row for row in rows if row["role"].startswith("moved_to_case_study")]
    pending = [row for row in rows if row["role"] == "pending_anchor_sae_gate"]
    table = "\n".join(
        [
            (
                f"| {row['concept']} | {row['max_outgroup_conn_damage']} | "
                f"{row['max_outgroup_pair']} | {row['A_geo']} | {row['WBI']} | "
                f"{row['random_WBI']} | {row['passes_A']} | {row['passes_B']} | {row['role']} |"
            )
            for row in rows
        ]
    )
    if len(clean) >= 2:
        prereg = (
            "Stage IV may proceed only for the clean survivors listed as "
            "`positive_control`, after locking off-targets among the clean set."
        )
    else:
        prereg = (
            "Stage IV positive-control steering is blocked: fewer than two clean "
            "positive-control survivors are available, so a locked off-target cannot "
            "be preregistered without violating Step 0."
        )
    return f"""# Stage IV Triad Cleanliness Gate

This report implements the mandatory pre-steering gate from Stage IV. It uses
existing LEACE coupling and SAE diagnostics only; no new steering job is
submitted here.

Thresholds:

- GATE-A: max source-to-outgroup connected damage `< {GATE_A_CONN_THRESHOLD}`
- GATE-B: A_geo_cav `>= {GATE_B_A_GEO_THRESHOLD}` and WBI not higher than random
  WBI by more than `{GATE_B_WBI_EXCESS_TOLERANCE}`

| concept | max outgroup conn | strongest outgroup pair | A_geo | WBI | random WBI | passes A | passes B | role |
|---|---:|---|---:|---:|---:|---|---|---|
{table}

Summary:

- Clean positive-control survivors: `{len(clean)}` ({", ".join(row["concept"] for row in clean) or "none"})
- Moved to case study: `{len(moved)}` ({", ".join(row["concept"] for row in moved) or "none"})
- Pending anchor SAE gate: `{len(pending)}` ({", ".join(row["concept"] for row in pending) or "none"})

Decision:

{prereg}

Interpretation:

- `hr_ventricular` fails GATE-A because its representation is strongly tied to
  the out-of-group rate concept `rr_mean`.
- `qrs_duration` fails GATE-A because it is coupled to `qrst_angle`; existing
  SAE diagnostics also show a low-A_geo/nonselective regime in the main
  recon-0.90 setting.
- `qrs_axis_front` passes GATE-A, but existing WBI evidence is not clean enough
  to preregister it as a positive-control survivor by itself.

The corrected implication is stronger and stricter than the initial triad
assumption: Stage IV cannot be run as a three-concept positive control until the
clean-survivor set is repaired or additional anchor SAE evidence changes the
gate status.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthogonal-dir", type=Path, default=ORTHO_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_gate_rows(args)
    out_dir = args.orthogonal_dir
    clean = [row for row in rows if row["role"] == "positive_control"]
    write_csv(
        out_dir / "triad_cleanliness_gate.csv",
        rows,
        [
            "concept",
            "max_outgroup_conn_damage",
            "max_outgroup_pair",
            "max_outgroup_abs_spearman",
            "max_outgroup_model",
            "max_outgroup_task",
            "max_outgroup_layer",
            "max_bidirectional_outgroup_conn_damage",
            "max_bidirectional_outgroup_pair",
            "passes_A",
            "A_gate_reason",
            "A_geo",
            "WBI",
            "random_WBI",
            "wbi_minus_random_WBI",
            "passes_B",
            "B_gate_reason",
            "B_metric_source",
            "B_model",
            "B_task",
            "B_analysis",
            "B_recon_R2",
            "role",
        ],
    )
    write_csv(
        out_dir / "stage4_clean_positive_control_set.csv",
        [
            {
                "concept_id": row["concept"],
                "A_geo": row["A_geo"],
                "WBI": row["WBI"],
                "random_WBI": row["random_WBI"],
                "off_target_concept": "",
                "prediction": "selective_steering_expected",
                "preregistered": "False",
            }
            for row in clean
        ],
        [
            "concept_id",
            "A_geo",
            "WBI",
            "random_WBI",
            "off_target_concept",
            "prediction",
            "preregistered",
        ],
    )
    status = "ready_for_preregistration" if len(clean) >= 2 else "blocked_clean_survivors_lt2"
    write_csv(
        out_dir / "stage4_preregistration_status.csv",
        [
            {
                "status": status,
                "clean_survivor_count": len(clean),
                "clean_survivors": ";".join(row["concept"] for row in clean),
                "may_submit_stage4_steering": str(len(clean) >= 2),
                "reason": (
                    "at_least_two_clean_survivors_available"
                    if len(clean) >= 2
                    else "fewer_than_two_clean_positive_controls_after_mandatory_gate"
                ),
            }
        ],
        ["status", "clean_survivor_count", "clean_survivors", "may_submit_stage4_steering", "reason"],
    )
    write_text(out_dir / "stage4_triad_cleanliness_gate_report.md", markdown_report(rows))


if __name__ == "__main__":
    main()
