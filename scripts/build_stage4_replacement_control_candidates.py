#!/usr/bin/env python
"""Find replacement Stage IV orthogonal positive-control candidates.

The original triad failed the mandatory cleanliness gate. This script searches
the existing LEACE coupling matrix for alternative concepts that are:

1. pairwise orthogonal under GATE-2;
2. individually clean under GATE-A (source-to-outgroup connected damage < 0.20);
3. optionally supported by existing l0-clamp B-gate diagnostics.

It writes a replacement-candidate report and a missing-B-gate run list. No jobs
are submitted here.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ORTHO_DIR = ROOT / "results" / "analysis" / "model_comparison" / "orthogonal_concepts"
SAE_DIR = ROOT / "results" / "sae_extension"
SIX_MODEL_DIR = SAE_DIR / "six_model_sae_audit"

PAIR_SPEARMAN_THRESHOLD = 0.30
PAIR_CONN_THRESHOLD = 0.20
SOURCE_GATE_A_THRESHOLD = 0.20
GATE_B_A_GEO_THRESHOLD = 0.50
GATE_B_WBI_EXCESS_TOL = 0.25


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


def fmt(value: Any, digits: int = 8) -> str:
    value = to_float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def load_coupling(path: Path) -> tuple[dict[tuple[str, str], float], dict[frozenset[str], float], dict[str, str]]:
    drops: dict[tuple[str, str], float] = {}
    spearman: dict[frozenset[str], float] = {}
    families: dict[str, str] = {}
    for row in read_csv(path):
        source = row.get("source_concept", "")
        target = row.get("target_concept", "")
        if not source or not target:
            continue
        families[source] = row.get("source_family", "")
        families[target] = row.get("target_family", "")
        drop = to_float(row.get("target_r2_drop"))
        if math.isfinite(drop):
            drops[(source, target)] = max(drops.get((source, target), float("-inf")), drop)
        sp = to_float(row.get("source_target_abs_spearman_r"))
        if math.isfinite(sp):
            spearman[frozenset([source, target])] = sp
    return drops, spearman, families


def pair_gate(
    a: str,
    b: str,
    drops: dict[tuple[str, str], float],
    spearman: dict[frozenset[str], float],
) -> tuple[bool, float, float, float]:
    sp = spearman.get(frozenset([a, b]), float("nan"))
    ab = drops.get((a, b), float("nan"))
    ba = drops.get((b, a), float("nan"))
    ok = (
        math.isfinite(sp)
        and math.isfinite(ab)
        and math.isfinite(ba)
        and sp < PAIR_SPEARMAN_THRESHOLD
        and ab < PAIR_CONN_THRESHOLD
        and ba < PAIR_CONN_THRESHOLD
    )
    return ok, sp, ab, ba


def strongest_outgroup(
    concept: str,
    drops: dict[tuple[str, str], float],
    spearman: dict[frozenset[str], float],
) -> dict[str, Any]:
    candidates = [
        (target, drop)
        for (source, target), drop in drops.items()
        if source == concept and target != concept
    ]
    if not candidates:
        return {
            "max_source_outgroup_conn_damage": "",
            "max_source_outgroup_pair": "",
            "max_source_outgroup_abs_spearman": "",
            "passes_A": "False",
        }
    target, drop = max(candidates, key=lambda item: item[1])
    sp = spearman.get(frozenset([concept, target]), float("nan"))
    return {
        "max_source_outgroup_conn_damage": fmt(drop),
        "max_source_outgroup_pair": f"{concept}->{target}",
        "max_source_outgroup_abs_spearman": fmt(sp),
        "passes_A": str(drop < SOURCE_GATE_A_THRESHOLD),
    }


def discover_source_clean(
    concepts: list[str],
    drops: dict[tuple[str, str], float],
    spearman: dict[frozenset[str], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        gate = strongest_outgroup(concept, drops, spearman)
        if gate["passes_A"] == "True":
            rows.append({"concept": concept, **gate})
    return rows


def find_cliques(
    source_clean: list[str],
    drops: dict[tuple[str, str], float],
    spearman: dict[frozenset[str], float],
    min_size: int = 3,
) -> list[tuple[str, ...]]:
    cliques: list[tuple[str, ...]] = []
    max_size = min(6, len(source_clean))
    for size in range(max_size, min_size - 1, -1):
        for combo in itertools.combinations(source_clean, size):
            if all(pair_gate(a, b, drops, spearman)[0] for a, b in itertools.combinations(combo, 2)):
                cliques.append(combo)
    return cliques


def clique_summary(
    clique: tuple[str, ...],
    drops: dict[tuple[str, str], float],
    spearman: dict[frozenset[str], float],
    families: dict[str, str],
) -> dict[str, Any]:
    max_sp = 0.0
    max_conn = 0.0
    pairs = []
    family_count: dict[str, int] = {}
    for concept in clique:
        family_count[families.get(concept, "")] = family_count.get(families.get(concept, ""), 0) + 1
    for a, b in itertools.combinations(clique, 2):
        _, sp, ab, ba = pair_gate(a, b, drops, spearman)
        max_sp = max(max_sp, sp)
        max_conn = max(max_conn, ab, ba)
        pairs.append(f"{a}<->{b}")
    return {
        "clique_size": len(clique),
        "concepts": ";".join(clique),
        "families": ";".join(f"{k}:{v}" for k, v in sorted(family_count.items())),
        "max_pairwise_spearman": fmt(max_sp),
        "max_pairwise_conn_damage": fmt(max_conn),
        "pair_count": len(pairs),
        "pairs": "|".join(pairs),
    }


def load_b_gate_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    sources = [
        (
            SIX_MODEL_DIR / "l0clamp_reclassified" / "sae_l0clamp_reclassified_cells.csv",
            "l0clamp_reclassified",
        ),
        (SIX_MODEL_DIR / "l0clamp_summary" / "sae_l0clamp_combined_results.csv", "l0clamp_main"),
        (
            SIX_MODEL_DIR / "l0clamp_sensitivity95_summary" / "sae_l0clamp_combined_results.csv",
            "l0clamp_sensitivity95",
        ),
        (SAE_DIR / "csfm_sae_main_robustness_summary.csv", "csfm_robustness"),
        (SAE_DIR / "csfm_sae_fixed_interpretation_summary.csv", "csfm_fixed"),
    ]
    for path, source_name in sources:
        for row in read_csv(path):
            enriched = dict(row)
            enriched["_source_file"] = str(path.relative_to(ROOT))
            enriched["_source_name"] = source_name
            rows.append(enriched)
    stage4_dir = SIX_MODEL_DIR / "stage4_replacement_b_gate"
    for path in sorted(stage4_dir.glob("task_*_cell_*/sae_layer_per_cell.csv")):
        for row in read_csv(path):
            enriched = dict(row)
            enriched["_source_file"] = str(path.relative_to(ROOT))
            enriched["_source_name"] = "stage4_replacement_b_gate"
            enriched.setdefault("analysis", "stage4_B_gate_only_floor")
            rows.append(enriched)
    return rows


def b_pass(row: dict[str, str]) -> tuple[str, str, float, float, float]:
    a_geo = to_float(row.get("A_geo_cav") or row.get("A_geo_cav_mean") or row.get("A_geo"))
    wbi = to_float(row.get("wbi") or row.get("wbi_median"))
    random_wbi = to_float(row.get("random_wbi_mean") or row.get("random_wbi_seed_mean_mean"))
    target_positive = row.get("target_effect_positive", "True")
    random_target_positive = row.get("random_target_effect_mean_positive", "True")
    ratio_stable = row.get("wbi_ratio_stable", row.get("ratio_stable", "True"))
    diff = wbi - random_wbi
    if not math.isfinite(a_geo):
        return "NA", "missing_A_geo", a_geo, wbi, random_wbi
    if not math.isfinite(wbi) or not math.isfinite(random_wbi):
        return "NA", "missing_WBI_or_random_WBI", a_geo, wbi, random_wbi
    valid_ratio = (
        str(target_positive).lower() == "true"
        and str(random_target_positive).lower() == "true"
        and str(ratio_stable).lower() == "true"
    )
    passes = a_geo >= GATE_B_A_GEO_THRESHOLD and valid_ratio and diff <= GATE_B_WBI_EXCESS_TOL
    reasons = []
    if a_geo < GATE_B_A_GEO_THRESHOLD:
        reasons.append("A_geo_below_0.50")
    if not valid_ratio:
        reasons.append("WBI_ratio_unstable_or_nonpositive_target_effect")
    if diff > GATE_B_WBI_EXCESS_TOL:
        reasons.append("WBI_above_random_by_gt_0.25")
    return str(passes), "pass" if passes else "|".join(reasons), a_geo, wbi, random_wbi


def choose_b_row(concept: str, b_rows: list[dict[str, str]]) -> dict[str, Any]:
    hits = [row for row in b_rows if row.get("concept") == concept]
    if not hits:
        return {
            "B_status": "missing",
            "B_reason": "no_existing_l0clamp_or_anchor_sae_metric",
            "B_model": "",
            "B_task": "",
            "B_layer": "",
            "B_analysis": "",
            "B_A_geo": "",
            "B_WBI": "",
            "B_random_WBI": "",
            "B_wbi_minus_random": "",
            "B_source": "",
        }
    # Prefer main recon-0.90 rows, then sensitivity, then other summaries. Within
    # a source, keep the row with the strongest B-gate evidence.
    def key(row: dict[str, str]) -> tuple[int, float, float]:
        source = row.get("_source_name", "")
        analysis = row.get("analysis", "")
        if source == "l0clamp_reclassified" and analysis == "main_recon_0.90":
            rank = 0
        elif source == "l0clamp_main":
            rank = 1
        elif source == "l0clamp_reclassified" and analysis == "sensitivity_recon_0.95":
            rank = 2
        elif source == "l0clamp_sensitivity95":
            rank = 3
        elif source == "csfm_robustness":
            rank = 4
        elif source == "stage4_replacement_b_gate":
            rank = 4
        else:
            rank = 5
        status, _, a_geo, wbi, random_wbi = b_pass(row)
        diff = wbi - random_wbi if math.isfinite(wbi) and math.isfinite(random_wbi) else float("inf")
        ratio_stable = str(row.get("wbi_ratio_stable", row.get("ratio_stable", "True"))).lower() == "true"
        target_positive = str(row.get("target_effect_positive", "True")).lower() == "true"
        random_target_positive = (
            str(row.get("random_target_effect_mean_positive", "True")).lower() == "true"
        )
        validity_rank = 0 if (ratio_stable and target_positive and random_target_positive) else 1
        status_rank = 0 if status == "True" else 1
        return (rank, status_rank, validity_rank, -a_geo if math.isfinite(a_geo) else 0.0, diff)

    row = sorted(hits, key=key)[0]
    status, reason, a_geo, wbi, random_wbi = b_pass(row)
    diff = wbi - random_wbi if math.isfinite(wbi) and math.isfinite(random_wbi) else float("nan")
    return {
        "B_status": status,
        "B_reason": reason,
        "B_model": row.get("model", ""),
        "B_task": row.get("task", ""),
        "B_layer": row.get("layer", ""),
        "B_analysis": row.get("analysis", row.get("recon_target", "")),
        "B_A_geo": fmt(a_geo),
        "B_WBI": fmt(wbi),
        "B_random_WBI": fmt(random_wbi),
        "B_wbi_minus_random": fmt(diff),
        "B_source": row.get("_source_file", ""),
    }


def build_candidate_rows(
    source_clean_rows: list[dict[str, Any]],
    b_rows: list[dict[str, str]],
    families: dict[str, str],
) -> list[dict[str, Any]]:
    out = []
    for row in source_clean_rows:
        concept = row["concept"]
        b = choose_b_row(concept, b_rows)
        role = "ready_candidate" if b["B_status"] == "True" else "needs_B_gate" if b["B_status"] == "missing" else "fails_B_gate"
        out.append(
            {
                "concept": concept,
                "family": families.get(concept, ""),
                **row,
                **b,
                "candidate_role": role,
            }
        )
    return sorted(out, key=lambda r: (r["candidate_role"], r["family"], r["concept"]))


def ready_pair_rows(
    candidate_rows: list[dict[str, Any]],
    drops: dict[tuple[str, str], float],
    spearman: dict[frozenset[str], float],
    families: dict[str, str],
) -> list[dict[str, Any]]:
    ready = [r["concept"] for r in candidate_rows if r["candidate_role"] == "ready_candidate"]
    rows = []
    for a, b in itertools.combinations(ready, 2):
        ok, sp, ab, ba = pair_gate(a, b, drops, spearman)
        if not ok:
            continue
        rows.append(
            {
                "concept_pair": f"{a};{b}",
                "families": f"{families.get(a, '')};{families.get(b, '')}",
                "spearman": fmt(sp),
                "a_to_b_conn": fmt(ab),
                "b_to_a_conn": fmt(ba),
            }
        )
    return rows


def markdown_report(
    candidate_rows: list[dict[str, Any]],
    clique_rows: list[dict[str, Any]],
    ready_pairs: list[dict[str, Any]],
) -> str:
    ready = [r for r in candidate_rows if r["candidate_role"] == "ready_candidate"]
    missing = [r for r in candidate_rows if r["candidate_role"] == "needs_B_gate"]
    failed = [r for r in candidate_rows if r["candidate_role"] == "fails_B_gate"]
    cand_table = "\n".join(
        [
            (
                f"| {r['concept']} | {r['family']} | {r['max_source_outgroup_conn_damage']} | "
                f"{r['max_source_outgroup_pair']} | {r['B_A_geo']} | {r['B_WBI']} | "
                f"{r['B_random_WBI']} | {r['B_status']} | {r['candidate_role']} |"
            )
            for r in candidate_rows
        ]
    )
    clique_table = "\n".join(
        [
            (
                f"| {r['clique_size']} | {r['concepts']} | {r['families']} | "
                f"{r['max_pairwise_spearman']} | {r['max_pairwise_conn_damage']} |"
            )
            for r in clique_rows[:20]
        ]
    )
    ready_pair_table = "\n".join(
        [
            (
                f"| {r['concept_pair']} | {r['families']} | {r['spearman']} | "
                f"{r['a_to_b_conn']} | {r['b_to_a_conn']} |"
            )
            for r in ready_pairs
        ]
    )
    ready_three_way = [
        r
        for r in clique_rows
        if int(r["clique_size"]) >= 3
        and all(
            next((c for c in candidate_rows if c["concept"] == concept), {}).get("candidate_role")
            == "ready_candidate"
            for concept in str(r["concepts"]).split(";")
        )
    ]
    if ready_three_way:
        decision = (
            "A B-ready pairwise-clean replacement set is available. The strongest "
            f"B-ready clique is `{ready_three_way[0]['concepts']}`."
        )
    elif ready_pairs:
        decision = (
            "No B-ready three-concept replacement set is available yet. The current "
            "B-ready evidence supports only pairwise-clean controls: "
            f"`{'; '.join(r['concept_pair'] for r in ready_pairs)}`. These are useful "
            "as two-concept orthogonality checks, but not as a preregistered "
            "three/four-concept Stage IV control set."
        )
    else:
        decision = (
            "No B-ready pairwise-clean replacement control is available yet. Do not "
            "preregister a replacement positive-control set."
        )
    return f"""# Stage IV Replacement Positive-Control Candidate Search

This file searches for replacements after the original triad failed the Stage IV
cleanliness gate. A replacement positive-control concept must pass source
GATE-A, be pairwise GATE-2 compatible with the other controls, and then pass the
B-gate once anchor SAE metrics are available.

## Source-Clean Candidates

| concept | family | max source outgroup conn | strongest outgroup pair | A_geo | WBI | random WBI | B status | role |
|---|---|---:|---|---:|---:|---:|---|---|
{cand_table}

## Pairwise Orthogonal Cliques Among Source-Clean Concepts

| size | concepts | families | max Spearman | max conn |
|---:|---|---|---:|---:|
{clique_table}

## Pairwise Orthogonal Cliques Among B-Ready Concepts

| concept pair | families | Spearman | a->b conn | b->a conn |
|---|---|---:|---:|---:|
{ready_pair_table}

## Decision

- Ready candidates with existing B-gate pass: `{len(ready)}` ({", ".join(r["concept"] for r in ready) or "none"})
- Candidates needing B-gate runs: `{len(missing)}` ({", ".join(r["concept"] for r in missing) or "none"})
- Candidates failing existing B-gate: `{len(failed)}` ({", ".join(r["concept"] for r in failed) or "none"})

{decision}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthogonal-dir", type=Path, default=ORTHO_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drops, spearman, families = load_coupling(args.orthogonal_dir / "selection_coupling_residual_matrix.csv")
    concepts = sorted(set(families))
    source_clean_rows = discover_source_clean(concepts, drops, spearman)
    source_clean = [r["concept"] for r in source_clean_rows]
    cliques = find_cliques(source_clean, drops, spearman)
    clique_rows = [clique_summary(c, drops, spearman, families) for c in cliques]
    candidate_rows = build_candidate_rows(source_clean_rows, load_b_gate_rows(), families)

    out_dir = args.orthogonal_dir
    write_csv(
        out_dir / "stage4_replacement_candidate_gate.csv",
        candidate_rows,
        [
            "concept",
            "family",
            "max_source_outgroup_conn_damage",
            "max_source_outgroup_pair",
            "max_source_outgroup_abs_spearman",
            "passes_A",
            "B_status",
            "B_reason",
            "B_model",
            "B_task",
            "B_layer",
            "B_analysis",
            "B_A_geo",
            "B_WBI",
            "B_random_WBI",
            "B_wbi_minus_random",
            "B_source",
            "candidate_role",
        ],
    )
    write_csv(
        out_dir / "stage4_replacement_orthogonal_cliques.csv",
        clique_rows,
        ["clique_size", "concepts", "families", "max_pairwise_spearman", "max_pairwise_conn_damage", "pair_count", "pairs"],
    )
    missing = [r for r in candidate_rows if r["candidate_role"] == "needs_B_gate"]
    write_csv(
        out_dir / "stage4_replacement_missing_b_gate_runs.csv",
        [
            {
                "concept": r["concept"],
                "family": r["family"],
                "reason": r["B_reason"],
                "suggested_action": "run_anchor_l0clamp_B_gate_before_preregistration",
            }
            for r in missing
        ],
        ["concept", "family", "reason", "suggested_action"],
    )
    write_text(
        out_dir / "stage4_replacement_candidate_report.md",
        markdown_report(candidate_rows, clique_rows, ready_pair_rows(candidate_rows, drops, spearman, families)),
    )


if __name__ == "__main__":
    main()
