#!/usr/bin/env python
"""Select and preregister an orthogonal ECG concept control set.

This implements the locked "Orthogonal Concept Selection" protocol. The
selection uses only input-side evidence:

- PTB-XL+ ground-truth Spearman concept correlations.
- Representation connected damage from LEACE residual-probe coupling.
- A fixed physiological axis map and fixed axis backup order.

It intentionally does not read SAE steering/selectivity/WBI outputs.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CLEANUP = ROOT / "results" / "analysis" / "model_comparison" / "cleanup_audit"
DEFAULT_OUT = ROOT / "results" / "analysis" / "model_comparison" / "orthogonal_concepts"

SPEARMAN_THRESHOLD = 0.30
CONN_DAMAGE_THRESHOLD = 0.20

# Fixed, a-priori axis order and backup order. ST/REPOL is the optional fifth
# axis allowed by the protocol, included here before any steering run.
AXIS_BACKUPS: "OrderedDict[str, list[str]]" = OrderedDict(
    [
        ("TIMING_RATE", ["hr_ventricular", "rr_mean"]),
        ("SPATIAL_AXIS", ["qrs_axis_front", "t_axis_front", "p_axis_front"]),
        ("DURATION", ["qrs_duration", "pr_interval", "p_duration_global"]),
        ("AMPLITUDE", ["t_amp_global", "p_amp_global"]),
        ("ST_REPOL", ["st_amp_global"]),
    ]
)

# Fixed tie-breaker for repair. Later/less central control axes are dropped
# first when failed-pair counts tie. This is deterministic and not correlation-
# ranked.
DROP_TIE_ORDER = {
    "AMPLITUDE": 0,
    "SPATIAL_AXIS": 1,
    "ST_REPOL": 2,
    "DURATION": 3,
    "TIMING_RATE": 4,
}


@dataclass
class PairGate:
    concept_i: str
    concept_j: str
    axis_i: str
    axis_j: str
    abs_spearman: float
    conn_i_to_j: float
    conn_j_to_i: float
    groundtruth_pass: bool
    representation_pass: bool
    pair_pass: bool
    fail_reason: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_coupling_module():
    path = ROOT / "scripts" / "make_concept_coupling_audit.py"
    spec = importlib.util.spec_from_file_location("make_concept_coupling_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def concept_axis(concept_id: str, family: str) -> str:
    for axis, concepts in AXIS_BACKUPS.items():
        if concept_id in concepts:
            return axis
    if concept_id in {"qrst_angle", "qrs_axis_front", "p_axis_front", "t_axis_front"}:
        return "SPATIAL_AXIS"
    if family == "RATE_RHYTHM":
        return "TIMING_RATE"
    if family == "AXIS":
        return "SPATIAL_AXIS"
    if family == "INTERVAL" or "duration" in concept_id or "interval" in concept_id:
        return "DURATION"
    if concept_id.startswith("st_") or concept_id.startswith("t_area") or concept_id == "t_duration_global":
        return "ST_REPOL"
    if family == "AMPLITUDE" or "_amp_" in concept_id or "_area_" in concept_id:
        return "AMPLITUDE"
    return "UNASSIGNED"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def ffmt(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.10g}"


def build_selection_coupling(args: argparse.Namespace, concepts: pd.DataFrame) -> pd.DataFrame:
    """Compute/append residual-probe rows for the fixed selection pool."""
    existing = pd.read_csv(args.base_coupling)
    if not args.refresh_coupling:
        return existing

    module = load_coupling_module()
    family_by_concept = concepts.set_index("concept_id")["family"].to_dict()
    concept_values = pd.read_csv(module.MANIFEST / "concepts_matrix.csv")
    concept_values = concept_values[[c for c in concept_values.columns if c != "ecg_id"]].apply(
        pd.to_numeric, errors="coerce"
    )
    corr = concept_values.corr(method="spearman", min_periods=100)
    corr_map = {(a, b): corr.loc[a, b] for a in corr.index for b in corr.columns}

    canonical = pd.read_csv(CLEANUP / "continuation_canonical_strict_fdr.csv")
    confirmed = canonical[canonical["canonical_confirmed"].map(truthy)].copy()
    confirmed = confirmed[confirmed["eraser_method"] == "leace"].copy()

    pool = sorted({concept for concepts_i in AXIS_BACKUPS.values() for concept in concepts_i})
    source_records = confirmed[confirmed["concept_id"].isin(pool)].copy()
    if source_records.empty:
        raise RuntimeError("no confirmed LEACE source records available for orthogonal selection pool")

    jobs = [
        (source, pool, family_by_concept, corr_map)
        for source in source_records.to_dict("records")
    ]
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for job in jobs:
            rows.extend(module.process_source_record(job))
    else:
        with Pool(processes=args.workers) as pool_exec:
            for part in pool_exec.imap_unordered(module.process_source_record, jobs):
                rows.extend(part)
    computed = pd.DataFrame(rows)
    if computed.empty:
        return existing
    combined = pd.concat([existing, computed], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[
            "model",
            "suffix",
            "source_concept",
            "source_task",
            "source_layer",
            "target_concept",
        ],
        keep="last",
    )
    return combined


def aggregate_conn_damage(coupling: pd.DataFrame) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for (source, target), part in coupling.groupby(["source_concept", "target_concept"]):
        vals = pd.to_numeric(part["target_r2_drop"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[(str(source), str(target))] = float(np.max(vals))
    return out


def load_spearman(path: Path) -> dict[tuple[str, str], float]:
    matrix = pd.read_csv(path).set_index("concept_id")
    out: dict[tuple[str, str], float] = {}
    for source in matrix.index:
        for target in matrix.columns:
            value = pd.to_numeric(pd.Series([matrix.loc[source, target]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                out[(str(source), str(target))] = float(value)
    return out


def gate_pair(
    a: str,
    b: str,
    axis_by_concept: dict[str, str],
    spearman: dict[tuple[str, str], float],
    conn: dict[tuple[str, str], float],
) -> PairGate:
    sp = abs(spearman.get((a, b), float("nan")))
    ab = conn.get((a, b), float("nan"))
    ba = conn.get((b, a), float("nan"))
    groundtruth_pass = bool(np.isfinite(sp) and sp < SPEARMAN_THRESHOLD)
    representation_pass = bool(
        np.isfinite(ab)
        and np.isfinite(ba)
        and ab < CONN_DAMAGE_THRESHOLD
        and ba < CONN_DAMAGE_THRESHOLD
    )
    reasons = []
    if not np.isfinite(sp):
        reasons.append("missing_spearman")
    elif not groundtruth_pass:
        reasons.append("spearman_ge_0.30")
    if not (np.isfinite(ab) and np.isfinite(ba)):
        reasons.append("missing_conn_damage")
    elif not representation_pass:
        reasons.append("conn_damage_ge_0.20")
    pair_pass = groundtruth_pass and representation_pass
    return PairGate(
        concept_i=a,
        concept_j=b,
        axis_i=axis_by_concept[a],
        axis_j=axis_by_concept[b],
        abs_spearman=sp,
        conn_i_to_j=ab,
        conn_j_to_i=ba,
        groundtruth_pass=groundtruth_pass,
        representation_pass=representation_pass,
        pair_pass=pair_pass,
        fail_reason="pass" if pair_pass else "|".join(reasons),
    )


def gate_candidate(
    candidate: OrderedDict[str, str],
    axis_by_concept: dict[str, str],
    spearman: dict[tuple[str, str], float],
    conn: dict[tuple[str, str], float],
) -> list[PairGate]:
    concepts = list(candidate.values())
    out: list[PairGate] = []
    for i in range(len(concepts)):
        for j in range(i + 1, len(concepts)):
            out.append(gate_pair(concepts[i], concepts[j], axis_by_concept, spearman, conn))
    return out


def select_locked_set(
    axis_by_concept: dict[str, str],
    spearman: dict[tuple[str, str], float],
    conn: dict[tuple[str, str], float],
) -> tuple[OrderedDict[str, str], list[dict[str, Any]], list[PairGate]]:
    indices = {axis: 0 for axis in AXIS_BACKUPS}
    active_axes = list(AXIS_BACKUPS.keys())
    candidate: "OrderedDict[str, str]" = OrderedDict(
        (axis, AXIS_BACKUPS[axis][0]) for axis in active_axes
    )
    trace: list[dict[str, Any]] = []
    iteration = 0
    while True:
        gates = gate_candidate(candidate, axis_by_concept, spearman, conn)
        failed = [gate for gate in gates if not gate.pair_pass]
        trace.append(
            {
                "iteration": iteration,
                "candidate": "|".join(f"{axis}:{concept}" for axis, concept in candidate.items()),
                "n_concepts": len(candidate),
                "n_pairs": len(gates),
                "n_failed_pairs": len(failed),
                "action": "lock" if not failed else "repair",
            }
        )
        if not failed or len(candidate) <= 2:
            return candidate, trace, gates

        fail_counts = {axis: 0 for axis in candidate}
        missing_counts = {axis: 0 for axis in candidate}
        for gate in failed:
            fail_counts[gate.axis_i] += 1
            fail_counts[gate.axis_j] += 1
            if "missing_conn_damage" in gate.fail_reason:
                missing_counts[gate.axis_i] += 1
                missing_counts[gate.axis_j] += 1
        drop_axis = max(
            candidate.keys(),
            key=lambda axis: (
                fail_counts[axis],
                missing_counts[axis],
                -DROP_TIE_ORDER.get(axis, 99),
            ),
        )
        old_concept = candidate[drop_axis]
        indices[drop_axis] += 1
        if indices[drop_axis] < len(AXIS_BACKUPS[drop_axis]):
            candidate[drop_axis] = AXIS_BACKUPS[drop_axis][indices[drop_axis]]
            repair = f"replace {drop_axis}:{old_concept}->{candidate[drop_axis]}"
        else:
            del candidate[drop_axis]
            repair = f"drop exhausted axis {drop_axis}:{old_concept}"
        trace[-1]["repair_axis"] = drop_axis
        trace[-1]["repair_detail"] = repair
        trace[-1]["fail_counts"] = json.dumps(fail_counts, sort_keys=True)
        trace[-1]["missing_counts"] = json.dumps(missing_counts, sort_keys=True)
        iteration += 1


def pair_gate_rows(gates: list[PairGate]) -> list[dict[str, Any]]:
    return [
        {
            "concept_i": gate.concept_i,
            "concept_j": gate.concept_j,
            "axis_i": gate.axis_i,
            "axis_j": gate.axis_j,
            "abs_spearman": ffmt(gate.abs_spearman),
            "conn_i_to_j": ffmt(gate.conn_i_to_j),
            "conn_j_to_i": ffmt(gate.conn_j_to_i),
            "groundtruth_pass": gate.groundtruth_pass,
            "representation_pass": gate.representation_pass,
            "pair_pass": gate.pair_pass,
            "fail_reason": gate.fail_reason,
        }
        for gate in gates
    ]


def locked_rows(
    locked: OrderedDict[str, str],
    final_gates: list[PairGate],
) -> list[dict[str, Any]]:
    pair_by_concept: dict[str, list[PairGate]] = {concept: [] for concept in locked.values()}
    for gate in final_gates:
        pair_by_concept[gate.concept_i].append(gate)
        pair_by_concept[gate.concept_j].append(gate)
    axis_items = list(locked.items())
    rows = []
    for idx, (axis, concept) in enumerate(axis_items):
        pairs = pair_by_concept.get(concept, [])
        max_sp = max([gate.abs_spearman for gate in pairs], default=float("nan"))
        max_conn = max(
            [
                max(gate.conn_i_to_j, gate.conn_j_to_i)
                for gate in pairs
                if np.isfinite(gate.conn_i_to_j) and np.isfinite(gate.conn_j_to_i)
            ],
            default=float("nan"),
        )
        off_axis, off_concept = axis_items[(idx + 1) % len(axis_items)]
        rows.append(
            {
                "concept_id": concept,
                "physical_axis": axis,
                "max_pairwise_spearman": ffmt(max_sp),
                "max_pairwise_conn_damage": ffmt(max_conn),
                "locked": True,
                "off_target_concept": off_concept,
                "off_target_axis": off_axis,
                "prediction": "selective_steering_expected_vs_locked_off_target",
            }
        )
    return rows


def render_preregistration(
    locked: OrderedDict[str, str],
    locked_table: list[dict[str, Any]],
    final_gates: list[PairGate],
    trace: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    generated = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Orthogonal Concept Positive-Control Preregistration",
        "",
        f"- generated_utc: `{generated}`",
        "- evidence used: PTB-XL+ Spearman matrix, LEACE representation connected-damage matrix, fixed physiological axis map",
        "- evidence explicitly not used: SAE steering, selectivity, ExcessSelectivity, WBI, or any downstream steering result",
        f"- ground-truth gate: abs(Spearman) < {SPEARMAN_THRESHOLD}",
        f"- representation gate: conn_damage both directions < {CONN_DAMAGE_THRESHOLD}",
        "- representation aggregation: max target_r2_drop over available source model/task/layer rows",
        "- optional ST/REPOL axis: included a priori as the fifth physiological axis before any steering run",
        "",
        "## Locked Set",
        "",
        "| Concept | Axis | Off-target | Max abs Spearman | Max conn damage |",
        "|---|---|---|---:|---:|",
    ]
    for row in locked_table:
        lines.append(
            f"| {row['concept_id']} | {row['physical_axis']} | {row['off_target_concept']} | "
            f"{row['max_pairwise_spearman']} | {row['max_pairwise_conn_damage']} |"
        )
    lines.extend(
        [
            "",
            "## Final Pairwise Gates",
            "",
            "| Pair | Spearman | Conn i->j | Conn j->i | Pass | Reason |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for gate in final_gates:
        lines.append(
            f"| {gate.concept_i} / {gate.concept_j} | {ffmt(gate.abs_spearman)} | "
            f"{ffmt(gate.conn_i_to_j)} | {ffmt(gate.conn_j_to_i)} | {gate.pair_pass} | {gate.fail_reason} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Steering Prediction",
            "",
            "For each locked concept, the preregistered off-target is the next locked concept in fixed axis order. "
            "The positive-control prediction is that SAE steering should be selective relative to that locked off-target. "
            "If steering is still nonselective on this orthogonal set, the failure cannot be attributed only to clinical concept collinearity.",
            "",
            "## Selection Trace",
            "",
        ]
    )
    for step in trace:
        lines.append(
            f"- iteration {step['iteration']}: {step['candidate']} -> {step['action']}"
            + (f" ({step.get('repair_detail')})" if step.get("repair_detail") else "")
        )
    lines.extend(
        [
            "",
            "## Source Files",
            "",
            f"- spearman: `{args.spearman}`",
            f"- base coupling: `{args.base_coupling}`",
            f"- output coupling: `{args.out_dir / 'selection_coupling_residual_matrix.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select preregistered orthogonal ECG concepts.")
    parser.add_argument("--concepts", type=Path, default=ROOT / "configs" / "concepts.csv")
    parser.add_argument("--spearman", type=Path, default=CLEANUP / "concept_spearman_correlation_matrix.csv")
    parser.add_argument("--base-coupling", type=Path, default=CLEANUP / "concept_coupling_residual_matrix.csv")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--refresh-coupling", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    concepts = pd.read_csv(args.concepts)
    concepts = concepts[concepts["main"].astype(str).str.lower() == "yes"].copy()
    concepts["physical_axis"] = [
        concept_axis(str(row.concept_id), str(row.family))
        for row in concepts.itertuples(index=False)
    ]
    axis_by_concept = concepts.set_index("concept_id")["physical_axis"].to_dict()

    coupling = build_selection_coupling(args, concepts)
    coupling.to_csv(args.out_dir / "selection_coupling_residual_matrix.csv", index=False)
    conn = aggregate_conn_damage(coupling)
    spearman = load_spearman(args.spearman)

    locked, trace, final_gates = select_locked_set(axis_by_concept, spearman, conn)
    locked_table = locked_rows(locked, final_gates)

    axis_rows = [
        {
            "concept_id": row.concept_id,
            "family": row.family,
            "physical_axis": row.physical_axis,
            "selection_pool_rank": (
                AXIS_BACKUPS[row.physical_axis].index(row.concept_id)
                if row.physical_axis in AXIS_BACKUPS and row.concept_id in AXIS_BACKUPS[row.physical_axis]
                else ""
            ),
        }
        for row in concepts.itertuples(index=False)
    ]
    write_csv(
        args.out_dir / "physical_axis_map.csv",
        axis_rows,
        ["concept_id", "family", "physical_axis", "selection_pool_rank"],
    )
    write_csv(
        args.out_dir / "pairwise_gate_table.csv",
        pair_gate_rows(final_gates),
        [
            "concept_i",
            "concept_j",
            "axis_i",
            "axis_j",
            "abs_spearman",
            "conn_i_to_j",
            "conn_j_to_i",
            "groundtruth_pass",
            "representation_pass",
            "pair_pass",
            "fail_reason",
        ],
    )
    write_csv(
        args.out_dir / "selection_trace.csv",
        trace,
        [
            "iteration",
            "candidate",
            "n_concepts",
            "n_pairs",
            "n_failed_pairs",
            "action",
            "repair_axis",
            "repair_detail",
            "fail_counts",
            "missing_counts",
        ],
    )
    write_csv(
        args.out_dir / "locked_orthogonal_set.csv",
        locked_table,
        [
            "concept_id",
            "physical_axis",
            "max_pairwise_spearman",
            "max_pairwise_conn_damage",
            "locked",
            "off_target_concept",
            "off_target_axis",
            "prediction",
        ],
    )
    (args.out_dir / "orthogonal_concept_preregistration.md").write_text(
        render_preregistration(locked, locked_table, final_gates, trace, args),
        encoding="utf-8",
    )
    print(f"locked {len(locked_table)} concepts to {args.out_dir / 'locked_orthogonal_set.csv'}")


if __name__ == "__main__":
    main()
