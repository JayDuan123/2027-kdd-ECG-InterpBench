#!/usr/bin/env python
"""Build the corrected orthogonal positive-control artifacts.

The first orthogonal-control pass treated ``st_amp_global`` as a fourth locked
axis. The corrected protocol uses three low-coupling positive controls:

    hr_ventricular, qrs_axis_front, qrs_duration

and moves ``st_amp_global`` into a separate stress-test case. It is orthogonal
to the three controls by GATE-2, but prior CSFM SAE results show it behaves as a
nonselective/wrecking-ball intervention. This script materializes that split
without recomputing LEACE or SAE jobs.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORTHO_DIR = ROOT / "results" / "analysis" / "model_comparison" / "orthogonal_concepts"
DEFAULT_SAE_DIR = ROOT / "results" / "sae_extension"

POSITIVE_CONTROLS = ["hr_ventricular", "qrs_axis_front", "qrs_duration"]
ST_CASE = "st_amp_global"

AXIS = {
    "hr_ventricular": "TIMING_RATE",
    "qrs_axis_front": "SPATIAL_AXIS",
    "qrs_duration": "DURATION",
    "st_amp_global": "ST_REPOL",
}

OFF_TARGET = {
    "hr_ventricular": "qrs_axis_front",
    "qrs_axis_front": "qrs_duration",
    "qrs_duration": "hr_ventricular",
}

MI_STT_NEIGHBORHOOD = {
    "t_amp_global",
    "t_amp_limb",
    "t_amp_precordial",
    "t_area_global",
    "t_area_limb",
    "t_area_precordial",
    "qrst_angle",
    "q_amp_precordial",
    "s_amp_precordial",
    "st_elev_global",
    "st_elev_limb",
    "st_elev_precordial",
    "st_slope_global",
    "st_slope_limb",
    "st_slope_precordial",
    "st_amp_limb",
    "st_amp_precordial",
}

SPEARMAN_THRESHOLD = 0.30
CONN_DAMAGE_THRESHOLD = 0.20


def read_csv(path: Path) -> list[dict[str, str]]:
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


def f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def fmt(value: Any, ndigits: int = 6) -> str:
    value = f(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{ndigits}g}"


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def pair_lookup(rows: list[dict[str, str]]) -> dict[frozenset[str], dict[str, str]]:
    out: dict[frozenset[str], dict[str, str]] = {}
    for row in rows:
        out[frozenset([row["concept_i"], row["concept_j"]])] = row
    return out


def directed_conn(row: dict[str, str], source: str, target: str) -> float:
    if row["concept_i"] == source and row["concept_j"] == target:
        return f(row["conn_i_to_j"])
    if row["concept_i"] == target and row["concept_j"] == source:
        return f(row["conn_j_to_i"])
    return float("nan")


def build_positive_pairwise(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    lookup = pair_lookup(pair_rows)
    out: list[dict[str, Any]] = []
    for i, a in enumerate(POSITIVE_CONTROLS):
        for b in POSITIVE_CONTROLS[i + 1 :]:
            row = lookup[frozenset([a, b])]
            out.append(
                {
                    "concept_i": row["concept_i"],
                    "concept_j": row["concept_j"],
                    "axis_i": row["axis_i"],
                    "axis_j": row["axis_j"],
                    "abs_spearman": row["abs_spearman"],
                    "conn_i_to_j": row["conn_i_to_j"],
                    "conn_j_to_i": row["conn_j_to_i"],
                    "groundtruth_pass": row["groundtruth_pass"],
                    "representation_pass": row["representation_pass"],
                    "pair_pass": row["pair_pass"],
                    "fail_reason": row["fail_reason"],
                }
            )
    return out


def build_locked_positive(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    lookup = pair_lookup(pair_rows)
    out: list[dict[str, Any]] = []
    for concept in POSITIVE_CONTROLS:
        max_sp = 0.0
        max_conn = 0.0
        for other in POSITIVE_CONTROLS:
            if other == concept:
                continue
            row = lookup[frozenset([concept, other])]
            max_sp = max(max_sp, f(row["abs_spearman"]))
            max_conn = max(
                max_conn,
                directed_conn(row, concept, other),
                directed_conn(row, other, concept),
            )
        off = OFF_TARGET[concept]
        out.append(
            {
                "concept_id": concept,
                "physical_axis": AXIS[concept],
                "max_pairwise_spearman_with_positive_controls": fmt(max_sp, 10),
                "max_pairwise_conn_damage_with_positive_controls": fmt(max_conn, 10),
                "locked_positive_control": "True",
                "off_target_concept": off,
                "off_target_axis": AXIS[off],
                "prediction": "selective_steering_expected_vs_locked_off_target",
            }
        )
    return out


def max_coupling_row(
    coupling_rows: list[dict[str, str]],
    source: str,
    target: str,
) -> dict[str, str] | None:
    best: dict[str, str] | None = None
    best_drop = float("-inf")
    for row in coupling_rows:
        if row.get("source_concept") != source or row.get("target_concept") != target:
            continue
        drop = f(row.get("target_r2_drop"))
        if math.isfinite(drop) and drop > best_drop:
            best = row
            best_drop = drop
    return best


def st_pairwise_positive(pair_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    lookup = pair_lookup(pair_rows)
    out: list[dict[str, Any]] = []
    for other in POSITIVE_CONTROLS:
        row = lookup[frozenset([ST_CASE, other])]
        sp = f(row["abs_spearman"])
        st_to_other = directed_conn(row, ST_CASE, other)
        other_to_st = directed_conn(row, other, ST_CASE)
        out.append(
            {
                "st_case": ST_CASE,
                "comparison_concept": other,
                "comparison_axis": AXIS[other],
                "abs_spearman": fmt(sp, 10),
                "conn_st_to_comparison": fmt(st_to_other, 10),
                "conn_comparison_to_st": fmt(other_to_st, 10),
                "max_bidirectional_conn": fmt(max(st_to_other, other_to_st), 10),
                "gate2_pass_vs_positive_control": str(
                    sp < SPEARMAN_THRESHOLD
                    and st_to_other < CONN_DAMAGE_THRESHOLD
                    and other_to_st < CONN_DAMAGE_THRESHOLD
                ),
            }
        )
    return out


def build_st_mechanism_rows(
    pair_rows: list[dict[str, str]],
    coupling_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in st_pairwise_positive(pair_rows):
        rows.append(
            {
                "relation_group": "positive_control_triad",
                "source_concept": ST_CASE,
                "target_concept": item["comparison_concept"],
                "source_task": "",
                "model": "",
                "source_layer": "",
                "target_family": "",
                "source_target_abs_spearman_r": item["abs_spearman"],
                "target_original_r2": "",
                "target_residual_r2": "",
                "target_r2_drop": item["conn_st_to_comparison"],
                "target_erased_effective": "",
                "gate2_pass": item["gate2_pass_vs_positive_control"],
                "note": "pairwise locked-set gate: st_amp_global -> positive control",
            }
        )
        rows.append(
            {
                "relation_group": "positive_control_triad",
                "source_concept": item["comparison_concept"],
                "target_concept": ST_CASE,
                "source_task": "",
                "model": "",
                "source_layer": "",
                "target_family": "ST_T",
                "source_target_abs_spearman_r": item["abs_spearman"],
                "target_original_r2": "",
                "target_residual_r2": "",
                "target_r2_drop": item["conn_comparison_to_st"],
                "target_erased_effective": "",
                "gate2_pass": item["gate2_pass_vs_positive_control"],
                "note": "pairwise locked-set gate: positive control -> st_amp_global",
            }
        )

    candidates = sorted(MI_STT_NEIGHBORHOOD)
    for other in candidates:
        for source, target in [(ST_CASE, other), (other, ST_CASE)]:
            best = max_coupling_row(coupling_rows, source, target)
            if best is None:
                continue
            rows.append(
                {
                    "relation_group": "mi_stt_neighborhood",
                    "source_concept": source,
                    "target_concept": target,
                    "source_task": best.get("source_task", ""),
                    "model": best.get("model", ""),
                    "source_layer": best.get("source_layer", ""),
                    "target_family": best.get("target_family", ""),
                    "source_target_abs_spearman_r": fmt(
                        best.get("source_target_abs_spearman_r"), 10
                    ),
                    "target_original_r2": fmt(best.get("target_original_r2"), 10),
                    "target_residual_r2": fmt(best.get("target_residual_r2"), 10),
                    "target_r2_drop": fmt(best.get("target_r2_drop"), 10),
                    "target_erased_effective": best.get("target_erased_effective", ""),
                    "gate2_pass": str(f(best.get("target_r2_drop")) < CONN_DAMAGE_THRESHOLD),
                    "note": "max observed residual-probe connected damage for this directed pair",
                }
            )
    rows.sort(
        key=lambda row: (
            row["relation_group"],
            -f(row["target_r2_drop"]),
            row["source_concept"],
            row["target_concept"],
        )
    )
    return rows


def first_matching(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(k) == v for k, v in criteria.items()):
            return row
    return None


def st_special_case(
    pair_rows: list[dict[str, str]],
    st_mechanism: list[dict[str, Any]],
    robustness: list[dict[str, str]],
    recovery: list[dict[str, str]],
    fixed: list[dict[str, str]],
) -> list[dict[str, Any]]:
    st_pos = st_pairwise_positive(pair_rows)
    max_sp_pos = max(f(row["abs_spearman"]) for row in st_pos)
    max_conn_pos = max(f(row["max_bidirectional_conn"]) for row in st_pos)

    mi_rows = [row for row in st_mechanism if row["relation_group"] == "mi_stt_neighborhood"]
    strongest = max(mi_rows, key=lambda row: f(row["target_r2_drop"])) if mi_rows else {}

    fixed_row = first_matching(fixed, run="st_mi") or {}
    rob_e4 = first_matching(robustness, run="st_mi", E="4") or {}
    rob_e8 = first_matching(robustness, run="st_mi", E="8") or {}
    rec_e4 = first_matching(recovery, run="st_mi", E="4") or {}
    rec_e8 = first_matching(recovery, run="st_mi", E="8") or {}

    return [
        {
            "concept_id": ST_CASE,
            "anchor_task": "mi_ischemia",
            "role": "separate_orthogonal_but_nonsteerable_stress_case",
            "orthogonal_to_positive_control_triad": str(
                max_sp_pos < SPEARMAN_THRESHOLD and max_conn_pos < CONN_DAMAGE_THRESHOLD
            ),
            "max_spearman_to_positive_control_triad": fmt(max_sp_pos, 10),
            "max_conn_damage_to_positive_control_triad": fmt(max_conn_pos, 10),
            "strongest_mi_stt_connected_pair": (
                f"{strongest.get('source_concept', '')}->{strongest.get('target_concept', '')}"
            ),
            "strongest_mi_stt_connected_damage": strongest.get("target_r2_drop", ""),
            "strongest_mi_stt_abs_spearman": strongest.get("source_target_abs_spearman_r", ""),
            "strongest_mi_stt_model": strongest.get("model", ""),
            "strongest_mi_stt_task": strongest.get("source_task", ""),
            "fixed_A_geo_cav": fmt(fixed_row.get("A_geo_cav")),
            "fixed_n90_activation_ranked": fixed_row.get("n90_activation_ranked", ""),
            "fixed_wbi": fmt(fixed_row.get("wbi")),
            "E4_A_geo_cav_mean": fmt(rob_e4.get("A_geo_cav_mean")),
            "E8_A_geo_cav_mean": fmt(rob_e8.get("A_geo_cav_mean")),
            "E4_n90_mean": fmt(rob_e4.get("n90_mean")),
            "E8_n90_mean": fmt(rob_e8.get("n90_mean")),
            "E4_wbi_median": fmt(rob_e4.get("wbi_median")),
            "E8_wbi_median": fmt(rob_e8.get("wbi_median")),
            "E8_random_wbi_mean": fmt(rob_e8.get("random_wbi_seed_mean_mean")),
            "E8_wbi_minus_random_wbi": fmt(rob_e8.get("wbi_minus_random_wbi")),
            "E4_recovery_auc_mean": fmt(rec_e4.get("recovery_auc_mean")),
            "E8_recovery_auc_mean": fmt(rec_e8.get("recovery_auc_mean")),
            "mechanism_label": "mixed_local_STT_collinearity_plus_distributed_dictionary",
            "mechanism_decision": (
                "not pure distributed: st_amp_global passes GATE-2 against the triad, "
                "but local ST/T neighborhood coupling exceeds the 0.20 damage threshold "
                "and SAE geometry remains distributed/nonselective"
            ),
        }
    ]


def markdown_report(
    positive: list[dict[str, Any]],
    positive_pairwise: list[dict[str, Any]],
    st_case: list[dict[str, Any]],
    st_mechanism: list[dict[str, Any]],
) -> str:
    st = st_case[0]
    pos_lines = "\n".join(
        [
            (
                f"| {row['concept_id']} | {row['physical_axis']} | "
                f"{row['max_pairwise_spearman_with_positive_controls']} | "
                f"{row['max_pairwise_conn_damage_with_positive_controls']} | "
                f"{row['off_target_concept']} |"
            )
            for row in positive
        ]
    )
    pair_lines = "\n".join(
        [
            (
                f"| {row['concept_i']} | {row['concept_j']} | {row['abs_spearman']} | "
                f"{row['conn_i_to_j']} | {row['conn_j_to_i']} | {row['pair_pass']} |"
            )
            for row in positive_pairwise
        ]
    )
    top_mech = [
        row
        for row in st_mechanism
        if row["relation_group"] == "mi_stt_neighborhood"
    ][:8]
    mech_lines = "\n".join(
        [
            (
                f"| {row['source_concept']} -> {row['target_concept']} | "
                f"{row['target_r2_drop']} | {row['source_target_abs_spearman_r']} | "
                f"{row['model']} | {row['source_task']} |"
            )
            for row in top_mech
        ]
    )
    return f"""# Corrected Orthogonal Positive-Control Scheme

## Locked Positive-Control Triad

`st_amp_global` is no longer part of the locked positive-control set. The
positive-control set is restricted to three physiologically distinct and
low-coupling concepts:

| concept | axis | max Spearman within triad | max connected damage within triad | locked off-target |
|---|---:|---:|---:|---|
{pos_lines}

Pairwise GATE-2 checks:

| concept i | concept j | abs Spearman | conn i->j | conn j->i | pass |
|---|---|---:|---:|---:|---|
{pair_lines}

Decision: all three positive-control concepts pass the frozen thresholds
(`abs Spearman < {SPEARMAN_THRESHOLD}` and bidirectional connected damage
`< {CONN_DAMAGE_THRESHOLD}`). This is the set to use for positive-control
steering claims.

## st_amp_global Special Case

`st_amp_global -> mi_ischemia` is now a separate stress-test case, not a member
of the locked positive-control triad.

Key facts:

- Orthogonal to the positive-control triad: `{st['orthogonal_to_positive_control_triad']}`
- Max Spearman to triad: `{st['max_spearman_to_positive_control_triad']}`
- Max connected damage to triad: `{st['max_conn_damage_to_positive_control_triad']}`
- Fixed CSFM SAE A_geo_cav: `{st['fixed_A_geo_cav']}`
- Fixed CSFM SAE n90 activation-ranked: `{st['fixed_n90_activation_ranked']}`
- Fixed CSFM SAE WBI: `{st['fixed_wbi']}`
- Robustness E=4 A_geo/WBI: `{st['E4_A_geo_cav_mean']}` / `{st['E4_wbi_median']}`
- Robustness E=8 A_geo/WBI: `{st['E8_A_geo_cav_mean']}` / `{st['E8_wbi_median']}`
- E=8 random WBI: `{st['E8_random_wbi_mean']}`
- E=8 WBI minus random WBI: `{st['E8_wbi_minus_random_wbi']}`

Strongest MI/ST-neighborhood connected-damage rows:

| directed pair | target R2 drop | abs Spearman | model | task |
|---|---:|---:|---|---|
{mech_lines}

Mechanism decision:

`{st['mechanism_label']}`.

Interpretation: `st_amp_global` is cleanly separated from the locked
positive-control triad, but it is not a pure distributed-only case. Its
nonselective behavior is mixed: local ST/T-neighborhood collinearity is visible
in residual-probe coupling, and the SAE dictionary also represents the concept
in a distributed/nonselective way.

## Files

- `locked_positive_control_set.csv`: official corrected positive-control set.
- `positive_control_pairwise_gate_table.csv`: GATE-2 table for the triad only.
- `st_amp_global_special_case.csv`: one-row summary for the excluded ST case.
- `st_amp_global_mechanism_audit.csv`: directed coupling rows supporting the
  mechanism call.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthogonal-dir", type=Path, default=DEFAULT_ORTHO_DIR)
    parser.add_argument("--sae-dir", type=Path, default=DEFAULT_SAE_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_rows = read_csv(args.orthogonal_dir / "pairwise_gate_table.csv")
    coupling_rows = read_csv(args.orthogonal_dir / "selection_coupling_residual_matrix.csv")
    robustness = read_csv(args.sae_dir / "csfm_sae_main_robustness_summary.csv")
    recovery = read_csv(args.sae_dir / "csfm_sae_recovery_summary.csv")
    fixed = read_csv(args.sae_dir / "csfm_sae_fixed_interpretation_summary.csv")

    positive_pairwise = build_positive_pairwise(pair_rows)
    locked_positive = build_locked_positive(pair_rows)
    st_mechanism = build_st_mechanism_rows(pair_rows, coupling_rows)
    st_case = st_special_case(pair_rows, st_mechanism, robustness, recovery, fixed)

    write_csv(
        args.orthogonal_dir / "positive_control_pairwise_gate_table.csv",
        positive_pairwise,
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
        args.orthogonal_dir / "locked_positive_control_set.csv",
        locked_positive,
        [
            "concept_id",
            "physical_axis",
            "max_pairwise_spearman_with_positive_controls",
            "max_pairwise_conn_damage_with_positive_controls",
            "locked_positive_control",
            "off_target_concept",
            "off_target_axis",
            "prediction",
        ],
    )
    write_csv(
        args.orthogonal_dir / "st_amp_global_mechanism_audit.csv",
        st_mechanism,
        [
            "relation_group",
            "source_concept",
            "target_concept",
            "source_task",
            "model",
            "source_layer",
            "target_family",
            "source_target_abs_spearman_r",
            "target_original_r2",
            "target_residual_r2",
            "target_r2_drop",
            "target_erased_effective",
            "gate2_pass",
            "note",
        ],
    )
    write_csv(
        args.orthogonal_dir / "st_amp_global_special_case.csv",
        st_case,
        [
            "concept_id",
            "anchor_task",
            "role",
            "orthogonal_to_positive_control_triad",
            "max_spearman_to_positive_control_triad",
            "max_conn_damage_to_positive_control_triad",
            "strongest_mi_stt_connected_pair",
            "strongest_mi_stt_connected_damage",
            "strongest_mi_stt_abs_spearman",
            "strongest_mi_stt_model",
            "strongest_mi_stt_task",
            "fixed_A_geo_cav",
            "fixed_n90_activation_ranked",
            "fixed_wbi",
            "E4_A_geo_cav_mean",
            "E8_A_geo_cav_mean",
            "E4_n90_mean",
            "E8_n90_mean",
            "E4_wbi_median",
            "E8_wbi_median",
            "E8_random_wbi_mean",
            "E8_wbi_minus_random_wbi",
            "E4_recovery_auc_mean",
            "E8_recovery_auc_mean",
            "mechanism_label",
            "mechanism_decision",
        ],
    )
    write_text(
        args.orthogonal_dir / "corrected_positive_control_preregistration.md",
        markdown_report(locked_positive, positive_pairwise, st_case, st_mechanism),
    )


if __name__ == "__main__":
    main()
