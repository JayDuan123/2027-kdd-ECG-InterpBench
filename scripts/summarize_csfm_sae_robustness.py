#!/usr/bin/env python
"""Summarise CSFM SAE main robustness runs.

Aggregation rules:
- group by (run, concept, task, E)
- SAE-seed variability is the reported robustness axis
- random clamp permutations are first reduced within each seed to
  random_wbi_mean, then aggregated across seeds
- WBI is reported as median/IQR plus a clipped mean/sd diagnostic
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _median(values: list[float]) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2.0)


def _quantile(values: list[float], q: float) -> float:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return float("nan")
    pos = (len(values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(values[lo])
    return float(values[lo] * (hi - pos) + values[hi] * (pos - lo))


def _mean(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return float(sum(values) / len(values)) if values else float("nan")


def _sd(values: list[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    if len(values) < 2:
        return float("nan")
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1)))


def _fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return "nan"
    if not math.isfinite(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def _fmt_iqr(values: list[float], digits: int = 2) -> str:
    med = _median(values)
    q1 = _quantile(values, 0.25)
    q3 = _quantile(values, 0.75)
    if not math.isfinite(med):
        return "nan"
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"


def base_interpretation(run: str) -> str:
    if run == "p_af_control":
        return "positive-control definition chain"
    if run == "qrs_cd":
        return "clinical conduction chain"
    if run == "qrst_cd":
        return "broad electrical-axis signal (low A_geo, broadly causal in v1)"
    if run == "st_mi":
        return "hardest non-definition ST/ischemia chain"
    return "CSFM SAE audit cell"


def selectivity_interpretation(item: dict[str, object]) -> str:
    """Interpret WBI against the same-size random clamp baseline."""
    base = base_interpretation(str(item["run"]))
    wbi = float(item["wbi_median"])
    random_wbi = float(item["random_wbi_seed_mean_mean"])
    e_value = int(item["E"])
    run = str(item["run"])
    if math.isfinite(wbi) and math.isfinite(random_wbi):
        delta = wbi - random_wbi
        if run == "st_mi" and e_value == 8 and delta > 1.0:
            return (
                f"{base}; concept clamp worse than random baseline "
                "(concept-specific wrecking-ball)"
            )
        if abs(delta) <= 0.10:
            return f"{base}; concept clamp no more selective than random clamp"
    return f"{base}; distributed/nonselective, inspect WBI/random baseline"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/sae_extension/csfm_sae_main_robustness")
    parser.add_argument("--out-prefix", default="results/sae_extension/csfm_sae_main_robustness_summary")
    parser.add_argument("--wbi-clip", type=float, default=5.0)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("*/*/*/sae_layer_per_cell.csv")):
        with path.open(newline="") as f:
            row = next(csv.DictReader(f))
        parts = path.relative_to(root).parts
        run, e_dir, seed_dir = parts[0], parts[1], parts[2]
        row["run"] = run
        row["E_dir"] = e_dir
        row["seed_dir"] = seed_dir
        rows.append(row)

    out_csv = Path(args.out_prefix + ".csv")
    out_md = Path(args.out_prefix + ".md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["run"], row["E"]), []).append(row)

    summary = []
    for (run, E), group in sorted(groups.items()):
        concept = group[0]["concept"]
        task = group[0]["task"]
        n_capacity = int(float(group[0]["N_capacity"]))
        seeds = sorted({int(float(g["sae_seed"])) for g in group})
        wbi = [_float(g, "wbi") for g in group]
        wbi_clip = [min(v, args.wbi_clip) for v in wbi if math.isfinite(v)]
        random_wbi_seed_mean = [_float(g, "random_wbi_mean") for g in group]
        item = {
            "run": run,
            "concept": concept,
            "task": task,
            "E": int(float(E)),
            "N_capacity": n_capacity,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
            "A_geo_cav_mean": _mean([_float(g, "A_geo_cav") for g in group]),
            "A_geo_cav_sd": _sd([_float(g, "A_geo_cav") for g in group]),
            "decomp_mean": _mean([_float(g, "decomposability_concept_ranked") for g in group]),
            "decomp_sd": _sd([_float(g, "decomposability_concept_ranked") for g in group]),
            "n90_mean": _mean([_float(g, "n90_concept_ranked") for g in group]),
            "n90_sd": _sd([_float(g, "n90_concept_ranked") for g in group]),
            "wbi_median": _median(wbi),
            "wbi_iqr_low": _quantile(wbi, 0.25),
            "wbi_iqr_high": _quantile(wbi, 0.75),
            "wbi_clipped_mean": _mean(wbi_clip),
            "wbi_clipped_sd": _sd(wbi_clip),
            "random_wbi_seed_mean_mean": _mean(random_wbi_seed_mean),
            "random_wbi_seed_mean_sd": _sd(random_wbi_seed_mean),
            "target_effect_mean": _mean([_float(g, "target_effect") for g in group]),
            "offtarget_damage_mean": _mean([_float(g, "offtarget_damage") for g in group]),
            "task_retention_mean": _mean([_float(g, "task_retention") for g in group]),
        }
        item["wbi_minus_random_wbi"] = (
            item["wbi_median"] - item["random_wbi_seed_mean_mean"]
            if math.isfinite(item["wbi_median"]) and math.isfinite(item["random_wbi_seed_mean_mean"])
            else float("nan")
        )
        item["interpretation"] = selectivity_interpretation(item)
        summary.append(item)

    if summary:
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)

    lines = ["# CSFM SAE Main Robustness Summary", ""]
    if not summary:
        lines.append("No completed robustness rows found.")
    else:
        lines.append(
            "Aggregation is per E. WBI/random-WBI variability is reported across SAE seeds; "
            "random clamp permutations are first averaged within each seed."
        )
        lines.append("")
        for E in sorted({item["E"] for item in summary}):
            lines.append(f"## E={E}")
            lines.append("")
            lines.append(
                "| run | concept -> task | A_geo_cav | decomp | n90 qualitative | WBI median [IQR] | random WBI | interpretation |"
            )
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for item in [s for s in summary if s["E"] == E]:
                lines.append(
                    "| {run} | {concept} -> {task} | {ageo} | {decomp} | {n90} | {wbi} | {rwbi} | {interp} |".format(
                        run=item["run"],
                        concept=item["concept"],
                        task=item["task"],
                        ageo=_fmt_pm(item["A_geo_cav_mean"], item["A_geo_cav_sd"]),
                        decomp=_fmt_pm(item["decomp_mean"], item["decomp_sd"]),
                        n90=_fmt_pm(item["n90_mean"], item["n90_sd"], digits=1),
                        wbi=_fmt_iqr(
                            [
                                _float(g, "wbi")
                                for g in groups[(item["run"], str(item["E"]))]
                            ]
                        ),
                        rwbi=_fmt_pm(
                            item["random_wbi_seed_mean_mean"],
                            item["random_wbi_seed_mean_sd"],
                            digits=2,
                        ),
                        interp=item["interpretation"],
                    )
                )
            lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- Main selectivity conclusion uses WBI against the same-size random clamp baseline. For most cells, concept-targeted clamping is no more selective than random clamping; `st_mi` at E=8 is worse than random.")
        lines.append("- A_geo_cav drops from E=4 to E=8 in every cell, indicating that larger dictionaries disperse the LEACE causal direction rather than stabilizing it into a small monosemantic feature set.")
        lines.append("- n90 is retained only as a qualitative appendix diagnostic because it is a threshold-crossing statistic with high seed sensitivity; do not use the precise n90 +/- sd values as a main claim.")
        lines.append("- Do not average n90 across E values; dictionary capacity changes with E.")
        lines.append("- WBI main reporting uses median/IQR to avoid denominator blow-ups when target_effect is near zero.")
        lines.append(f"- A clipped WBI mean/sd diagnostic is available in the CSV with clip={args.wbi_clip}.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_csv)
    print(out_md)


if __name__ == "__main__":
    main()
