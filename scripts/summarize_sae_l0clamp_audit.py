#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def collect_result_rows(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for root in paths:
        for csv_path in sorted(root.glob("**/sae_layer_per_cell.csv")):
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df = df.copy()
            df["result_csv"] = str(csv_path)
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def expected_rows(csfm_manifest: Path, transformer_manifest: Path) -> pd.DataFrame:
    rows = []
    csfm_grid = [0, 16, 23, 3, 15, 5, 20, 4, 11, 9, 6]
    cells = pd.read_csv(csfm_manifest)
    for cell_index in csfm_grid:
        row = cells.iloc[int(cell_index)]
        concept, rest = str(row["candidate"]).split("->", 1)
        task, layer = rest.rsplit("@L", 1)
        rows.append(
            {
                "source": "csfm",
                "cell_index": int(cell_index),
                "model": row["model"],
                "concept": concept,
                "task": task,
                "layer": int(layer),
            }
        )
    for row in read_csv_rows(transformer_manifest):
        rows.append(
            {
                "source": "transformer",
                "cell_index": int(row["cell_index"]),
                "model": row["model"],
                "concept": row["concept"],
                "task": row["task"],
                "layer": int(float(row["layer"])),
            }
        )
    return pd.DataFrame(rows)


def write_summary(results: pd.DataFrame, expected: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "sae_l0clamp_combined_results.csv"
    expected_path = out_dir / "sae_l0clamp_expected_cells.csv"
    missing_path = out_dir / "sae_l0clamp_missing_cells.csv"
    summary_path = out_dir / "sae_l0clamp_summary.md"

    expected.to_csv(expected_path, index=False)
    if results.empty:
        missing = expected.copy()
        missing.to_csv(missing_path, index=False)
        combined_path.write_text("", encoding="utf-8")
        summary_path.write_text(
            "# SAE L0-Clamp Audit Summary\n\n"
            f"- completed_cells: 0\n"
            f"- expected_cells: {len(expected)}\n"
            f"- missing_cells: {len(missing)}\n",
            encoding="utf-8",
        )
        return

    key_cols = ["model", "concept", "task", "layer"]
    for col in key_cols:
        if col not in results.columns:
            raise ValueError(f"missing result column: {col}")
    results["layer"] = results["layer"].astype(int)
    if "target_effect" in results.columns:
        results["target_effect_positive"] = pd.to_numeric(
            results["target_effect"], errors="coerce"
        ) > 0.0
    if "random_target_effect_mean" in results.columns:
        results["random_target_effect_mean_positive"] = pd.to_numeric(
            results["random_target_effect_mean"], errors="coerce"
        ) > 0.0
    if {"target_effect_positive", "random_target_effect_mean_positive"}.issubset(results.columns):
        results["wbi_ratio_stable"] = (
            results["target_effect_positive"] & results["random_target_effect_mean_positive"]
        )
    results.to_csv(combined_path, index=False)

    done_keys = set(map(tuple, results[key_cols].astype(str).to_numpy()))
    exp = expected.copy()
    exp_key = exp[key_cols].astype(str).apply(tuple, axis=1)
    missing = exp.loc[~exp_key.isin(done_keys)].copy()
    missing.to_csv(missing_path, index=False)

    completed = len(results)
    expected_n = len(expected)
    in_band = int((results.get("matched_tier", pd.Series(dtype=object)) == "in_band").sum())
    completed_status = results.get("steering_status", pd.Series(dtype=object)).fillna("")
    completed_steering = int((completed_status == "completed").sum())
    pass_col = results.get("steering_pass", pd.Series(dtype=object)).fillna(False)
    passes = int(pass_col.astype(bool).sum()) if len(pass_col) else 0
    ratio_stable = (
        int(results["wbi_ratio_stable"].sum())
        if "wbi_ratio_stable" in results.columns
        else 0
    )
    by_model = (
        results.groupby("model")
        .agg(
            completed_cells=("model", "size"),
            in_band_cells=("matched_tier", lambda x: int((x == "in_band").sum())),
            steering_completed=("steering_status", lambda x: int((x == "completed").sum())),
            passing_cells=("steering_pass", lambda x: int(pd.Series(x).fillna(False).astype(bool).sum())),
            wbi_ratio_stable_cells=(
                "wbi_ratio_stable",
                lambda x: int(pd.Series(x).fillna(False).astype(bool).sum()),
            )
            if "wbi_ratio_stable" in results.columns
            else ("model", lambda x: 0),
        )
        .reset_index()
    )

    lines = [
        "# SAE L0-Clamp Audit Summary",
        "",
        f"- expected_cells: {expected_n}",
        f"- completed_cells: {completed}",
        f"- missing_cells: {len(missing)}",
        f"- in_band_completed_cells: {in_band}",
        f"- steering_completed_cells: {completed_steering}",
        f"- passing_cells: {passes}",
        f"- wbi_ratio_stable_cells: {ratio_stable}",
        "- WBI ratio summaries are treated as unstable when concept or random target effect is non-positive; such cells cannot support a positive steering claim.",
        "",
        "## By Model",
        "",
        markdown_table(by_model),
    ]
    if len(missing):
        lines.extend(["", "## Missing Cells", "", markdown_table(missing)])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csfm-root",
        type=Path,
        default=Path("results/sae_extension/six_model_sae_audit/csfm_steering_main_l0clamp"),
    )
    parser.add_argument(
        "--transformer-root",
        type=Path,
        default=Path("results/sae_extension/six_model_sae_audit/transformer_steering_main_l0clamp"),
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=Path("results/sae_extension/six_model_sae_audit/phase0_low_coupling_cells.csv"),
    )
    parser.add_argument(
        "--transformer-manifest",
        type=Path,
        default=Path("results/sae_extension/six_model_sae_audit/phase0_selected_transformer_operating_points_l0clamp.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/sae_extension/six_model_sae_audit/l0clamp_summary"),
    )
    args = parser.parse_args()

    results = collect_result_rows([args.csfm_root, args.transformer_root])
    expected = expected_rows(args.cells, args.transformer_manifest)
    write_summary(results, expected, args.out_dir)
    print(f"wrote summary to {args.out_dir}")


if __name__ == "__main__":
    main()
