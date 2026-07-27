#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODEL_COMPARISON = ROOT / "results" / "analysis" / "model_comparison"
FIGURE_DIR = ROOT / "results" / "figures"
MODELS = ["CSFM", "ECG-FM", "ECG-JEPA", "ST-MEM", "HuBERT-ECG", "CARDIAC-FM"]
TASKS = [
    "ptbxl_norm",
    "ptbxl_mi",
    "ptbxl_sttc",
    "ptbxl_cd",
    "ptbxl_hyp",
    "mi_ischemia",
    "bbb_conduction",
    "hypertrophy",
    "af_rhythm",
]
FAMILIES = ["RATE_RHYTHM", "INTERVAL", "AXIS", "AMPLITUDE", "ST_T"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def f(value: str) -> float:
    return float(value) if value not in {"", "nan", "None"} else float("nan")


def fmt(value: float) -> str:
    return "" if value != value else f"{value:.8g}"


def summary_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    task_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    family_task_rows: list[dict[str, object]] = []

    for model in MODELS:
        for task in TASKS:
            subset = [r for r in rows if r["model"] == model and r["task_id"] == task]
            confirmed = [r for r in subset if r["candidate_status"] == "confirmed_screen"]
            vals = [f(r["delta_auroc_minus_random"]) for r in confirmed]
            task_rows.append(
                {
                    "model": model,
                    "task_id": task,
                    "tested_count": len(subset),
                    "confirmed_count": len(confirmed),
                    "max_confirmed_adj_drop": fmt(max(vals) if vals else float("nan")),
                    "sum_confirmed_adj_drop": fmt(sum(vals) if vals else float("nan")),
                }
            )

        for family in FAMILIES:
            subset = [r for r in rows if r["model"] == model and r["family"] == family]
            confirmed = [r for r in subset if r["candidate_status"] == "confirmed_screen"]
            vals = [f(r["delta_auroc_minus_random"]) for r in confirmed]
            family_rows.append(
                {
                    "model": model,
                    "family": family,
                    "tested_count": len(subset),
                    "confirmed_count": len(confirmed),
                    "max_confirmed_adj_drop": fmt(max(vals) if vals else float("nan")),
                    "mean_confirmed_adj_drop": fmt(statistics.mean(vals) if vals else float("nan")),
                    "sum_confirmed_adj_drop": fmt(sum(vals) if vals else float("nan")),
                }
            )

            for task in TASKS:
                subset_ft = [r for r in subset if r["task_id"] == task]
                confirmed_ft = [r for r in subset_ft if r["candidate_status"] == "confirmed_screen"]
                vals_ft = [f(r["delta_auroc_minus_random"]) for r in confirmed_ft]
                family_task_rows.append(
                    {
                        "model": model,
                        "family": family,
                        "task_id": task,
                        "tested_count": len(subset_ft),
                        "confirmed_count": len(confirmed_ft),
                        "max_confirmed_adj_drop": fmt(max(vals_ft) if vals_ft else float("nan")),
                        "sum_confirmed_adj_drop": fmt(sum(vals_ft) if vals_ft else float("nan")),
                    }
                )

    return task_rows, family_rows, family_task_rows


def matrix(rows: list[dict[str, object]], row_key: str, col_key: str, value_key: str, row_order: list[str], col_order: list[str]):
    import numpy as np

    values = {(str(r[row_key]), str(r[col_key])): r[value_key] for r in rows}
    out = np.zeros((len(row_order), len(col_order)), dtype=float)
    for i, row in enumerate(row_order):
        for j, col in enumerate(col_order):
            val = values.get((row, col), "")
            out[i, j] = float(val) if val not in {"", None} else 0.0
    return out


def draw_heatmap(ax, data, row_labels, col_labels, title, cmap, vmin=None, vmax=None, integer=False):
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.tick_params(length=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if val <= 0:
                text = ""
            elif integer:
                text = str(int(val))
            else:
                text = f"{val:.3f}"
            color = "white" if val > (vmax or data.max()) * 0.55 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    return im


def make_figures(task_rows, family_rows, family_task_rows) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    task_drop = matrix(task_rows, "model", "task_id", "max_confirmed_adj_drop", MODELS, TASKS)
    task_count = matrix(task_rows, "model", "task_id", "confirmed_count", MODELS, TASKS)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)
    im0 = draw_heatmap(
        axes[0],
        task_drop,
        MODELS,
        TASKS,
        "Max confirmed adjusted AUROC drop",
        "YlGnBu",
        vmin=0,
        vmax=max(0.04, float(np.nanmax(task_drop))),
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    im1 = draw_heatmap(
        axes[1],
        task_count,
        MODELS,
        TASKS,
        "Confirmed concept count",
        "OrRd",
        vmin=0,
        vmax=max(4, float(np.nanmax(task_count))),
        integer=True,
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    fig.suptitle("Figure 3. Continuation-Erasure Causal-Use Atlas", fontsize=13)
    fig.savefig(FIGURE_DIR / "figure3_causal_use_atlas.png", dpi=220)
    fig.savefig(FIGURE_DIR / "figure3_causal_use_atlas.pdf")
    plt.close(fig)

    family_drop = matrix(family_rows, "model", "family", "max_confirmed_adj_drop", MODELS, FAMILIES)
    family_count = matrix(family_rows, "model", "family", "confirmed_count", MODELS, FAMILIES)

    family_task_sum: dict[tuple[str, str], float] = {}
    for row in family_task_rows:
        key = (str(row["family"]), str(row["task_id"]))
        family_task_sum[key] = family_task_sum.get(key, 0.0) + (
            float(row["sum_confirmed_adj_drop"]) if row["sum_confirmed_adj_drop"] else 0.0
        )
    ft_rows = [
        {"family": family, "task_id": task, "sum_confirmed_adj_drop": family_task_sum.get((family, task), 0.0)}
        for family in FAMILIES
        for task in TASKS
    ]
    family_task = matrix(ft_rows, "family", "task_id", "sum_confirmed_adj_drop", FAMILIES, TASKS)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    im0 = draw_heatmap(
        axes[0],
        family_count,
        MODELS,
        FAMILIES,
        "Confirmed count by family",
        "OrRd",
        vmin=0,
        vmax=max(4, float(np.nanmax(family_count))),
        integer=True,
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02)
    im1 = draw_heatmap(
        axes[1],
        family_drop,
        MODELS,
        FAMILIES,
        "Max adjusted drop by family",
        "YlGnBu",
        vmin=0,
        vmax=max(0.04, float(np.nanmax(family_drop))),
    )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02)
    im2 = draw_heatmap(
        axes[2],
        family_task,
        FAMILIES,
        TASKS,
        "Family-task causal mass",
        "PuBuGn",
        vmin=0,
        vmax=max(0.05, float(np.nanmax(family_task))),
    )
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02)
    fig.suptitle("Figure 4. Family-Level Causal Mass and Redundancy View", fontsize=13)
    fig.savefig(FIGURE_DIR / "figure4_family_causal_summary.png", dpi=220)
    fig.savefig(FIGURE_DIR / "figure4_family_causal_summary.pdf")
    plt.close(fig)


def update_table2(family_rows: list[dict[str, object]]) -> None:
    table_path = MODEL_COMPARISON / "table2_interpretability_profile.csv"
    rows = read_csv(table_path)
    family_by_model: dict[str, list[dict[str, object]]] = {}
    for row in family_rows:
        family_by_model.setdefault(str(row["model"]), []).append(row)
    for row in rows:
        fam_rows = family_by_model.get(row["model"], [])
        confirmed = [r for r in fam_rows if int(r["confirmed_count"]) > 0]
        if confirmed:
            strongest_count = max(confirmed, key=lambda r: int(r["confirmed_count"]))
            strongest_drop = max(confirmed, key=lambda r: float(r["max_confirmed_adj_drop"] or 0.0))
            row["strongest_causal_family_by_count"] = strongest_count["family"]
            row["strongest_causal_family_by_drop"] = strongest_drop["family"]
        else:
            row["strongest_causal_family_by_count"] = ""
            row["strongest_causal_family_by_drop"] = ""
    fields = list(rows[0].keys())
    write_csv(table_path, rows, fields)
    write_csv(MODEL_COMPARISON / "interpretability_profile_summary.csv", rows, fields)


def main() -> None:
    rows = read_csv(MODEL_COMPARISON / "continuation_erasure_summary.csv")
    task_rows, family_rows, family_task_rows = summary_rows(rows)

    write_csv(
        MODEL_COMPARISON / "figure3_causal_task_summary.csv",
        task_rows,
        ["model", "task_id", "tested_count", "confirmed_count", "max_confirmed_adj_drop", "sum_confirmed_adj_drop"],
    )
    write_csv(
        MODEL_COMPARISON / "figure4_family_causal_summary.csv",
        family_rows,
        [
            "model",
            "family",
            "tested_count",
            "confirmed_count",
            "max_confirmed_adj_drop",
            "mean_confirmed_adj_drop",
            "sum_confirmed_adj_drop",
        ],
    )
    write_csv(
        MODEL_COMPARISON / "figure4_family_task_causal_mass.csv",
        family_task_rows,
        ["model", "family", "task_id", "tested_count", "confirmed_count", "max_confirmed_adj_drop", "sum_confirmed_adj_drop"],
    )
    make_figures(task_rows, family_rows, family_task_rows)
    update_table2(family_rows)
    print("wrote Figure 3 and Figure 4 summaries and figures")


if __name__ == "__main__":
    main()
