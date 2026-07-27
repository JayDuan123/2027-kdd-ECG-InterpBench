#!/usr/bin/env python
"""Re-audit final-layer waveform feature yield with matched budgets and BH-FDR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmark_v1.sparse_accessibility import bh_adjust  # noqa: E402
from scripts.run_accessibility_calibration_worker import atomic_json  # noqa: E402


PROTOCOL = "final_layer_feature_yield_e8_v2"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dictionary-root",
        type=Path,
        default=ROOT / "results/dictionary_accessibility_e8_v1/workers",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/final_layer_sparse_accessibility_e8_v2/feature_yield",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.20)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    return parser.parse_args()


def pearson_pvalues(correlations: np.ndarray, n_test: int) -> np.ndarray:
    values = np.asarray(correlations, dtype=np.float64)
    denominator = np.maximum(1.0 - np.square(values), 1e-12)
    statistic = np.abs(values) * np.sqrt((n_test - 2) / denominator)
    return 2.0 * stats.t.sf(statistic, n_test - 2)


def feature_yield_row(
    *,
    model: str,
    method: str,
    replicate_kind: str,
    replicate: int,
    scores: np.ndarray,
    selected_targets: np.ndarray,
    live: np.ndarray,
    n_targets: int,
    n_test: int,
    correlation_threshold: float,
    fdr_threshold: float,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=np.float64)
    selected_targets = np.asarray(selected_targets, dtype=np.int64)
    live = np.asarray(live, dtype=bool)
    if scores.ndim != 1 or selected_targets.shape != scores.shape or live.shape != scores.shape:
        raise ValueError("scores, selected targets, and live mask must be equal vectors")
    qvalues = bh_adjust(pearson_pvalues(scores, n_test))
    qualified = (scores >= correlation_threshold) & (qvalues < fdr_threshold) & live
    count = int(qualified.sum())
    live_count = int(live.sum())
    covered_targets = int(np.unique(selected_targets[qualified]).size) if count else 0
    return {
        "protocol": PROTOCOL,
        "model": model,
        "method": method,
        "replicate_kind": replicate_kind,
        "replicate": replicate,
        "candidate_count": len(scores),
        "live_count": live_count,
        "qualified_feature_count": count,
        "qualified_fraction_all": count / len(scores),
        "qualified_fraction_live": count / live_count if live_count else 0.0,
        "qualified_per_1000_candidates": 1000.0 * count / len(scores),
        "covered_target_count": covered_targets,
        "target_coverage": covered_targets / n_targets,
        "n_targets": int(n_targets),
        "correlation_threshold": correlation_threshold,
        "fdr_threshold": fdr_threshold,
        "n_test": n_test,
    }


def final_worker_summaries(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted(root.glob("group_*/summary.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete" and np.isclose(
            float(payload.get("relative_depth", -1.0)), 1.0
        ):
            result.append((path, payload))
    if len(result) != 6:
        raise RuntimeError(f"expected six complete final-layer dictionary workers, found {len(result)}")
    return result


def worker_rows(
    summary_path: Path,
    summary: dict[str, Any],
    correlation_threshold: float,
    fdr_threshold: float,
) -> list[dict[str, Any]]:
    archive_path = summary_path.parent / "feature_score_arrays.npz"
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    rows = []
    with np.load(archive_path, allow_pickle=False) as archive:
        targets = np.asarray(archive["waveform_targets"])
        subsets = np.asarray(archive["budget_subsets"], dtype=np.int64)
        dense_live = np.asarray(archive["dense_live"], dtype=bool)
        sae_live = np.asarray(archive["sae_live"], dtype=bool)
        random_live = np.asarray(archive["random_live"], dtype=bool)
        model = str(summary["model"])
        n_test = int(summary["n_test"])

        rows.append(
            feature_yield_row(
                model=model,
                method="dense_768",
                replicate_kind="native",
                replicate=0,
                scores=archive["dense_waveform_feature_score"],
                selected_targets=archive["dense_waveform_selected_target"],
                live=dense_live,
                n_targets=len(targets),
                n_test=n_test,
                correlation_threshold=correlation_threshold,
                fdr_threshold=fdr_threshold,
            )
        )

        sae_keys = sorted(
            key
            for key in archive.files
            if key.startswith("sae_seed") and key.endswith("_waveform_feature_score")
        )
        if len(sae_keys) != 3 or sae_live.shape != (3, int(summary["N"])):
            raise RuntimeError(f"unexpected SAE arrays in {archive_path}")
        for seed_index, score_key in enumerate(sae_keys):
            prefix = score_key[: -len("_waveform_feature_score")]
            scores = np.asarray(archive[score_key])
            selected = np.asarray(archive[f"{prefix}_waveform_selected_target"])
            rows.append(
                feature_yield_row(
                    model=model,
                    method="sae_full_6144",
                    replicate_kind="sae_seed",
                    replicate=seed_index,
                    scores=scores,
                    selected_targets=selected,
                    live=sae_live[seed_index],
                    n_targets=len(targets),
                    n_test=n_test,
                    correlation_threshold=correlation_threshold,
                    fdr_threshold=fdr_threshold,
                )
            )
            for budget_replicate, subset in enumerate(subsets):
                rows.append(
                    feature_yield_row(
                        model=model,
                        method="sae_matched_768",
                        replicate_kind="sae_seed_budget",
                        replicate=seed_index * len(subsets) + budget_replicate,
                        scores=scores[subset],
                        selected_targets=selected[subset],
                        live=sae_live[seed_index, subset],
                        n_targets=len(targets),
                        n_test=n_test,
                        correlation_threshold=correlation_threshold,
                        fdr_threshold=fdr_threshold,
                    )
                )

        random_keys = sorted(
            key
            for key in archive.files
            if key.startswith("random_seed") and key.endswith("_waveform_feature_score")
        )
        if len(random_keys) != len(subsets) or random_live.shape != (
            len(random_keys),
            int(summary["N"]),
        ):
            raise RuntimeError(f"unexpected random arrays in {archive_path}")
        for replicate, score_key in enumerate(random_keys):
            prefix = score_key[: -len("_waveform_feature_score")]
            scores = np.asarray(archive[score_key])
            selected = np.asarray(archive[f"{prefix}_waveform_selected_target"])
            rows.append(
                feature_yield_row(
                    model=model,
                    method="random_full_6144",
                    replicate_kind="random_dictionary",
                    replicate=replicate,
                    scores=scores,
                    selected_targets=selected,
                    live=random_live[replicate],
                    n_targets=len(targets),
                    n_test=n_test,
                    correlation_threshold=correlation_threshold,
                    fdr_threshold=fdr_threshold,
                )
            )
            subset = subsets[replicate]
            rows.append(
                feature_yield_row(
                    model=model,
                    method="random_matched_768",
                    replicate_kind="random_dictionary_budget",
                    replicate=replicate,
                    scores=scores[subset],
                    selected_targets=selected[subset],
                    live=random_live[replicate, subset],
                    n_targets=len(targets),
                    n_test=n_test,
                    correlation_threshold=correlation_threshold,
                    fdr_threshold=fdr_threshold,
                )
            )
    return rows


def make_figure(summary: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = [
        "dense_768",
        "sae_full_6144",
        "random_full_6144",
        "sae_matched_768",
        "random_matched_768",
    ]
    labels = ["Dense 768", "SAE 6144", "Random 6144", "SAE 768", "Random 768"]
    colors = ["#4C78A8", "#E45756", "#72B7B2", "#F58518", "#54A24B"]
    models = list(summary.model.drop_duplicates())
    x = np.arange(len(models))
    width = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.4))
    for method_index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = summary[summary.method == method].set_index("model")
        offset = (method_index - 2) * width
        axes[0].bar(
            x + offset,
            [values.loc[model, "qualified_feature_count"] for model in models],
            width,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + offset,
            [values.loc[model, "qualified_per_1000_candidates"] for model in models],
            width,
            label=label,
            color=color,
        )
    axes[0].set_title("Qualified feature count")
    axes[0].set_ylabel("Features")
    axes[1].set_title("Width-normalized feature yield")
    axes[1].set_ylabel("Qualified features per 1,000 candidates")
    for axis in axes:
        axis.set_xticks(x, models, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output / "final_layer_feature_yield.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "final_layer_feature_yield.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, summary in final_worker_summaries(args.dictionary_root):
        rows.extend(
            worker_rows(
                path,
                summary,
                args.correlation_threshold,
                args.fdr_threshold,
            )
        )
    replicate = pd.DataFrame(rows)
    replicate.to_csv(args.output_root / "feature_yield_replicates.csv", index=False)
    aggregate = (
        replicate.groupby(["model", "method"], as_index=False)
        .agg(
            replicates=("replicate", "size"),
            candidate_count=("candidate_count", "mean"),
            live_count=("live_count", "mean"),
            qualified_feature_count=("qualified_feature_count", "mean"),
            qualified_feature_count_sd=("qualified_feature_count", "std"),
            qualified_fraction_all=("qualified_fraction_all", "mean"),
            qualified_fraction_live=("qualified_fraction_live", "mean"),
            qualified_per_1000_candidates=("qualified_per_1000_candidates", "mean"),
            target_coverage=("target_coverage", "mean"),
        )
    )
    aggregate.to_csv(args.output_root / "feature_yield_model_summary.csv", index=False)
    make_figure(aggregate, args.output_root)
    audit = {
        "status": "complete",
        "protocol": PROTOCOL,
        "models": int(aggregate.model.nunique()),
        "methods": sorted(aggregate.method.unique().tolist()),
        "replicate_rows": len(replicate),
        "waveform_concepts": int(replicate["n_targets"].iloc[0]),
        "correlation_threshold": args.correlation_threshold,
        "fdr_threshold": args.fdr_threshold,
        "fdr_family": "all candidate waveform features within each frozen dictionary or matched subset",
        "selection": "target and sign selected on fixed semantic train; test used once",
        "claim_boundary": "association yield, not monosemanticity or mechanism",
    }
    atomic_json(args.output_root / "audit.json", audit)
    report = [
        "# Final-layer waveform feature yield",
        "",
        "A feature qualifies when its train-selected orientation has held-out "
        f"Pearson r >= {args.correlation_threshold:.2f} and dictionary-wise BH q < "
        f"{args.fdr_threshold:.2f}. Absolute counts, candidate-normalized yield, and "
        "target coverage are reported together.",
        "",
        markdown_table(aggregate.round(4)),
        "",
        "This is an association-yield audit. It does not establish monosemanticity, "
        "causal use, or clinical utility.",
    ]
    (args.output_root / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
