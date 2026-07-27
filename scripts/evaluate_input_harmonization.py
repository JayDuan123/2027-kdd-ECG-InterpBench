#!/usr/bin/env python3
"""Evaluate final-layer concept accessibility under harmonized input protocols."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import kendalltau, spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark_v1.accessibility_calibration import feature_concept_correlations  # noqa: E402
from benchmark_v1.input_harmonization import MODEL_INTERFACES, PROTOCOLS, final_layer_for_model  # noqa: E402
from benchmark_v1.multiscale_sae import MODEL_SUFFIXES, read_csv, selected_concept_metrics, standardized_concepts  # noqa: E402
try:
    from paper_figure_style import configure_paper_fonts  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - package import path
    from scripts.paper_figure_style import configure_paper_fonts  # noqa: E402


MODEL_DISPLAY = {
    "cardiac_fm": "CARDIAC-FM",
    "csfm": "CSFM",
    "ecg_fm": "ECG-FM",
    "ecg_jepa": "ECG-JEPA",
    "hubert_ecg": "HuBERT-ECG",
    "st_mem": "ST-MEM",
}
DISPLAY_TO_SUFFIX = {name: MODEL_SUFFIXES[name] for name in MODEL_SUFFIXES}
SEMANTIC_SEED = 20_260_714
BOOTSTRAP_SEED = 20_260_715


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=ROOT / "results/input_harmonization_v1/probe_features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/input_harmonization_v1/summary",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=ROOT / "results/manifest/split.csv",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        default=ROOT / "results/manifest/concepts_matrix.csv",
    )
    parser.add_argument("--semantic-train-limit", type=int, default=4096)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def canonical_records(path: Path) -> list[dict[str, str]]:
    order = {"train": 0, "val": 1, "test": 2}
    rows = [row for row in read_csv(path) if row.get("split") in order]
    rows.sort(key=lambda row: (order[row["split"]], int(row["ecg_id"])))
    return rows


def native_paths(model: str) -> tuple[Path, Path]:
    display = MODEL_DISPLAY[model]
    suffix = DISPLAY_TO_SUFFIX[display]
    root = ROOT / "results/probe_features" / suffix
    layer = final_layer_for_model(model)
    return root / f"layer_{layer:02d}_mean.npy", root / "records.csv"


def protocol_paths(model: str, protocol: str, feature_root: Path) -> tuple[Path, Path]:
    if protocol == "native":
        return native_paths(model)
    root = feature_root / protocol / model
    layer = final_layer_for_model(model)
    return root / f"layer_{layer:02d}_mean.npy", root / "records.csv"


def aligned_features(model: str, protocol: str, feature_root: Path, ids: list[str]) -> np.ndarray:
    feature_path, records_path = protocol_paths(model, protocol, feature_root)
    records = read_csv(records_path)
    row_by_id = {row["ecg_id"]: index for index, row in enumerate(records)}
    missing = [ecg_id for ecg_id in ids if ecg_id not in row_by_id]
    if missing:
        raise RuntimeError(f"{model}/{protocol} missing {len(missing)} canonical records")
    source = np.load(feature_path, mmap_mode="r")
    indices = np.asarray([row_by_id[ecg_id] for ecg_id in ids], dtype=np.int64)
    values = np.asarray(source[indices], dtype=np.float32)
    if values.shape != (len(ids), 768) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid feature matrix {model}/{protocol}: {values.shape}")
    return values


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = x.T @ y
    xx = x.T @ x
    yy = y.T @ y
    denominator = np.sqrt(np.square(xx).sum() * np.square(yy).sum())
    return float(np.square(cross).sum() / denominator) if denominator > 0 else 0.0


def bootstrap_weights(patient_ids: np.ndarray, draws: int) -> tuple[np.ndarray, np.ndarray]:
    patients, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    weights = rng.multinomial(len(patients), np.full(len(patients), 1.0 / len(patients)), size=draws)
    return inverse, weights.astype(np.float64)


def clustered_correlations(
    selected_features: np.ndarray,
    concepts: np.ndarray,
    patient_inverse: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray(selected_features, dtype=np.float64)
    y = np.asarray(concepts, dtype=np.float64)
    n_patients = int(patient_inverse.max()) + 1
    n_concepts = x.shape[1]
    count = np.zeros(n_patients, dtype=np.float64)
    sum_x = np.zeros((n_patients, n_concepts), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    sum_y = np.zeros_like(sum_x)
    sum_y2 = np.zeros_like(sum_x)
    sum_xy = np.zeros_like(sum_x)
    np.add.at(count, patient_inverse, 1.0)
    np.add.at(sum_x, patient_inverse, x)
    np.add.at(sum_x2, patient_inverse, x * x)
    np.add.at(sum_y, patient_inverse, y)
    np.add.at(sum_y2, patient_inverse, y * y)
    np.add.at(sum_xy, patient_inverse, x * y)
    n = weights @ count
    sx = weights @ sum_x
    sx2 = weights @ sum_x2
    sy = weights @ sum_y
    sy2 = weights @ sum_y2
    sxy = weights @ sum_xy
    covariance = sxy - sx * sy / n[:, None]
    var_x = np.maximum(sx2 - sx * sx / n[:, None], 0.0)
    var_y = np.maximum(sy2 - sy * sy / n[:, None], 0.0)
    denominator = np.sqrt(var_x * var_y)
    return np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)


def bootstrap_pvalue(values: np.ndarray) -> float:
    below = (np.sum(values <= 0) + 1.0) / (len(values) + 1.0)
    above = (np.sum(values >= 0) + 1.0) / (len(values) + 1.0)
    return float(min(1.0, 2.0 * min(below, above)))


def bh_adjust(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate((ranked * len(values) / np.arange(1, len(values) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output.tolist()


def make_figure(model_rows: list[dict[str, Any]], cka_rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_paper_fonts()
    import matplotlib.pyplot as plt

    models = [MODEL_DISPLAY[key] for key in sorted(MODEL_INTERFACES)]
    protocols = list(PROTOCOLS)
    colors = {"native": "#66A9D6", "lead": "#E69A4E", "temporal": "#66C2A4", "joint": "#DCA6C8"}
    lookup = {(row["model"], row["protocol"]): row for row in model_rows}
    cka = {(row["model"], row["protocol"]): row for row in cka_rows}
    x = np.arange(len(models))
    width = 0.19
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))
    for index, protocol in enumerate(protocols):
        shift = (index - 1.5) * width
        axes[0].bar(
            x + shift,
            [float(lookup[(model, protocol)]["mean_abs_correlation"]) for model in models],
            width,
            color=colors[protocol],
            label=protocol,
        )
        axes[1].bar(
            x + shift,
            [float(lookup[(model, protocol)]["coverage_abs_r_ge_020"]) for model in models],
            width,
            color=colors[protocol],
        )
    for index, protocol in enumerate(PROTOCOLS[1:]):
        shift = (index - 1) * 0.24
        axes[2].bar(
            x + shift,
            [float(cka[(model, protocol)]["linear_cka_to_native"]) for model in models],
            0.23,
            color=colors[protocol],
            label=protocol,
        )
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(models, rotation=35, ha="right")
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Test mean |r|")
    axes[0].set_title("Dense concept accessibility")
    axes[1].set_ylabel("Concept coverage")
    axes[1].set_title("Coverage at |r| >= 0.20")
    axes[2].set_ylabel("Linear CKA")
    axes[2].set_title("Similarity to native input")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output / "input_harmonization_final_layer.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "input_harmonization_final_layer.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = canonical_records(args.split_csv)
    ids = [row["ecg_id"] for row in records]
    splits = np.asarray([row["split"] for row in records])
    patients = np.asarray([row.get("patient_id", row["ecg_id"]) for row in records])
    train_idx = np.flatnonzero(splits == "train")
    test_idx = np.flatnonzero(splits == "test")
    rng = np.random.default_rng(SEMANTIC_SEED)
    semantic_idx = np.sort(rng.choice(train_idx, size=min(args.semantic_train_limit, len(train_idx)), replace=False))
    concepts, concept_names, _, _ = standardized_concepts(ids, read_csv(args.concepts), splits == "train")
    family = {
        row["concept_id"]: row["family"]
        for row in read_csv(ROOT / "configs/concepts.csv")
        if row.get("main") == "yes"
    }
    patient_inverse, weights = bootstrap_weights(patients[test_idx], args.bootstrap)

    concept_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    cka_rows: list[dict[str, Any]] = []
    paired_concept_rows: list[dict[str, Any]] = []
    paired_model_rows: list[dict[str, Any]] = []
    summaries: dict[tuple[str, str], dict[str, float]] = {}

    for model in sorted(MODEL_INTERFACES):
        display = MODEL_DISPLAY[model]
        matrices = {protocol: aligned_features(model, protocol, args.feature_root, ids) for protocol in PROTOCOLS}
        selected_test: dict[str, np.ndarray] = {}
        boot_corr: dict[str, np.ndarray] = {}
        for protocol, matrix in matrices.items():
            train_corr = feature_concept_correlations(matrix[semantic_idx], concepts[semantic_idx])
            test_corr = feature_concept_correlations(matrix[test_idx], concepts[test_idx])
            rows, summary = selected_concept_metrics(train_corr, test_corr, concept_names)
            summaries[(display, protocol)] = summary
            selected = np.asarray([int(row["selected_feature"]) for row in rows], dtype=np.int64)
            selected_test[protocol] = matrix[test_idx][:, selected]
            boot_corr[protocol] = clustered_correlations(
                selected_test[protocol], concepts[test_idx], patient_inverse, weights
            )
            for row in rows:
                concept_rows.append(
                    {
                        "model": display,
                        "protocol": protocol,
                        "layer": final_layer_for_model(model),
                        "concept": row["concept"],
                        "family": family[row["concept"]],
                        "selected_feature": row["selected_feature"],
                        "train_correlation": row["train_correlation"],
                        "test_correlation": row["eval_correlation"],
                        "abs_test_correlation": row["abs_eval_correlation"],
                        "covered_020": int(float(row["abs_eval_correlation"]) >= 0.20),
                    }
                )
            model_rows.append(
                {
                    "model": display,
                    "protocol": protocol,
                    "layer": final_layer_for_model(model),
                    "mean_abs_correlation": summary["mean_train_selected_abs_correlation"],
                    "median_abs_correlation": summary["median_train_selected_abs_correlation"],
                    "coverage_abs_r_ge_020": summary["coverage_abs_r_ge_0_20"],
                    "sign_consistency": summary["sign_consistency_fraction"],
                    "n_concepts": len(concept_names),
                }
            )
            if protocol != "native":
                cka_rows.append(
                    {
                        "model": display,
                        "protocol": protocol,
                        "layer": final_layer_for_model(model),
                        "linear_cka_to_native": linear_cka(matrices["native"][test_idx], matrix[test_idx]),
                        "n_test": len(test_idx),
                    }
                )

        native_boot = np.abs(boot_corr["native"])
        native_observed = np.asarray(
            [float(row["abs_test_correlation"]) for row in concept_rows if row["model"] == display and row["protocol"] == "native"]
        )
        for protocol in PROTOCOLS[1:]:
            protocol_observed = np.asarray(
                [float(row["abs_test_correlation"]) for row in concept_rows if row["model"] == display and row["protocol"] == protocol]
            )
            delta_boot = np.abs(boot_corr[protocol]) - native_boot
            for concept_index, concept in enumerate(concept_names):
                values = delta_boot[:, concept_index]
                paired_concept_rows.append(
                    {
                        "model": display,
                        "protocol": protocol,
                        "concept": concept,
                        "family": family[concept],
                        "observed_delta_abs_r": protocol_observed[concept_index] - native_observed[concept_index],
                        "bootstrap_mean_delta_abs_r": float(values.mean()),
                        "ci_low": float(np.quantile(values, 0.025)),
                        "ci_high": float(np.quantile(values, 0.975)),
                        "p_value": bootstrap_pvalue(values),
                    }
                )
            mean_delta = delta_boot.mean(axis=1)
            coverage_delta = (np.abs(boot_corr[protocol]) >= 0.20).mean(axis=1) - (native_boot >= 0.20).mean(axis=1)
            paired_model_rows.append(
                {
                    "model": display,
                    "protocol": protocol,
                    "observed_delta_mean_abs_r": float(protocol_observed.mean() - native_observed.mean()),
                    "mean_delta_ci_low": float(np.quantile(mean_delta, 0.025)),
                    "mean_delta_ci_high": float(np.quantile(mean_delta, 0.975)),
                    "mean_delta_p_value": bootstrap_pvalue(mean_delta),
                    "observed_delta_coverage": float((protocol_observed >= 0.20).mean() - (native_observed >= 0.20).mean()),
                    "coverage_delta_ci_low": float(np.quantile(coverage_delta, 0.025)),
                    "coverage_delta_ci_high": float(np.quantile(coverage_delta, 0.975)),
                    "coverage_delta_p_value": bootstrap_pvalue(coverage_delta),
                }
            )

    concept_q = bh_adjust([float(row["p_value"]) for row in paired_concept_rows])
    for row, q_value in zip(paired_concept_rows, concept_q):
        row["q_value_bh"] = q_value
    model_mean_q = bh_adjust([float(row["mean_delta_p_value"]) for row in paired_model_rows])
    model_coverage_q = bh_adjust([float(row["coverage_delta_p_value"]) for row in paired_model_rows])
    for row, mean_q, coverage_q in zip(paired_model_rows, model_mean_q, model_coverage_q):
        row["mean_delta_q_value_bh"] = mean_q
        row["coverage_delta_q_value_bh"] = coverage_q

    rank_rows = []
    for protocol in PROTOCOLS[1:]:
        for metric in ("mean_train_selected_abs_correlation", "coverage_abs_r_ge_0_20"):
            native_values = [summaries[(MODEL_DISPLAY[key], "native")][metric] for key in sorted(MODEL_INTERFACES)]
            protocol_values = [summaries[(MODEL_DISPLAY[key], protocol)][metric] for key in sorted(MODEL_INTERFACES)]
            rank_rows.append(
                {
                    "protocol": protocol,
                    "metric": metric,
                    "spearman_rho": float(spearmanr(native_values, protocol_values).statistic),
                    "kendall_tau": float(kendalltau(native_values, protocol_values).statistic),
                    "n_models": len(native_values),
                }
            )

    write_csv(args.output_dir / "concept_metrics.csv", concept_rows)
    write_csv(args.output_dir / "model_summary.csv", model_rows)
    write_csv(args.output_dir / "cka_to_native.csv", cka_rows)
    write_csv(args.output_dir / "paired_concept_bootstrap.csv", paired_concept_rows)
    write_csv(args.output_dir / "paired_model_bootstrap.csv", paired_model_rows)
    write_csv(args.output_dir / "rank_stability.csv", rank_rows)
    make_figure(model_rows, cka_rows, args.output_dir)
    audit = {
        "status": "complete",
        "models": len(MODEL_INTERFACES),
        "protocols": list(PROTOCOLS),
        "records": len(records),
        "train": len(train_idx),
        "semantic_train": len(semantic_idx),
        "test": len(test_idx),
        "concepts": len(concept_names),
        "bootstrap_draws": args.bootstrap,
        "selection": "best dense coordinate on fixed semantic training subset",
        "evaluation": "patient-disjoint test split",
        "fdr": "Benjamini-Hochberg across 18x49 paired concept tests and separately across 18 model-level tests",
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "report.md").write_text(
        "# Input Harmonization Ablation\n\n"
        "Final-layer dense-coordinate accessibility under native, lead-harmonized, temporal-harmonized, and jointly harmonized input protocols. "
        "Feature selection uses training data only; evaluation and clustered bootstrap use the patient-disjoint test split.\n\n"
        f"- records: {len(records)}\n- test records: {len(test_idx)}\n- concepts: {len(concept_names)}\n"
        f"- patient-cluster bootstrap draws: {args.bootstrap}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
