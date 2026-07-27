#!/usr/bin/env python
"""Build figures, report, and README section for the fair method comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.method_comparison_common import BASE, atomic_write_text, write_json  # noqa: E402


README = ROOT / "README.md"
FIGURES = ROOT / "docs" / "figures"
START = "<!-- METHOD_COMPARISON_V1_START -->"
END = "<!-- METHOD_COMPARISON_V1_END -->"
METHOD_LABELS = {
    "sae_common64": "SAE common-64",
    "sae_existing_8d": "SAE existing 8d",
    "pca64": "PCA-64",
    "ica64": "ICA-64",
    "semi_nmf64": "Semi-NMF-64",
    "random_basis64": "Random basis-64",
    "sparse_probe": "Sparse probe",
    "supervised_cav": "Supervised CAV",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    return parser.parse_args()


def number(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def make_main_figure(base: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = base / "summary"
    matched = pd.read_csv(summary / "reconstruction_matched_profile.csv")
    inference = pd.read_csv(summary / "hierarchical_method_inference.csv")
    functional = pd.read_csv(summary / "functional_seed_pair_stability.csv")
    label_budget = pd.read_csv(summary / "label_budget_profile.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.4))

    matched = matched.sort_values("absolute_r2_gap_mean")
    labels = matched.method.map(METHOD_LABELS)
    axes[0, 0].barh(labels, matched.absolute_r2_gap_mean, color="#457b9d")
    axes[0, 0].set_xlabel("Mean absolute R2 gap to SAE common-64")
    axes[0, 0].set_title("Reconstruction-matched operating points")
    for index, row in enumerate(matched.itertuples(index=False)):
        axes[0, 0].text(
            row.absolute_r2_gap_mean,
            index,
            f"  k={row.selected_code_budget_mean:.1f}",
            va="center",
            fontsize=8,
        )

    selectivity = inference[
        inference.regime.eq("common64_energy")
        & inference.metric.eq("selectivity_margin")
    ].sort_values("mean_delta")
    y = np.arange(len(selectivity))
    axes[0, 1].errorbar(
        selectivity.mean_delta,
        y,
        xerr=np.vstack(
            [
                selectivity.mean_delta - selectivity.ci_low,
                selectivity.ci_high - selectivity.mean_delta,
            ]
        ),
        fmt="o",
        color="#d1495b",
        capsize=3,
    )
    axes[0, 1].axvline(0, color="#444444", linewidth=1)
    axes[0, 1].set_yticks(y, selectivity.method.map(METHOD_LABELS), fontsize=8)
    axes[0, 1].set_xlabel("Baseline minus SAE selectivity (95% CI)")
    axes[0, 1].set_title("Common-budget intervention comparison")

    functional_profile = (
        functional.groupby("method", as_index=False)
        .functional_subspace_overlap.mean()
        .sort_values("functional_subspace_overlap", ascending=False)
    )
    axes[1, 0].barh(
        functional_profile.method.map(METHOD_LABELS),
        functional_profile.functional_subspace_overlap,
        color="#2a9d8f",
    )
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_xlabel("Mean cross-seed functional subspace overlap")
    axes[1, 0].set_title("Direction reproducibility")

    colors = {
        "sae_common64": "#d1495b",
        "pca64": "#457b9d",
        "ica64": "#2a9d8f",
        "semi_nmf64": "#e9c46a",
        "random_basis64": "#8d99ae",
        "sparse_probe": "#6a4c93",
        "supervised_cav": "#f4a261",
    }
    for method, group in label_budget.groupby("method"):
        axes[1, 1].plot(
            group.label_budget_requested,
            group.selectivity_margin_mean,
            marker="o",
            linewidth=1.4,
            label=METHOD_LABELS[method],
            color=colors[method],
        )
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks([32, 128, 512, 2048], ["32", "128", "512", "2048"])
    axes[1, 1].set_xlabel("Target labels")
    axes[1, 1].set_ylabel("Mean selectivity margin")
    axes[1, 1].set_title("Label-efficiency curve")
    axes[1, 1].legend(fontsize=7, ncol=2)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_waveform_figure(base: Path, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(base / "waveform_triangle" / "method_triangle_summary.csv")
    qc = summary[summary.analysis_set.eq("measurement_qc")].copy()
    qc = qc.sort_values("joint_pass_cells")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    labels = qc.method.map(METHOD_LABELS)
    axes[0].barh(labels, qc.joint_pass_cells, color="#457b9d")
    axes[0].set_xlim(0, 48)
    axes[0].set_xlabel("Joint-pass intervention cells out of 48")
    axes[0].set_title("Waveform triangle joint pass")
    x = np.arange(len(qc))
    axes[1].bar(
        x - 0.18,
        qc.selected_raw_sign_concordance,
        width=0.36,
        label="selected",
        color="#2a9d8f",
    )
    axes[1].bar(
        x + 0.18,
        qc.random_raw_sign_concordance,
        width=0.36,
        label="random",
        color="#9aa0a6",
    )
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Sign concordance")
    axes[1].set_title("Selected direction vs matched random")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="x" if axis is axes[0] else "y", alpha=0.2)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_content(base: Path) -> tuple[str, str, dict]:
    summary = base / "summary"
    reconstruction = pd.read_csv(summary / "reconstruction_profile.csv")
    matched = pd.read_csv(summary / "reconstruction_matched_profile.csv")
    method_profile = pd.read_csv(summary / "method_profile.csv")
    inference = pd.read_csv(summary / "hierarchical_method_inference.csv")
    envelopes = pd.read_csv(summary / "hierarchical_leave_one_out_envelopes.csv")
    dictionary = pd.read_csv(summary / "dictionary_seed_pair_stability.csv")
    functional = pd.read_csv(summary / "functional_seed_pair_stability.csv")
    label_budget = pd.read_csv(summary / "label_budget_profile.csv")
    waveform = pd.read_csv(base / "waveform_triangle" / "method_triangle_summary.csv")
    waveform_inference = pd.read_csv(
        base / "waveform_triangle" / "method_triangle_hierarchical_inference.csv"
    )

    common_selectivity = inference[
        inference.regime.eq("common64_energy")
        & inference.metric.eq("selectivity_margin")
    ].sort_values("method")
    existing_selectivity = inference[
        inference.regime.eq("existing_sae_energy")
        & inference.metric.eq("selectivity_margin")
    ].sort_values("method")
    simple = {"pca64", "ica64", "semi_nmf64", "random_basis64"}
    simple_rows = common_selectivity[common_selectivity.method.isin(simple)]
    sae_significantly_better = int(
        ((simple_rows.ci_high < 0) & simple_rows.q_two_sided.lt(0.05)).sum()
    )
    baseline_significantly_better = int(
        ((simple_rows.ci_low > 0) & simple_rows.q_two_sided.lt(0.05)).sum()
    )
    leave_robust = int(envelopes.leave_one_out_all_same_sign.sum())
    dictionary_profile = (
        dictionary.groupby("method", as_index=False)
        .agg(
            matched_cosine=("matched_abs_cosine_mean", "mean"),
            subspace_overlap=("subspace_overlap", "mean"),
        )
    )
    functional_profile = (
        functional.groupby("method", as_index=False)
        .agg(
            functional_cosine=("functional_abs_cosine_mean", "mean"),
            functional_overlap=("functional_subspace_overlap", "mean"),
        )
    )
    waveform_qc = waveform[waveform.analysis_set.eq("measurement_qc")].sort_values("method")

    reconstruction_rows = []
    for row in reconstruction.sort_values("method").itertuples(index=False):
        reconstruction_rows.append(
            [
                METHOD_LABELS[row.method],
                number(row.dense_recon_r2_mean),
                number(row.topk_recon_r2_mean),
                number(row.mean_active_coefficients, 2),
            ]
        )
    matched_rows = [
        [
            METHOD_LABELS[row.method],
            number(row.selected_code_budget_mean, 2),
            number(row.recon_r2_mean),
            number(row.reference_recon_r2_mean),
            number(row.absolute_r2_gap_mean),
        ]
        for row in matched.sort_values("method").itertuples(index=False)
    ]

    def inference_rows(frame: pd.DataFrame) -> list[list[str]]:
        return [
            [
                METHOD_LABELS[row.method],
                number(row.mean_delta),
                f"[{number(row.ci_low)}, {number(row.ci_high)}]",
                number(row.q_two_sided),
            ]
            for row in frame.itertuples(index=False)
        ]

    stability_rows = []
    merged_stability = dictionary_profile.merge(functional_profile, on="method", how="outer")
    for row in merged_stability.sort_values("method").itertuples(index=False):
        stability_rows.append(
            [
                METHOD_LABELS[row.method],
                number(float(row.matched_cosine)),
                number(float(row.subspace_overlap)),
                number(float(row.functional_cosine)),
                number(float(row.functional_overlap)),
            ]
        )
    waveform_rows = [
        [
            METHOD_LABELS[row.method],
            f"{int(row.joint_pass_cells)}/48",
            number(row.selected_raw_sign_concordance),
            number(row.random_raw_sign_concordance),
            number(row.selected_vs_random_mean),
        ]
        for row in waveform_qc.itertuples(index=False)
    ]

    report_lines = [
        "# Fair SAE Method Comparison",
        "",
        "## Design",
        "",
        "The benchmark uses 6 ECG foundation models, 4 external cohorts, 3 data-resampled seeds, and 270 target-seed cells. Common-budget methods use rank 64 and top-5 target directions. All readout interventions are matched per test record to the reference SAE L2 norm in frozen-head standardized activation space.",
        "",
        "The supervised sparse probe and CAV are direction-only upper-bound comparators. Reconstruction is therefore reported only for SAE, PCA, ICA, Semi-NMF, and random-basis decompositions.",
        "",
        "## Reconstruction and capacity matching",
        "",
        markdown_table(
            ["Method", "Dense R2", "Top-k R2", "Mean active coefficients"],
            reconstruction_rows,
        ),
        "",
        "Nearest rate-distortion points to the common-64 SAE top-5 R2:",
        "",
        markdown_table(
            ["Method", "Selected k", "Method R2", "SAE R2", "Absolute gap"],
            matched_rows,
        ),
        "",
        "![Fair comparison overview](../../docs/figures/method_comparison_v1_main.png)",
        "",
        "## Hierarchical intervention comparison",
        "",
        "Deltas below are baseline minus SAE. Negative selectivity deltas favor SAE. Inference averages seeds first, then uses model/cohort/family crossed bootstrap with 10,000 samples and BH-FDR.",
        "",
        "### Common rank-64 reference",
        "",
        markdown_table(["Baseline", "Delta selectivity", "95% CI", "q"], inference_rows(common_selectivity)),
        "",
        "### Existing overcomplete 8d SAE reference",
        "",
        markdown_table(["Baseline", "Delta selectivity", "95% CI", "q"], inference_rows(existing_selectivity)),
        "",
        f"Among the four simple unsupervised controls, SAE common-64 is significantly better for {sae_significantly_better}/4 selectivity contrasts and significantly worse for {baseline_significantly_better}/4. Across all endpoints and regimes, {leave_robust}/{len(envelopes)} contrasts retain their sign in every leave-one-model/cohort/family-out check.",
        "",
        "## Cross-seed stability",
        "",
        markdown_table(
            ["Method", "Dictionary cosine", "Dictionary overlap", "Functional cosine", "Functional overlap"],
            stability_rows,
        ),
        "",
        "## Label efficiency",
        "",
        "Target directions are re-selected using deterministic stratified budgets of 32, 128, 512, and 2,048 labels. The representation fit and test set remain frozen. Full curves and crossed-bootstrap contrasts are stored in the summary directory.",
        "",
        markdown_table(
            ["Labels", "Method", "STE", "OTD", "Selectivity"],
            [
                [
                    str(int(row.label_budget_requested)),
                    METHOD_LABELS[row.method],
                    number(row.ste_mean),
                    number(row.otd_mean),
                    number(row.selectivity_margin_mean),
                ]
                for row in label_budget.itertuples(index=False)
            ],
        ),
        "",
        "## Waveform triangle",
        "",
        markdown_table(
            ["Method", "QC joint pass", "Selected sign", "Random sign", "Selected-random"],
            waveform_rows,
        ),
        "",
        "![Method waveform triangle](../../docs/figures/method_comparison_v1_waveform.png)",
        "",
        "The waveform panel reuses the same 12 cells and 256 records per cell as benchmark_extension_v2. It evaluates whether each frozen direction mediates the response to controlled input perturbations; it does not generate ECG waveforms from latent edits.",
        "",
        "## Claim boundary",
        "",
        "This comparison can establish whether SAE offers a reproducibility/selectivity/off-target tradeoff beyond linear and supervised controls under matched budgets. It does not establish biological mechanism, clinical utility, or waveform-generative causality. A null SAE-vs-PCA/ICA result must be reported as evidence for generic low-dimensional decomposition rather than SAE-specific value.",
    ]
    report = "\n".join(report_lines) + "\n"

    common_best = common_selectivity.sort_values("mean_delta").iloc[0]
    existing_best = existing_selectivity.sort_values("mean_delta").iloc[0]
    readme_lines = [
        START,
        "### 4.10 SAE 与传统表示方法的公平比较（benchmark_method_comparison_v1）",
        "",
        "该实验冻结既有 activation、head、数据划分和测试记录，不重新提取原始 ECG。共同预算比较使用 rank=64、top-k=5；另以现有 8d SAE 为 practical reference。所有 readout intervention 都在每条测试记录上精确匹配 reference SAE 的 activation-space L2 norm。",
        "",
        f"完整矩阵包含 6 models x 4 cohorts x 3 seeds，共 270 个 target-seed cells。PCA、ICA、Semi-NMF 和随机正交基参加重建与干预比较；sparse probe 和 supervised CAV 只参加方向比较，不为它们伪造 reconstruction 指标。四个简单无监督基线中，common-64 SAE 在 {sae_significantly_better}/4 个 selectivity contrasts 上显著更好，在 {baseline_significantly_better}/4 个上显著更差。",
        "",
        f"Common-budget 中相对 SAE 最负的 baseline-minus-SAE selectivity delta 来自 {METHOD_LABELS[common_best.method]}（{number(common_best.mean_delta)}, 95% CI [{number(common_best.ci_low)}, {number(common_best.ci_high)}], q={number(common_best.q_two_sided)}）；existing-8d reference 中对应方法为 {METHOD_LABELS[existing_best.method]}（{number(existing_best.mean_delta)}）。负值表示 SAE selectivity 更高。",
        "",
        "![Fair SAE method comparison](docs/figures/method_comparison_v1_main.png)",
        "",
        "Label-budget 曲线使用 32/128/512/2048 个分层训练标签重新选择方向；waveform triangle 复用 v2 的 12 个 cells 和每 cell 256 条记录，只评估受控输入变化的 readout mediation，不生成或保存 ECG。",
        "",
        "![Method-specific waveform triangle](docs/figures/method_comparison_v1_waveform.png)",
        "",
        "完整 reconstruction、层级 bootstrap、leave-one-factor-out、跨 seed 稳定性、label-efficiency 和 waveform 结果见 `results/benchmark_method_comparison_v1/method_comparison_report.md`。",
        END,
    ]
    readme_section = "\n".join(readme_lines)
    metadata = {
        "schema_version": 1,
        "common_selectivity_contrasts": len(common_selectivity),
        "existing_selectivity_contrasts": len(existing_selectivity),
        "simple_baselines_sae_significantly_better": sae_significantly_better,
        "simple_baselines_baseline_significantly_better": baseline_significantly_better,
        "leave_one_out_same_sign": leave_robust,
        "leave_one_out_contrasts": len(envelopes),
        "waveform_methods": len(waveform_qc),
        "waveform_inference_rows": len(waveform_inference),
        "claim_boundary_present": True,
    }
    return report, readme_section, metadata


def update_readme(section: str) -> None:
    content = README.read_text()
    if START in content or END in content:
        if content.count(START) != 1 or content.count(END) != 1:
            raise RuntimeError("README method-comparison markers are malformed")
        before, rest = content.split(START, 1)
        _, after = rest.split(END, 1)
        content = before.rstrip() + "\n\n" + section + after
    else:
        marker = "\n## 5. 如何理解这些结果"
        if marker not in content:
            raise RuntimeError("README insertion point not found")
        content = content.replace(marker, "\n\n" + section + marker, 1)
    atomic_write_text(README, content)


def main() -> None:
    args = parse_args()
    required = [
        args.base / "summary" / "metadata.json",
        args.base / "waveform_triangle" / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Cannot build report; missing summaries: {missing}")
    main_figure = FIGURES / "method_comparison_v1_main.png"
    waveform_figure = FIGURES / "method_comparison_v1_waveform.png"
    make_main_figure(args.base, main_figure)
    make_waveform_figure(args.base, waveform_figure)
    report, section, metadata = build_content(args.base)
    atomic_write_text(args.base / "method_comparison_report.md", report)
    update_readme(section)
    metadata.update(
        {
            "report": str((args.base / "method_comparison_report.md").relative_to(ROOT)),
            "figures": [str(main_figure.relative_to(ROOT)), str(waveform_figure.relative_to(ROOT))],
            "readme_markers": README.read_text().count(START),
            "all_complete": True,
        }
    )
    write_json(args.base / "report_metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
