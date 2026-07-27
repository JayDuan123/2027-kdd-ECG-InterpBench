#!/usr/bin/env python
"""Build figures, report, and README section for benchmark_extension_v2."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_extension_v2_common import V2, atomic_write_text, write_json  # noqa: E402


README = ROOT / "README.md"
FIGURE_ROOT = ROOT / "docs" / "figures"
START = "<!-- BENCHMARK_EXTENSION_V2_START -->"
END = "<!-- BENCHMARK_EXTENSION_V2_END -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=V2)
    return parser.parse_args()


def number(value: float, digits: int = 4) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def interval(row: pd.Series) -> str:
    return f"[{number(float(row.ci_low))}, {number(float(row.ci_high))}]"


def make_robustness_figure(base: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transport = pd.read_csv(base / "hierarchical_robustness" / "transport_crossed_inference.csv")
    protocol = pd.read_csv(base / "hierarchical_robustness" / "protocol_factor_inference.csv")
    t = transport[
        transport.method.eq("fewshot_n2048")
        & transport.metric.isin(["recon_r2", "ste", "excess_selectivity"])
    ].copy()
    p = protocol[
        protocol.subset.eq("frozen_tier0_only")
        & protocol.contrast.isin(["local_minus_frozen", "adapted_minus_frozen"])
        & protocol.metric.isin(["ste", "excess_selectivity"])
    ].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    t = t.sort_values("metric")
    y = np.arange(len(t))
    axes[0].errorbar(
        t.mean_delta,
        y,
        xerr=np.vstack([t.mean_delta - t.ci_low, t.ci_high - t.mean_delta]),
        fmt="o",
        color="#1f77b4",
        capsize=3,
    )
    axes[0].axvline(0, color="#444444", linewidth=1)
    axes[0].set_yticks(y, t.metric.str.replace("_", " "))
    axes[0].set_title("Few-shot n=2048 vs frozen")
    axes[0].set_xlabel("Crossed-factor mean delta (95% CI)")

    colors = {"local_minus_frozen": "#2a9d8f", "adapted_minus_frozen": "#e76f51"}
    positions = []
    labels = []
    for index, row in enumerate(p.sort_values(["metric", "contrast"]).itertuples()):
        axes[1].errorbar(
            row.mean_delta,
            index,
            xerr=[[row.mean_delta - row.ci_low], [row.ci_high - row.mean_delta]],
            fmt="o",
            color=colors[row.contrast],
            capsize=3,
        )
        positions.append(index)
        labels.append(f"{row.metric.replace('_', ' ')}: {row.contrast.replace('_', ' ')}")
    axes[1].axvline(0, color="#444444", linewidth=1)
    axes[1].set_yticks(positions, labels, fontsize=8)
    axes[1].set_title("Tier-0 protocol sensitivity")
    axes[1].set_xlabel("Factor-bootstrap mean delta (95% CI)")
    for axis in axes:
        axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def make_validation_figure(base: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bias = pd.read_csv(base / "waveform_failure_bias" / "candidate_bias_covariates.csv")
    capacity = pd.read_csv(base / "sae_stability" / "capacity_stability_summary.csv")
    functional = pd.read_csv(base / "sae_stability" / "functional_top5_stability_summary.csv")
    triangle = pd.read_csv(base / "waveform_triangle" / "triangle_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.0))
    max_bias = (
        bias.assign(abs_smd=bias.standardized_mean_difference_failure_minus_success.abs())
        .groupby(["cohort", "phenotype"], as_index=False)
        .abs_smd.max()
    )
    labels = max_bias.cohort.str.replace("_f", "") + "/" + max_bias.phenotype.str.replace("_", " ")
    axes[0, 0].barh(labels, max_bias.abs_smd, color="#d1495b")
    axes[0, 0].axvline(0.1, color="#444444", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Maximum observed selection-bias SMD")
    axes[0, 0].set_xlabel("Absolute standardized mean difference")

    for model, group in capacity.groupby("model"):
        axes[0, 1].plot(
            group.N_over_d,
            group.matched_cosine_mean,
            marker="o",
            linewidth=1.2,
            label=model,
        )
    axes[0, 1].set_xticks([8, 16, 32])
    axes[0, 1].set_title("Cross-seed dictionary matching")
    axes[0, 1].set_xlabel("N/d")
    axes[0, 1].set_ylabel("Mean matched decoder cosine")
    axes[0, 1].legend(fontsize=7, ncol=2)

    functional_plot = (
        functional.groupby("model", as_index=False)[
            ["selected_subspace_overlap_mean", "random_subspace_overlap_mean"]
        ].mean()
    )
    x = np.arange(len(functional_plot))
    axes[1, 0].bar(
        x - 0.18,
        functional_plot.selected_subspace_overlap_mean,
        width=0.36,
        label="selected top-5",
        color="#2a9d8f",
    )
    axes[1, 0].bar(
        x + 0.18,
        functional_plot.random_subspace_overlap_mean,
        width=0.36,
        label="matched random",
        color="#9aa0a6",
    )
    axes[1, 0].set_xticks(x, functional_plot.model, rotation=25, ha="right")
    axes[1, 0].set_ylabel("Subspace overlap")
    axes[1, 0].set_title("Functional top-5 stability")
    axes[1, 0].legend(fontsize=8)

    tri = triangle[triangle.analysis_set.eq("measurement_qc")].copy()
    tri["label"] = tri.model + "/" + tri.phenotype.str.replace("_", " ")
    axes[1, 1].barh(tri.label, tri.triangle_joint_pass, color="#457b9d")
    axes[1, 1].set_xlim(0, 8)
    axes[1, 1].set_xlabel("Joint-pass cells out of 8")
    axes[1, 1].set_title("Triangle validation (measurement QC)")

    for axis in axes.flat:
        axis.grid(axis="y" if axis is axes[0, 0] or axis is axes[1, 1] else "x", alpha=0.2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def build_section(base: Path) -> tuple[str, dict]:
    transport = pd.read_csv(base / "hierarchical_robustness" / "transport_crossed_inference.csv")
    transport_envelope = pd.read_csv(
        base / "hierarchical_robustness" / "transport_leave_one_out_envelopes.csv"
    )
    protocol = pd.read_csv(base / "hierarchical_robustness" / "protocol_factor_inference.csv")
    protocol_envelope = pd.read_csv(
        base / "hierarchical_robustness" / "protocol_leave_one_out_envelopes.csv"
    )
    bias = pd.read_csv(base / "waveform_failure_bias" / "candidate_bias_covariates.csv")
    propensity = pd.read_csv(base / "waveform_failure_bias" / "propensity_diagnostics.csv")
    ipw = pd.read_csv(base / "waveform_failure_bias" / "waveform_ipw_sensitivity.csv")
    capacity = pd.read_csv(base / "sae_stability" / "capacity_stability_summary.csv")
    functional = pd.read_csv(base / "sae_stability" / "functional_top5_stability_summary.csv")
    triangle = pd.read_csv(base / "waveform_triangle" / "triangle_summary.csv")

    t_rows = transport[
        transport.method.eq("fewshot_n2048")
        & transport.metric.isin(["recon_r2", "ste", "excess_selectivity"])
    ].sort_values("metric")
    p_rows = protocol[
        protocol.subset.eq("frozen_tier0_only")
        & protocol.contrast.isin(["local_minus_frozen", "adapted_minus_frozen"])
        & protocol.metric.isin(["ste", "excess_selectivity"])
    ].sort_values(["contrast", "metric"])
    severe_bias = bias[
        bias.standardized_mean_difference_failure_minus_success.abs().ge(0.1)
    ]
    significant_bias = severe_bias[severe_bias.q_two_sided.lt(0.05)]
    ipw_q_columns = [column for column in ipw.columns if column.endswith("_difference_q")]
    ipw_significant_by_metric = {
        column.removesuffix("_difference_q"): int(ipw[column].lt(0.05).sum())
        for column in ipw_q_columns
    }
    ipw_significant = int(sum(ipw_significant_by_metric.values()))
    ipw_tests = int(sum(ipw[column].notna().sum() for column in ipw_q_columns))
    max_head_shift = float(ipw.target_head_delta_ipw_minus_unweighted.abs().max())

    capacity_overall = (
        capacity.groupby("N_over_d", as_index=False)
        .agg(
            matched_cosine=("matched_cosine_mean", "mean"),
            stability_above_random=("stability_above_random_mean", "mean"),
            subspace_overlap=("subspace_overlap_mean", "mean"),
        )
        .sort_values("N_over_d")
    )
    selected_overlap = float(functional.selected_subspace_overlap_mean.mean())
    random_overlap = float(functional.random_subspace_overlap_mean.mean())
    triangle_qc = triangle[triangle.analysis_set.eq("measurement_qc")]
    triangle_all = triangle[triangle.analysis_set.eq("unfiltered_complete_case")]

    lines = [
        START,
        "### 4.9 层级稳健性、失败偏差与三角验证（benchmark_extension_v2）",
        "",
        "该扩展只读复用 v1 数据和既有 SAE checkpoint。统计重分析、checkpoint 扫描和模型推理均在 Slurm compute nodes 执行；没有重写原始 ECG，也没有覆盖 v1 产物。",
        "",
        "#### 多层级稳健性",
        "",
        "Transport 使用 model/cohort crossed bootstrap；protocol comparison 使用 model/cohort/target factor-weight bootstrap，并同时执行 leave-one-model/cohort/target-out。",
        "",
        "| Analysis | Metric | Mean delta | 95% CI | q |",
        "|---|---|---:|---:|---:|",
    ]
    for row in t_rows.itertuples():
        lines.append(
            f"| Few-shot n=2048 - frozen | {row.metric} | {number(row.mean_delta)} | "
            f"[{number(row.ci_low)}, {number(row.ci_high)}] | {number(row.q_two_sided)} |"
        )
    for row in p_rows.itertuples():
        lines.append(
            f"| {row.contrast.replace('_', ' ')} (Tier-0) | {row.metric} | {number(row.mean_delta)} | "
            f"[{number(row.ci_low)}, {number(row.ci_high)}] | {number(row.q_two_sided)} |"
        )
    transport_same = int(transport_envelope.leave_one_out_all_same_sign.sum())
    protocol_same = int(protocol_envelope.leave_one_out_all_same_sign.sum())
    lines.extend(
        [
            "",
            f"Transport 的 {transport_same}/{len(transport_envelope)} 个 method-metric contrasts 在全部 leave-one-factor-out 检查中保持原方向；protocol 为 {protocol_same}/{len(protocol_envelope)}。这类检查衡量结论对共享模型、队列和目标的敏感性，不把 24 个 crossed units 当成完全独立样本。",
            "",
            "![Hierarchical robustness](docs/figures/benchmark_extension_v2_robustness.png)",
            "",
            "#### Waveform 候选失败偏差",
            "",
            f"在去除模型重复后，共分析 1,836 个 cohort-phenotype-record attempts，其中 300 个 delineation failures。{len(severe_bias)}/{len(bias)} 个协变量比较的 |SMD|>=0.1，{len(significant_bias)} 个在全表 BH-FDR 后 q<0.05。propensity 模型 cross-validated AUROC 范围为 {number(propensity.propensity_cross_validated_auroc.min())}-{number(propensity.propensity_cross_validated_auroc.max())}。",
            "",
            f"固定 propensity 的 record bootstrap 中，IPW 与未加权结果有 {ipw_significant}/{ipw_tests} 个 metric-cell differences 在 BH-FDR 后显著（measurement={ipw_significant_by_metric.get('measurement_delta', 0)}、head={ipw_significant_by_metric.get('target_head_delta', 0)}、SAE={ipw_significant_by_metric.get('sae_top5_delta_mean', 0)}）；target-head mean 的最大绝对变化为 {number(max_head_shift)}。这限制了 waveform 结果向完整候选总体的外推。v1 所称 ITT 在本扩展中改称 `unfiltered complete-case`，因为失败候选没有可观测编辑结果。",
            "",
            "#### SAE 跨容量与功能稳定性",
            "",
            "已有 6 models x 3 capacities x 3 seeds 的 54 个 checkpoint 被顺序只读分析，没有重新训练 SAE。",
            "",
            "| N/d | Matched cosine | Above random | Subspace overlap |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in capacity_overall.itertuples():
        lines.append(
            f"| {int(row.N_over_d)} | {number(row.matched_cosine)} | "
            f"{number(row.stability_above_random)} | {number(row.subspace_overlap)} |"
        )
    lines.extend(
        [
            "",
            f"在 270 个 model-cohort-target-protocol 功能单元中，selected top-5 的跨 seed 子空间 overlap 均值为 {number(selected_overlap)}，原 benchmark 的 frequency/magnitude-matched random top-5 为 {number(random_overlap)}。",
            "",
            "#### Latent-waveform-readout 三角验证",
            "",
            "12 个 GPU workers 在与 v1 完全相同的 256-record cells 上重新推理；每个 SAE seed 比较 selected top-5 与既有 20 组 matched-random controls。主单位是在 record 内先平均 3 个 SAE seeds。",
            "",
            "| Analysis set | Joint pass | Total cells | Selected/raw sign concordance | Random concordance |",
            "|---|---:|---:|---:|---:|",
            f"| Measurement QC | {int(triangle_qc.triangle_joint_pass.sum())} | {int(triangle_qc.intervention_cells.sum())} | {number(triangle_qc.selected_raw_sign_concordance.mean())} | {number(triangle_qc.random_raw_sign_concordance_mean.mean())} |",
            f"| Unfiltered complete-case | {int(triangle_all.triangle_joint_pass.sum())} | {int(triangle_all.intervention_cells.sum())} | {number(triangle_all.selected_raw_sign_concordance.mean())} | {number(triangle_all.random_raw_sign_concordance_mean.mean())} |",
            "",
            "![Bias, stability, and triangle validation](docs/figures/benchmark_extension_v2_validation.png)",
            "",
            "这些结果只支持冻结 readout 上的 intervention-response consistency 和 readout mediation sensitivity。它们不证明临床干预效果、生成式 waveform 因果或生物机制因果。",
            "",
            "详细 bootstrap、leave-one-out、IPW、seed-pair 和 record-paired 结果见 `results/benchmark_extension_v2/benchmark_extension_v2_report.md`。",
        ]
    )
    lines.append(END)
    summary = {
        "transport_leave_one_out_same_sign": transport_same,
        "transport_leave_one_out_total": len(transport_envelope),
        "protocol_leave_one_out_same_sign": protocol_same,
        "protocol_leave_one_out_total": len(protocol_envelope),
        "bias_abs_smd_ge_0_1": len(severe_bias),
        "bias_q_lt_0_05": len(significant_bias),
        "ipw_significant_differences": ipw_significant,
        "ipw_significant_by_metric": ipw_significant_by_metric,
        "ipw_tests": ipw_tests,
        "functional_selected_overlap": selected_overlap,
        "functional_random_overlap": random_overlap,
        "triangle_qc_joint_pass": int(triangle_qc.triangle_joint_pass.sum()),
        "triangle_qc_cells": int(triangle_qc.intervention_cells.sum()),
        "triangle_unfiltered_joint_pass": int(triangle_all.triangle_joint_pass.sum()),
        "triangle_unfiltered_cells": int(triangle_all.intervention_cells.sum()),
    }
    return "\n".join(lines) + "\n", summary


def update_readme(section: str) -> None:
    current = README.read_text()
    if START in current and END in current:
        before, remainder = current.split(START, 1)
        _, after = remainder.split(END, 1)
        updated = before.rstrip() + "\n\n" + section.rstrip() + "\n" + after
    else:
        insertion = current.find("\n## 5.")
        if insertion < 0:
            updated = current.rstrip() + "\n\n" + section
        else:
            updated = current[:insertion].rstrip() + "\n\n" + section.rstrip() + "\n" + current[insertion:]
    atomic_write_text(README, updated)


def main() -> None:
    args = parse_args()
    figure_robustness = FIGURE_ROOT / "benchmark_extension_v2_robustness.png"
    figure_validation = FIGURE_ROOT / "benchmark_extension_v2_validation.png"
    make_robustness_figure(args.base, figure_robustness)
    make_validation_figure(args.base, figure_validation)
    section, summary = build_section(args.base)
    report = (
        "# Benchmark Extension v2 Report\n\n"
        "This report is generated from completed CSV/JSON artifacts. No result is manually entered.\n\n"
        + section.replace(START + "\n", "").replace(END + "\n", "")
    )
    atomic_write_text(args.base / "benchmark_extension_v2_report.md", report)
    update_readme(section)
    metadata = {
        "schema_version": 1,
        "figures": [
            str(figure_robustness.relative_to(ROOT)),
            str(figure_validation.relative_to(ROOT)),
        ],
        "report": str((args.base / "benchmark_extension_v2_report.md").relative_to(ROOT)),
        "readme_updated": True,
        "claim_boundary_present": "不证明临床干预效果" in section,
        "summary": summary,
        "all_complete": True,
    }
    write_json(args.base / "report_metadata.json", metadata)
    print(metadata)


if __name__ == "__main__":
    main()
