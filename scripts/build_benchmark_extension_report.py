#!/usr/bin/env python
"""Build audited figures/report and update README for benchmark_extension_v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = ROOT / "results" / "benchmark_extension_v1"
FIGURES = ROOT / "docs" / "figures"
README = ROOT / "README.md"
START = "<!-- BENCHMARK_EXTENSION_V1_START -->"
END = "<!-- BENCHMARK_EXTENSION_V1_END -->"

METHOD_LABELS = {
    "sae_top5": "SAE top-5",
    "supervised_cav": "Supervised CAV",
    "pca_top5": "PCA top-5",
    "random_orthogonal_5d": "Random orthogonal 5D",
    "frozen": "Frozen",
    "diagonal_full_train": "Diagonal/full-train",
    "coral_full_train": "CORAL/full-train",
    "fewshot_n128": "Few-shot n=128",
    "fewshot_n512": "Few-shot n=512",
    "fewshot_n2048": "Few-shot n=2048",
    "cohort_adapted_full": "Cohort-adapted/full",
}
CONTRAST_LABELS = {
    "local_minus_frozen": "Local - Frozen",
    "adapted_minus_frozen": "Adapted - Frozen",
    "adapted_minus_local": "Adapted - Local",
}
ROLE_LABELS = {
    "robust": "Robust (5 pairs)",
    "strict_null": "Strict null (3 pairs)",
    "near_null_tier2_le1": "Near-null (2 pairs)",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def f4(value: float) -> str:
    return f"{float(value):.4f}"


def audit_inputs() -> dict[str, dict]:
    metadata_paths = {
        "paired": BASE / "paired_protocols" / "metadata.json",
        "dose": BASE / "dose_direction" / "metadata.json",
        "baseline": BASE / "baseline_controls" / "metadata.json",
        "transport": BASE / "transport_ladder" / "metadata.json",
        "waveform": BASE / "waveform_interventions" / "metadata.json",
    }
    metadata = {name: load_json(path) for name, path in metadata_paths.items()}
    for name, item in metadata.items():
        if not item.get("all_complete", False):
            raise RuntimeError(f"{name} metadata is not complete: {item}")
    expected = {
        BASE / "paired_protocols" / "paired_protocol_seed_cells.csv": 810,
        BASE / "dose_direction" / "dose_direction_seed_cells.csv": 1200,
        BASE / "baseline_controls" / "baseline_method_seed_cells.csv": 120,
        BASE / "baseline_controls" / "baseline_paired_contrasts.csv": 90,
        BASE / "transport_ladder" / "transport_quality_seed_cells.csv": 504,
        BASE / "transport_ladder" / "transport_steering_seed_cells.csv": 1890,
        BASE / "transport_ladder" / "transport_paired_inference.csv": 54,
        BASE / "waveform_interventions" / "waveform_paired_records.csv": 12288,
        BASE / "waveform_interventions" / "waveform_intervention_profile.csv": 48,
    }
    for path, rows in expected.items():
        actual = len(pd.read_csv(path))
        if actual != rows:
            raise RuntimeError(f"{path}: expected {rows} rows, found {actual}")
    return metadata


def build_figures(data: dict[str, pd.DataFrame]) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    dose = data["dose_profile"]
    dose = dose[(dose.top_k == 5) & dose["mode"].eq("centroid_scale")]
    dose_group = dose.groupby(["panel_role", "alpha"], as_index=False).agg(
        signed_change=("signed_target_change_mean", "mean"),
        behavior=("behavior_excess_mean", "mean"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = {"robust": "#167D8D", "strict_null": "#B54A4A", "near_null_tier2_le1": "#6E6E6E"}
    for role, group in dose_group.groupby("panel_role"):
        group = group.sort_values("alpha")
        axes[0].plot(group.alpha, group.signed_change, marker="o", label=ROLE_LABELS[role], color=colors[role])
        axes[1].plot(group.alpha, group.behavior, marker="o", label=ROLE_LABELS[role], color=colors[role])
    axes[0].axhline(0, color="#222222", linewidth=.7)
    axes[1].axhline(0, color="#222222", linewidth=.7)
    axes[0].set(title="Top-5 dose: signed target-logit change", xlabel="Centroid alpha", ylabel="Standardized signed change")
    axes[1].set(title="Top-5 dose: behavior excess", xlabel="Centroid alpha", ylabel="Behavior excess")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    dose_path = FIGURES / "benchmark_extension_dose.png"
    fig.savefig(dose_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    paired = data["paired_stratified"]
    paired = paired[paired.analysis_stratum.eq("frozen_tier0_only")].copy()
    baseline = data["baseline_methods"]
    baseline = baseline[baseline.panel_role.eq("robust")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(paired)); width = .24
    for offset, (metric, label, color) in enumerate(
        [("delta_ste_mean", "Delta STE", "#167D8D"),
         ("delta_excess_selectivity_mean", "Delta excess selectivity", "#D38B2C"),
         ("delta_behavior_excess_mean", "Delta behavior excess", "#8A4F7D")]
    ):
        axes[0].bar(x + (offset - 1) * width, paired[metric], width, label=label, color=color)
    axes[0].axhline(0, color="#222222", linewidth=.7)
    axes[0].set_xticks(x, [CONTRAST_LABELS[value] for value in paired.contrast], rotation=15, ha="right")
    axes[0].set_title("Paired protocol deltas: Tier-0 cells")
    axes[0].legend(frameon=False, fontsize=8)
    baseline["label"] = baseline.method.map(METHOD_LABELS)
    x = np.arange(len(baseline))
    axes[1].bar(x - .18, baseline.ste_mean, .36, label="STE", color="#167D8D")
    axes[1].bar(x + .18, baseline.otd_mean, .36, label="OTD", color="#B54A4A")
    axes[1].set_xticks(x, baseline.label, rotation=20, ha="right")
    axes[1].set_title("Norm-matched controls: robust panel")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    protocol_path = FIGURES / "benchmark_extension_protocol_controls.png"
    fig.savefig(protocol_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    quality = data["transport_quality"].copy()
    waveform = data["waveform_summary"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    quality["label"] = quality.method.map(METHOD_LABELS)
    axes[0].barh(quality.label, quality.recon_r2_mean, color="#3D7D5A")
    axes[0].axvline(.85, color="#B54A4A", linestyle="--", linewidth=1, label="R2 gate 0.85")
    axes[0].set(xlabel="Mean held-out reconstruction R2", title="24-pair transport ladder")
    axes[0].legend(frameon=False, fontsize=8)
    waveform["cell"] = waveform.model + " / " + waveform.phenotype.str.replace("_", " ")
    x = np.arange(len(waveform))
    axes[1].bar(x - .18, waveform.joint_grounding_pass, .36, label="QC joint pass", color="#167D8D")
    axes[1].bar(x + .18, waveform.itt_joint_grounding_pass, .36, label="ITT joint pass", color="#D38B2C")
    axes[1].set_xticks(x, waveform.cell, rotation=35, ha="right")
    axes[1].set(ylabel="Intervention cells (of 8)", title="Waveform grounding")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    transport_path = FIGURES / "benchmark_extension_transport_waveform.png"
    fig.savefig(transport_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [dose_path, protocol_path, transport_path]


def make_section(metadata: dict[str, dict], data: dict[str, pd.DataFrame]) -> str:
    paired = data["paired_stratified"]
    tier0 = paired[paired.analysis_stratum.eq("frozen_tier0_only")]
    paired_rows = [
        [CONTRAST_LABELS[row.contrast], int(row.seed_cells), f4(row.delta_ste_mean),
         f4(row.delta_excess_selectivity_mean), f4(row.delta_behavior_excess_mean),
         int(row.selectivity_q05_cells)]
        for row in tier0.itertuples(index=False)
    ]
    local_tier0 = tier0[tier0.contrast.eq("local_minus_frozen")].iloc[0]
    dose = data["dose_profile"]
    dose = dose[(dose.top_k == 5) & dose["mode"].eq("centroid_scale") & np.isclose(dose.alpha, 1.0)]
    dose = dose.groupby("panel_role", as_index=False).agg(
        target_pairs=("target", "size"), signed_change=("signed_target_change_mean", "mean"),
        excess=("excess_selectivity_mean", "mean"), behavior=("behavior_excess_mean", "mean"),
    )
    dose_rows = [
        [ROLE_LABELS[row.panel_role], int(row.target_pairs), f4(row.signed_change), f4(row.excess), f4(row.behavior)]
        for row in dose.itertuples(index=False)
    ]
    baseline = data["baseline_methods"]
    baseline = baseline[baseline.panel_role.eq("robust")]
    best_baseline_wbi = baseline.sort_values("wbi_mean").iloc[0]
    baseline_rows = [
        [METHOD_LABELS[row.method], int(row.target_pairs), f4(row.ste_mean), f4(row.otd_mean),
         f4(row.selectivity_margin_mean), f4(row.wbi_mean)]
        for row in baseline.itertuples(index=False)
    ]
    transport = data["transport_quality"]
    best_transport = transport.sort_values("recon_r2_mean", ascending=False).iloc[0]
    transport_inference = data["transport_inference"]
    n2048_r2 = transport_inference.query("method == 'fewshot_n2048' and metric == 'recon_r2'").iloc[0]
    n2048_ste = transport_inference.query("method == 'fewshot_n2048' and metric == 'ste'").iloc[0]
    n2048_selectivity = transport_inference.query(
        "method == 'fewshot_n2048' and metric == 'excess_selectivity'"
    ).iloc[0]
    adapted_ste = transport_inference.query(
        "method == 'cohort_adapted_full' and metric == 'ste'"
    ).iloc[0]
    coral_quality = transport[transport.method.eq("coral_full_train")].iloc[0]
    transport_rows = [
        [METHOD_LABELS[row.method], f4(row.recon_r2_mean), f4(row.dead_fraction_mean),
         f4(row.readout_retention_median), f"{int(row.recon_pass_085)}/72"]
        for row in transport.itertuples(index=False)
    ]
    waveform = data["waveform_summary"]
    qc_joint_total = int(waveform.joint_grounding_pass.sum())
    itt_joint_total = int(waveform.itt_joint_grounding_pass.sum())
    waveform_rows = [
        [row.model, row.phenotype, f4(row.qc_pass_fraction),
         f"{int(row.joint_grounding_pass)}/{int(row.intervention_cells)}",
         f"{int(row.itt_joint_grounding_pass)}/{int(row.intervention_cells)}"]
        for row in waveform.itertuples(index=False)
    ]
    return "\n".join(
        [
            START,
            "### 4.8 扩展稳健性实验（benchmark_extension_v1）",
            "",
            "扩展实验在原 810-cell steering benchmark 冻结后执行。所有大规模 bootstrap、SAE adaptation 和 waveform inference 均在 Slurm compute nodes 上运行；原始 ECG、activation 和既有结果只读。五组实验均通过预期行数与 metadata 审计。",
            "",
            "#### Paired protocol comparison",
            "",
            "270 个相同 `model-cohort-target-seed` cells 使用完全相同的 patient/record bootstrap weights 做三种协议的配对差值。下表只报告 Frozen-Atom Tier-0 可解释的 42 cells；全 810 contrast rows 仍保存在结果文件中。",
            "",
            md_table(["Contrast", "Tier-0 cells", "Delta STE", "Delta excess selectivity", "Delta behavior excess", "Selectivity q<0.05"], paired_rows),
            "",
            f"Tier-0 中 Local-Frozen 的平均 `Delta STE={local_tier0.delta_ste_mean:.4f}`、`Delta excess selectivity={local_tier0.delta_excess_selectivity_mean:.4f}`；分模型结果显示该增益主要来自 ECG-JEPA，不能外推为所有 encoder 的共同规律。",
            "",
            "![Paired protocol and norm-matched controls](docs/figures/benchmark_extension_protocol_controls.png)",
            "",
            "#### Dose、direction 与零消融",
            "",
            "冻结面板包含 5 个 robust、3 个 strict-null 和 2 个 near-null cohort-target pairs；每个 pair 使用 3 seeds、top-k `1/3/5/10`、9 个 centroid alpha 和 zero-ablation，共 1,200 seed-level cells。下表为 top-5、`alpha=1`。",
            "",
            md_table(["Panel", "Target pairs", "Signed target change", "Excess selectivity", "Behavior excess"], dose_rows),
            "",
            f"Top-5 的 `alpha=+/-1` 在 {int(data['dose_monotonic'].query('top_k == 5').alpha_plus_minus_one_sign_reversal.sum())}/10 target pairs 上产生 signed-logit 符号反转。这里的 strict/near-null 是原 benchmark 的 Tier-2 状态分层，不代表所有剂量和所有指标上的绝对零效应。",
            "",
            "![Dose and direction](docs/figures/benchmark_extension_dose.png)",
            "",
            "#### Norm-matched baselines",
            "",
            "同一 10-pair 面板比较 SAE top-5、supervised CAV、PCA top-5 和 20 组 random orthogonal 5D。每条 test record 的 intervention L2 norm 在冻结 head 的标准化 activation space 中与 SAE 精确匹配。下表为 robust panel。",
            "",
            md_table(["Method", "Target pairs", "STE", "OTD", "Selectivity margin", "WBI"], baseline_rows),
            "",
            f"Robust panel 的最低平均 WBI 来自 {METHOD_LABELS[best_baseline_wbi.method]}（{best_baseline_wbi.wbi_mean:.4f}）；supervised CAV 常产生更大的目标效应，但也带来更高 off-target burden，因此不是 SAE 稀疏选择性的等价解释。",
            "",
            "#### 24-pair transport ladder",
            "",
            "6 models x 4 external cohorts x 3 seeds 比较 frozen、full-train diagonal/CORAL、`n=128/512/2048` few-shot adaptation 和 full cohort-adapted SAE。质量与 steering 推断先在 seed/target 内聚合，再以 24 个 model-cohort pairs 为独立单位做 10,000 次 paired bootstrap 和分域 BH-FDR。",
            "",
            md_table(["Method", "Recon R2 mean", "Dead fraction mean", "Readout retention median", "R2>=0.85 seed cells"], transport_rows),
            "",
            f"按 24-pair 平均 reconstruction R2，最高方法为 {METHOD_LABELS[best_transport.method]}（{best_transport.recon_r2_mean:.4f}）；正式比较以 `transport_paired_inference.csv` 的 paired CI/FDR 为准。",
            f"Few-shot `n=2048` 相对 frozen 的 `Delta R2={n2048_r2.mean_delta:+.4f}`，并同时提高 STE（`Delta={n2048_ste.mean_delta:+.4f}, q={n2048_ste.q_two_sided:.4f}`）和 excess selectivity（`Delta={n2048_selectivity.mean_delta:+.4f}, q={n2048_selectivity.q_two_sided:.4f}`）。Full cohort-adapted 的质量最高，但 aggregate STE 95% CI 为 `[{adapted_ste.ci_low:.4f}, {adapted_ste.ci_high:.4f}]`；CORAL 的平均 R2 仅 {coral_quality.recon_r2_mean:.4f}，其低 dead fraction/WBI 不能解释为成功 transport。",
            "",
            "#### Controlled waveform interventions",
            "",
            "ECG-JEPA 与 ECG-FM 在 Chapman/Ningbo 上分别进行 RR irregularity、QRS duration 和 QT interval 的双方向、双剂量干预。每个 worker 保证 256 条完整五变体配对记录；SAE 指标是 selected top-5 latents 对目标 head 的带符号 logit contribution。QC 是达到预注册最小 waveform-measurement change 的 per-protocol 分析，ITT 使用全部记录。",
            f"候选 waveform delineation 失败为 {metadata['waveform']['delineation_failures']}/{metadata['waveform']['candidate_records_examined']}；失败记录完整保留在各 worker 的 `failures.csv`，并通过确定性后备候选补齐样本。",
            "",
            md_table(["Model", "Phenotype", "QC pass fraction", "QC joint pass", "ITT joint pass"], waveform_rows),
            "",
            f"合计 48 个 waveform intervention cells 中，QC joint grounding 通过 {qc_joint_total}/48，ITT joint grounding 通过 {itt_joint_total}/48。",
            "",
            "![Transport ladder and waveform grounding](docs/figures/benchmark_extension_transport_waveform.png)",
            "",
            "这些 waveform 结果只支持受控输入敏感性/grounding，不支持临床干预、生成式 waveform 因果或生物机制因果。详细 CI、FDR、失败记录和复现参数见 `results/benchmark_extension_v1/benchmark_extension_report.md`。",
            "",
            "#### 扩展结果文件",
            "",
            md_table(
                ["Experiment", "Primary artifact"],
                [
                    ["Paired protocols", "`results/benchmark_extension_v1/paired_protocols/paired_protocol_seed_cells.csv`"],
                    ["Dose/direction", "`results/benchmark_extension_v1/dose_direction/dose_direction_seed_cells.csv`"],
                    ["Norm-matched controls", "`results/benchmark_extension_v1/baseline_controls/baseline_paired_contrasts.csv`"],
                    ["Transport ladder", "`results/benchmark_extension_v1/transport_ladder/transport_paired_inference.csv`"],
                    ["Waveform interventions", "`results/benchmark_extension_v1/waveform_interventions/waveform_intervention_profile.csv`"],
                    ["Final audit", "`results/benchmark_extension_v1/final_audit.json`"],
                ],
            ),
            END,
        ]
    )


def write_report(section: str, metadata: dict[str, dict], data: dict[str, pd.DataFrame]) -> Path:
    inference = data["transport_inference"]
    significant = inference[inference.q_two_sided.lt(.05)]
    lines = [
        "# Benchmark Extension v1 Report",
        "",
        "冻结日期：2026-07-13。该报告由完成后的 CSV/JSON 自动生成，不包含人工录入结果。",
        "",
        "## 完整性",
        "",
        f"- Paired protocols: {metadata['paired']['contrast_rows']} contrast rows, {metadata['paired']['bootstrap_samples']} bootstraps/cell。",
        f"- Dose/direction: 1,200 seed cells, {metadata['dose']['panel_targets']} target pairs。",
        f"- Baselines: {metadata['baseline']['method_rows']} method rows, {metadata['baseline']['contrast_rows']} paired contrasts。",
        f"- Transport: {metadata['transport']['workers']} workers, {metadata['transport']['quality_rows']} quality rows, {metadata['transport']['steering_rows']} steering rows。",
        f"- Waveform: {metadata['waveform']['workers']} workers, {metadata['waveform']['raw_variant_rows']} variant rows, {metadata['waveform']['paired_edited_rows']} paired edits；候选 delineation 失败 {metadata['waveform']['delineation_failures']}/{metadata['waveform']['candidate_records_examined']}。",
        "",
        "## README 摘要",
        "",
        section.replace(START + "\n", "").replace("\n" + END, ""),
        "",
        "## Transport paired inference",
        "",
        f"共 {len(inference)} 个 method-metric contrasts；双侧 BH-FDR q<0.05 为 {len(significant)} 个。完整表：`transport_ladder/transport_paired_inference.csv`。",
        "",
        md_table(
            ["Domain", "Method", "Metric", "Mean delta", "95% CI", "q(two-sided)"],
            [[row.domain, METHOD_LABELS[row.method], row.metric, f4(row.mean_delta),
              f"[{f4(row.ci_low)}, {f4(row.ci_high)}]", f4(row.q_two_sided)]
             for row in significant.sort_values(["domain", "metric", "q_two_sided"]).itertuples(index=False)],
        ) if len(significant) else "无 transport contrast 通过双侧 BH-FDR。",
        "",
        "## Claim boundary",
        "",
        "Paired/dose/baseline/transport 结果均是冻结 representation、SAE 或 linear readout 上的受控分析。Waveform intervention 增加输入级 sensitivity evidence，但不是临床或生物机制因果证据。",
    ]
    path = BASE / "benchmark_extension_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_checksums(paths: list[Path]) -> Path:
    output = BASE / "artifact_checksums.sha256"
    rows = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT)}")
    output.write_text("\n".join(rows) + "\n")
    return output


def main() -> None:
    metadata = audit_inputs()
    data = {
        "paired_stratified": pd.read_csv(BASE / "paired_protocols" / "paired_protocol_stratified_summary.csv"),
        "dose_profile": pd.read_csv(BASE / "dose_direction" / "dose_direction_profile.csv"),
        "dose_monotonic": pd.read_csv(BASE / "dose_direction" / "dose_monotonicity_summary.csv"),
        "baseline_methods": pd.read_csv(BASE / "baseline_controls" / "baseline_method_profile.csv"),
        "transport_quality": pd.read_csv(BASE / "transport_ladder" / "transport_quality_profile.csv"),
        "transport_inference": pd.read_csv(BASE / "transport_ladder" / "transport_paired_inference.csv"),
        "waveform_summary": pd.read_csv(BASE / "waveform_interventions" / "waveform_intervention_summary.csv"),
    }
    figures = build_figures(data)
    section = make_section(metadata, data)
    report = write_report(section, metadata, data)
    readme_text = README.read_text()
    if START in readme_text and END in readme_text:
        before, tail = readme_text.split(START, 1)
        _, after = tail.split(END, 1)
        readme_text = before.rstrip() + "\n\n" + section + "\n" + after.lstrip("\n")
    else:
        marker = "## 5. 如何理解这些结果"
        if marker not in readme_text:
            raise RuntimeError(f"README insertion marker missing: {marker}")
        readme_text = readme_text.replace(marker, section + "\n\n" + marker, 1)
    README.write_text(readme_text)

    key_paths = [
        BASE / "paired_protocols" / "paired_protocol_seed_cells.csv",
        BASE / "dose_direction" / "dose_direction_seed_cells.csv",
        BASE / "baseline_controls" / "baseline_method_seed_cells.csv",
        BASE / "transport_ladder" / "transport_quality_seed_cells.csv",
        BASE / "transport_ladder" / "transport_steering_seed_cells.csv",
        BASE / "transport_ladder" / "transport_paired_inference.csv",
        BASE / "waveform_interventions" / "waveform_paired_records.csv",
        BASE / "waveform_interventions" / "waveform_intervention_profile.csv",
        BASE / "waveform_interventions" / "waveform_worker_sample_audit.csv",
        report,
        README,
        *figures,
    ]
    checksums = write_checksums(key_paths)
    combined = {
        "schema_version": 1,
        "all_complete": True,
        "experiments": metadata,
        "figures": [str(path.relative_to(ROOT)) for path in figures],
        "report": str(report.relative_to(ROOT)),
        "readme_updated": True,
        "checksums": str(checksums.relative_to(ROOT)),
    }
    (BASE / "metadata.json").write_text(json.dumps(combined, indent=2) + "\n")
    print(json.dumps({"status": "complete", "report": str(report), "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
