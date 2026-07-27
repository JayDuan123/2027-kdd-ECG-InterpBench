from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from statistics import mean

from .config import (
    FEATURE_FILES,
    LABEL_FILES,
    ROOT,
    PTBXL_METADATA_ROOT,
    PTBXL_PLUS_ROOT,
    PTBXL_WAVEFORM_ROOT,
    expand_source_columns,
    load_concepts,
    load_tasks,
)


DEFAULT_OUT_DIR = ROOT / "results" / "manifest"
SPLIT_SEED = "ecg_fm_interpretability_benchmark_v1"

TASK_TOKENS = {
    "ptbxl_norm": {"NORM"},
    "ptbxl_mi": {"MI"},
    "ptbxl_sttc": {"STTC"},
    "ptbxl_cd": {"CD"},
    "ptbxl_hyp": {"HYP"},
    "mi_ischemia": {"MI", "IMI", "AMI", "ASMI", "ILMI", "IPLMI", "IPMI", "LMI", "PMI", "ISC_", "ISCA", "ISCI", "ISCAL"},
    "bbb_conduction": {"CD", "CLBBB", "CRBBB", "ILBBB", "IRBBB", "LAFB", "LPFB", "IVCD"},
    "hypertrophy": {"HYP", "LVH", "RVH", "LAO/LAE", "RAO/RAE"},
    "af_rhythm": {"AFIB", "AFLT"},
}


@dataclass(frozen=True)
class ManifestPaths:
    out_dir: Path
    concepts_matrix: Path
    tasks_matrix: Path
    split: Path
    concept_summary: Path
    report: Path
    provenance_report: Path


def manifest_paths(out_dir: Path = DEFAULT_OUT_DIR) -> ManifestPaths:
    return ManifestPaths(
        out_dir=out_dir,
        concepts_matrix=out_dir / "concepts_matrix.csv",
        tasks_matrix=out_dir / "tasks_matrix.csv",
        split=out_dir / "split.csv",
        concept_summary=out_dir / "concept_summary.csv",
        report=out_dir / "manifest_report.md",
        provenance_report=out_dir / "provenance_report.md",
    )


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def signed_absmax(values: list[float]) -> float | None:
    if not values:
        return None
    return max(values, key=lambda x: abs(x))


def angular_difference(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    diff = abs((a - b + 180.0) % 360.0 - 180.0)
    return diff


def read_feature_values() -> dict[str, dict[str, dict[str, str]]]:
    data: dict[str, dict[str, dict[str, str]]] = {}
    for name, path in FEATURE_FILES.items():
        if not path.exists():
            continue
        rows: dict[str, dict[str, str]] = {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ecg_id = row.get("ecg_id")
                if ecg_id:
                    rows[ecg_id] = row
        data[name] = rows
    return data


def read_csv_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        return {row["ecg_id"] for row in csv.DictReader(f) if row.get("ecg_id")}


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def waveform_file_counts() -> dict[str, int]:
    group_dirs = [p for p in PTBXL_WAVEFORM_ROOT.glob("g*") if p.is_dir()]
    all_files = 0
    hea_files = 0
    mat_files = 0
    records_files = 0
    for group in group_dirs:
        for path in group.rglob("*"):
            if not path.is_file():
                continue
            all_files += 1
            if path.suffix == ".hea":
                hea_files += 1
            elif path.suffix == ".mat":
                mat_files += 1
            elif path.name == "RECORDS":
                records_files += 1
    return {
        "group_dirs": len(group_dirs),
        "all_files": all_files,
        "hea_files": hea_files,
        "mat_files": mat_files,
        "records_files": records_files,
    }


def collect_provenance() -> dict[str, object]:
    ptbxl_ids = read_csv_ids(LABEL_FILES["ptbxl_database"])
    plus_feature_ids = read_csv_ids(FEATURE_FILES["12sl_features.csv"])
    plus_ecgdeli_ids = read_csv_ids(FEATURE_FILES["ecgdeli_features.csv"])
    plus_label_ids = read_csv_ids(LABEL_FILES["ptbxl_statements"])
    aligned_ids = ptbxl_ids & plus_feature_ids & plus_ecgdeli_ids & plus_label_ids
    union_ids = ptbxl_ids | plus_feature_ids | plus_ecgdeli_ids | plus_label_ids
    return {
        "ptbxl_waveform_root": str(PTBXL_WAVEFORM_ROOT),
        "ptbxl_metadata_root": str(PTBXL_METADATA_ROOT),
        "ptbxl_plus_root": str(PTBXL_PLUS_ROOT),
        "waveform_counts": waveform_file_counts(),
        "ptbxl_metadata_rows": count_csv_rows(LABEL_FILES["ptbxl_database"]),
        "ptbxl_plus_12sl_rows": count_csv_rows(FEATURE_FILES["12sl_features.csv"]),
        "ptbxl_plus_ecgdeli_rows": count_csv_rows(FEATURE_FILES["ecgdeli_features.csv"]),
        "ptbxl_plus_statement_rows": count_csv_rows(LABEL_FILES["ptbxl_statements"]),
        "ptbxl_metadata_ids": len(ptbxl_ids),
        "ptbxl_plus_12sl_ids": len(plus_feature_ids),
        "ptbxl_plus_ecgdeli_ids": len(plus_ecgdeli_ids),
        "ptbxl_plus_statement_ids": len(plus_label_ids),
        "aligned_ids": len(aligned_ids),
        "union_ids": len(union_ids),
        "unmatched_ids": len(union_ids - aligned_ids),
        "is_exact_id_match": bool(union_ids) and len(aligned_ids) == len(union_ids),
    }


def render_provenance_report(provenance: dict[str, object]) -> str:
    wc = provenance["waveform_counts"]
    assert isinstance(wc, dict)
    lines = [
        "# ECG FM Benchmark v1 Provenance Report",
        "",
        "## Resource Roles",
        "",
        f"- PTB-XL waveform source: {provenance['ptbxl_waveform_root']}",
        f"- PTB-XL metadata/split source: {provenance['ptbxl_metadata_root']}",
        f"- PTB-XL+ concept/statement source: {provenance['ptbxl_plus_root']}",
        "- Join key: exact `ecg_id` match across PTB-XL metadata and PTB-XL+ feature/label tables.",
        "",
        "## PTB-XL Waveform Files",
        "",
        f"- group directories: {wc['group_dirs']}",
        f"- all files under g*: {wc['all_files']}",
        f"- .hea files: {wc['hea_files']}",
        f"- .mat files: {wc['mat_files']}",
        f"- RECORDS files: {wc['records_files']}",
        "",
        "## CSV Rows",
        "",
        f"- PTB-XL metadata rows: {provenance['ptbxl_metadata_rows']}",
        f"- PTB-XL+ 12SL rows: {provenance['ptbxl_plus_12sl_rows']}",
        f"- PTB-XL+ ECGDeli rows: {provenance['ptbxl_plus_ecgdeli_rows']}",
        f"- PTB-XL+ statement rows: {provenance['ptbxl_plus_statement_rows']}",
        "",
        "## ECG ID Alignment",
        "",
        f"- PTB-XL metadata ids: {provenance['ptbxl_metadata_ids']}",
        f"- PTB-XL+ 12SL ids: {provenance['ptbxl_plus_12sl_ids']}",
        f"- PTB-XL+ ECGDeli ids: {provenance['ptbxl_plus_ecgdeli_ids']}",
        f"- PTB-XL+ statement ids: {provenance['ptbxl_plus_statement_ids']}",
        f"- aligned ids: {provenance['aligned_ids']}",
        f"- union ids: {provenance['union_ids']}",
        f"- unmatched ids: {provenance['unmatched_ids']}",
        f"- exact id match: {'yes' if provenance['is_exact_id_match'] else 'no'}",
        "",
    ]
    return "\n".join(lines)


def build_concept_row(ecg_id: str, feature_data: dict[str, dict[str, dict[str, str]]]) -> dict[str, str]:
    out = {"ecg_id": ecg_id}
    for concept in load_concepts():
        if concept.main.lower() != "yes":
            continue
        source_rows = feature_data.get(concept.source_file, {})
        source = source_rows.get(ecg_id, {})
        cols = expand_source_columns(concept.source_columns)
        vals = [parse_float(source.get(col)) for col in cols]
        vals_present = [v for v in vals if v is not None]

        value: float | None
        if concept.aggregation == "global":
            value = vals_present[0] if vals_present else None
        elif concept.aggregation == "lead_absmax":
            value = signed_absmax(vals_present)
        elif concept.aggregation == "derived" and concept.concept_id == "qrst_angle":
            value = angular_difference(vals[0] if len(vals) > 0 else None, vals[1] if len(vals) > 1 else None)
        else:
            value = None

        out[concept.concept_id] = "" if value is None else f"{value:.10g}"
    return out


def parse_statement_tokens(value: str) -> set[str]:
    if not value:
        return set()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return set()
    tokens: set[str] = set()
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, tuple) and item:
                tokens.add(str(item[0]).upper())
            elif isinstance(item, str):
                tokens.add(item.upper())
    return tokens


def read_task_tokens() -> dict[str, set[str]]:
    path = LABEL_FILES["ptbxl_statements"]
    tokens_by_id: dict[str, set[str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ecg_id = row.get("ecg_id")
            if not ecg_id:
                continue
            tokens = set()
            for col in ("scp_codes", "scp_codes_ext"):
                tokens |= parse_statement_tokens(row.get(col, ""))
            tokens_by_id[ecg_id] = tokens
    return tokens_by_id


def read_diagnostic_class_map() -> dict[str, str]:
    path = LABEL_FILES["scp_statements"]
    class_by_token: dict[str, str] = {}
    if not path.exists():
        return class_by_token
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token = (row.get("") or row.get("Unnamed: 0") or "").upper()
            diagnostic = row.get("diagnostic")
            diagnostic_class = row.get("diagnostic_class", "").upper()
            if token and diagnostic in {"1.0", "1"} and diagnostic_class:
                class_by_token[token] = diagnostic_class
    return class_by_token


def read_patient_ids() -> dict[str, str]:
    path = LABEL_FILES["ptbxl_database"]
    if not path.exists():
        return {}
    patient_by_ecg: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ecg_id = row.get("ecg_id")
            patient_id = row.get("patient_id")
            if ecg_id and patient_id:
                patient_by_ecg[ecg_id] = patient_id
    return patient_by_ecg


def build_task_row(ecg_id: str, tokens_by_id: dict[str, set[str]], diagnostic_class_by_token: dict[str, str]) -> dict[str, str]:
    tokens = tokens_by_id.get(ecg_id, set())
    diagnostic_classes = {diagnostic_class_by_token[token] for token in tokens if token in diagnostic_class_by_token}
    out = {"ecg_id": ecg_id}
    for task in load_tasks():
        if task.main.lower() not in {"yes", "conditional"}:
            continue
        wanted = TASK_TOKENS.get(task.task_id, {task.diagnostic_label.upper()})
        out[task.task_id] = "1" if (tokens & wanted) or (diagnostic_classes & wanted) else "0"
    return out


def split_for_key(key: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SEED}:{key}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < 0.70:
        return "train"
    if bucket < 0.85:
        return "val"
    return "test"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def concept_summary(concept_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    concepts = [c for c in load_concepts() if c.main.lower() == "yes"]
    rows: list[dict[str, str]] = []
    n = len(concept_rows)
    for concept in concepts:
        values = [parse_float(row.get(concept.concept_id)) for row in concept_rows]
        present = [v for v in values if v is not None]
        rows.append(
            {
                "concept_id": concept.concept_id,
                "family": concept.family,
                "n": str(n),
                "non_missing": str(len(present)),
                "missing_frac": f"{1.0 - len(present) / n:.6f}" if n else "",
                "mean": f"{mean(present):.10g}" if present else "",
                "min": f"{min(present):.10g}" if present else "",
                "max": f"{max(present):.10g}" if present else "",
            }
        )
    return rows


def render_report(paths: ManifestPaths, n: int, split_counts: dict[str, int], patient_level: bool) -> str:
    lines = [
        "# ECG FM Benchmark v1 Manifest Report",
        "",
        f"Rows: {n}",
        f"Patient-level split: {'yes' if patient_level else 'no'}",
        "",
        "## Outputs",
        "",
        f"- concepts_matrix: {paths.concepts_matrix}",
        f"- tasks_matrix: {paths.tasks_matrix}",
        f"- split: {paths.split}",
        f"- concept_summary: {paths.concept_summary}",
        f"- provenance_report: {paths.provenance_report}",
        "",
        "## Split Counts",
        "",
    ]
    for name in ("train", "val", "test"):
        lines.append(f"- {name}: {split_counts.get(name, 0)}")
    if not patient_level:
        lines.extend(
            [
                "",
                "## Warning",
                "",
                "No exact PTB-XL metadata/PTB-XL+ ecg_id match was available, so this scaffold uses a deterministic ecg_id-level split.",
                "Replace this with a patient-level split before main benchmark experiments.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_manifest(out_dir: Path = DEFAULT_OUT_DIR) -> ManifestPaths:
    paths = manifest_paths(out_dir)
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    feature_data = read_feature_values()
    tokens_by_id = read_task_tokens()
    diagnostic_class_by_token = read_diagnostic_class_map()
    patient_by_ecg = read_patient_ids()
    provenance = collect_provenance()
    ecg_ids = sorted(set(tokens_by_id) & set(feature_data.get("12sl_features.csv", {})) & set(feature_data.get("ecgdeli_features.csv", {})), key=lambda x: int(x))

    concept_rows = [build_concept_row(ecg_id, feature_data) for ecg_id in ecg_ids]
    task_rows = [build_task_row(ecg_id, tokens_by_id, diagnostic_class_by_token) for ecg_id in ecg_ids]
    patient_level = bool(provenance["is_exact_id_match"]) and bool(patient_by_ecg) and all(ecg_id in patient_by_ecg for ecg_id in ecg_ids)
    split_rows = []
    for ecg_id in ecg_ids:
        split_key = patient_by_ecg[ecg_id] if patient_level else ecg_id
        split_rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": patient_by_ecg.get(ecg_id, ""),
                "split": split_for_key(split_key),
                "patient_level": "true" if patient_level else "false",
            }
        )

    concept_fields = ["ecg_id"] + [c.concept_id for c in load_concepts() if c.main.lower() == "yes"]
    task_fields = ["ecg_id"] + [t.task_id for t in load_tasks() if t.main.lower() in {"yes", "conditional"}]
    write_csv(paths.concepts_matrix, concept_rows, concept_fields)
    write_csv(paths.tasks_matrix, task_rows, task_fields)
    write_csv(paths.split, split_rows, ["ecg_id", "patient_id", "split", "patient_level"])
    write_csv(paths.concept_summary, concept_summary(concept_rows), ["concept_id", "family", "n", "non_missing", "missing_frac", "mean", "min", "max"])

    split_counts: dict[str, int] = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    paths.report.write_text(render_report(paths, len(ecg_ids), split_counts, patient_level=patient_level), encoding="utf-8")
    paths.provenance_report.write_text(render_provenance_report(provenance), encoding="utf-8")
    return paths
