from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import gzip
from pathlib import Path

from .config import ROOT, load_concepts


MIMIC_ECG_ROOT = Path("/rhf/allocations/wq8/mimic_data/mimic-iv-ecg")
MIMIC_IV_HOSP_ROOT = Path("/rhf/allocations/wq8/mimic_data/mimic-iv/hosp")
DEFAULT_OUT_DIR = ROOT / "results" / "multicohort"


MIMIC_MEASUREMENT_CSV = MIMIC_ECG_ROOT / "machine_measurements.csv"
MIMIC_MEASUREMENT_DICTIONARY_CSV = MIMIC_ECG_ROOT / "machine_measurements_data_dictionary.csv"
MIMIC_RECORD_LIST_CSV = MIMIC_ECG_ROOT / "record_list.csv"
MIMIC_WAVEFORM_NOTE_LINKS_CSV = MIMIC_ECG_ROOT / "waveform_note_links.csv"
MIMIC_WAVEFORM_FILES_DIR = MIMIC_ECG_ROOT / "files"
MIMIC_DIAGNOSES_ICD_CSV_GZ = MIMIC_IV_HOSP_ROOT / "diagnoses_icd.csv.gz"
MIMIC_ADMISSIONS_CSV_GZ = MIMIC_IV_HOSP_ROOT / "admissions.csv.gz"
MIMIC_D_ICD_DIAGNOSES_CSV_GZ = MIMIC_IV_HOSP_ROOT / "d_icd_diagnoses.csv.gz"


@dataclass(frozen=True)
class MulticohortPaths:
    out_dir: Path
    mimic_measurement_audit: Path
    concept_crosswalk_track_v: Path
    cohort_gate_summary: Path
    external_task_feasibility: Path
    report: Path


def multicohort_paths(out_dir: Path = DEFAULT_OUT_DIR) -> MulticohortPaths:
    return MulticohortPaths(
        out_dir=out_dir,
        mimic_measurement_audit=out_dir / "mimic_measurement_audit.csv",
        concept_crosswalk_track_v=out_dir / "concept_crosswalk_track_v.csv",
        cohort_gate_summary=out_dir / "cohort_gate_summary.csv",
        external_task_feasibility=out_dir / "external_task_feasibility.csv",
        report=out_dir / "multicohort_gate_report.md",
    )


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return next(csv.reader(f))


def read_measurement_dictionary(path: Path = MIMIC_MEASUREMENT_DICTIONARY_CSV) -> dict[str, str]:
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            variable = (row.get("Variable") or "").strip()
            description = ""
            for key, value in row.items():
                if key != "Variable" and value:
                    description = value.strip()
                    break
            if variable:
                out[variable] = description
    return out


def direct_dictionary_description(column: str, dictionary: dict[str, str]) -> str:
    if column.startswith("report_"):
        return dictionary.get("report_#", "Machine-generated ECG report text.")
    return dictionary.get(column, "")


def mimic_measurement_column_rule(column: str) -> tuple[str, str, str, str, str]:
    timing_cols = {
        "rr_interval": (
            "yes",
            "rr_mean|hr_ventricular|hr_atrial|qtc_bazett|qtc_fridericia|qtc_framingham",
            "msec; derived HR = 60000 / rr_interval; QTc formulas also use RR",
            "yes",
            "",
        ),
        "p_onset": (
            "yes",
            "pr_interval|pq_interval|p_duration_global",
            "msec; used in qrs_onset - p_onset and p_end - p_onset",
            "yes",
            "",
        ),
        "p_end": (
            "yes",
            "p_duration_global",
            "msec; used in p_end - p_onset",
            "yes",
            "",
        ),
        "qrs_onset": (
            "yes",
            "pr_interval|pq_interval|qrs_duration|qt_interval|qtc_bazett|qtc_fridericia|qtc_framingham",
            "msec; used in interval formulas",
            "yes",
            "",
        ),
        "qrs_end": (
            "yes",
            "qrs_duration",
            "msec; qrs_end - qrs_onset",
            "yes",
            "",
        ),
        "t_end": (
            "yes",
            "qt_interval|qtc_bazett|qtc_fridericia|qtc_framingham",
            "msec; t_end - qrs_onset",
            "yes",
            "",
        ),
        "p_axis": ("yes", "p_axis_front", "degrees", "yes", ""),
        "qrs_axis": ("yes", "qrs_axis_front|qrst_angle", "degrees; qrst_angle uses axis difference", "yes", ""),
        "t_axis": ("yes", "t_axis_front|qrst_angle", "degrees; qrst_angle uses axis difference", "yes", ""),
    }
    if column in timing_cols:
        return timing_cols[column]
    if column.startswith("report_"):
        return (
            "no",
            "",
            "",
            "no",
            "report text is allowed only as a secondary label source, never as a measurement concept source",
        )
    if column in {"subject_id", "study_id", "cart_id", "ecg_time"}:
        return ("no", "", "", "no", "identifier/time metadata, not a clinical measurement concept")
    if column in {"bandwidth", "filtering"}:
        return ("no", "", "", "no", "device/acquisition metadata excluded from measurement concept audit")
    return ("no", "", "", "no", "not mapped to the frozen measurement concept lexicon")


def build_mimic_measurement_audit() -> list[dict[str, str]]:
    header = read_csv_header(MIMIC_MEASUREMENT_CSV)
    dictionary = read_measurement_dictionary()
    rows: list[dict[str, str]] = []
    for column in header:
        usable, derived_concept, unit_or_formula, eligible, exclusion = mimic_measurement_column_rule(column)
        rows.append(
            {
                "column": column,
                "description": direct_dictionary_description(column, dictionary),
                "usable": usable,
                "derived_concept": derived_concept,
                "unit_or_formula": unit_or_formula,
                "track_v_eligible": eligible,
                "exclusion_reason": exclusion,
            }
        )
    return rows


TRACK_V_CONCEPTS: dict[str, dict[str, str]] = {
    "hr_ventricular": {
        "mimic_source_column": "rr_interval",
        "derived_formula": "60000 / rr_interval",
        "unit": "bpm",
        "definition_match": "approximate_rate_proxy",
        "approximation_flag": "true",
        "exclusion_reason": "MIMIC vendor table has RR interval but no atrial/ventricular HR distinction.",
    },
    "hr_atrial": {
        "mimic_source_column": "rr_interval",
        "derived_formula": "60000 / rr_interval",
        "unit": "bpm",
        "definition_match": "approximate_rate_proxy",
        "approximation_flag": "true",
        "exclusion_reason": "MIMIC vendor table has RR interval but no atrial/ventricular HR distinction.",
    },
    "rr_mean": {
        "mimic_source_column": "rr_interval",
        "derived_formula": "rr_interval",
        "unit": "msec",
        "definition_match": "single_record_vendor_rr_proxy",
        "approximation_flag": "true",
        "exclusion_reason": "MIMIC provides one vendor RR interval rather than a beat-distribution mean.",
    },
    "pr_interval": {
        "mimic_source_column": "p_onset|qrs_onset",
        "derived_formula": "qrs_onset - p_onset",
        "unit": "msec",
        "definition_match": "derived_interval",
        "approximation_flag": "false",
        "exclusion_reason": "",
    },
    "pq_interval": {
        "mimic_source_column": "p_onset|qrs_onset",
        "derived_formula": "qrs_onset - p_onset",
        "unit": "msec",
        "definition_match": "pr_interval_proxy",
        "approximation_flag": "true",
        "exclusion_reason": "PQ approximated as P onset to QRS onset using available vendor timing points.",
    },
    "p_duration_global": {
        "mimic_source_column": "p_onset|p_end",
        "derived_formula": "p_end - p_onset",
        "unit": "msec",
        "definition_match": "derived_interval",
        "approximation_flag": "false",
        "exclusion_reason": "",
    },
    "qrs_duration": {
        "mimic_source_column": "qrs_onset|qrs_end",
        "derived_formula": "qrs_end - qrs_onset",
        "unit": "msec",
        "definition_match": "derived_interval",
        "approximation_flag": "false",
        "exclusion_reason": "",
    },
    "qt_interval": {
        "mimic_source_column": "qrs_onset|t_end",
        "derived_formula": "t_end - qrs_onset",
        "unit": "msec",
        "definition_match": "qt_like_interval",
        "approximation_flag": "true",
        "exclusion_reason": "MIMIC provides QRS onset and T end; use QT-like interval with explicit definition.",
    },
    "qtc_bazett": {
        "mimic_source_column": "qrs_onset|t_end|rr_interval",
        "derived_formula": "(t_end - qrs_onset) / sqrt(rr_interval / 1000)",
        "unit": "msec",
        "definition_match": "standard_bazett_formula_on_qt_like",
        "approximation_flag": "true",
        "exclusion_reason": "QTc-like because QT is derived from available vendor timing points.",
    },
    "qtc_fridericia": {
        "mimic_source_column": "qrs_onset|t_end|rr_interval",
        "derived_formula": "(t_end - qrs_onset) / (rr_interval / 1000) ** (1/3)",
        "unit": "msec",
        "definition_match": "standard_fridericia_formula_on_qt_like",
        "approximation_flag": "true",
        "exclusion_reason": "QTc-like because QT is derived from available vendor timing points.",
    },
    "qtc_framingham": {
        "mimic_source_column": "qrs_onset|t_end|rr_interval",
        "derived_formula": "(t_end - qrs_onset) + 154 * (1 - rr_interval / 1000)",
        "unit": "msec",
        "definition_match": "standard_framingham_formula_on_qt_like",
        "approximation_flag": "true",
        "exclusion_reason": "QTc-like because QT is derived from available vendor timing points.",
    },
    "p_axis_front": {
        "mimic_source_column": "p_axis",
        "derived_formula": "p_axis",
        "unit": "degrees",
        "definition_match": "direct_axis",
        "approximation_flag": "false",
        "exclusion_reason": "",
    },
    "qrs_axis_front": {
        "mimic_source_column": "qrs_axis",
        "derived_formula": "qrs_axis",
        "unit": "degrees",
        "definition_match": "direct_axis",
        "approximation_flag": "false",
        "exclusion_reason": "",
    },
    "t_axis_front": {
        "mimic_source_column": "t_axis",
        "derived_formula": "t_axis",
        "unit": "degrees",
        "definition_match": "direct_axis",
        "approximation_flag": "false",
        "exclusion_reason": "",
    },
    "qrst_angle": {
        "mimic_source_column": "qrs_axis|t_axis",
        "derived_formula": "abs(wrap_to_180(qrs_axis - t_axis))",
        "unit": "degrees",
        "definition_match": "axis_difference_definition_required",
        "approximation_flag": "true",
        "exclusion_reason": "Track V claim is valid only after PTB-XL is transformed to the same axis-difference definition.",
    },
}


def classify_non_track_v_concept(family: str, concept_id: str) -> tuple[str, str, str]:
    if family in {"AMPLITUDE", "ST_T"}:
        return (
            "track_f_only",
            "yes",
            "MIMIC vendor measurements do not include amplitude, ST/T, area, or per-lead morphology.",
        )
    if concept_id in {"p_found", "rr_iqr"}:
        return (
            "track_f_only",
            "yes",
            "Not present in MIMIC vendor measurements; requires waveform-derived Track F if used externally.",
        )
    return ("not_external_v1", "no", "Not derivable from MIMIC vendor measurements in v1.")


def build_concept_crosswalk_track_v() -> list[dict[str, str]]:
    mimic_header = set(read_csv_header(MIMIC_MEASUREMENT_CSV))
    rows: list[dict[str, str]] = []
    for concept in load_concepts():
        if concept.main != "yes":
            continue
        base = {
            "ptbxl_concept": concept.concept_id,
            "family": concept.family,
            "display_name": concept.display_name,
            "track_v_eligible": "false",
            "track_f_required": "false",
            "category": "",
            "mimic_source_column": "",
            "derived_formula": "",
            "unit": "",
            "definition_match": "",
            "approximation_flag": "false",
            "exclusion_reason": "",
        }
        spec = TRACK_V_CONCEPTS.get(concept.concept_id)
        if spec:
            cols = spec["mimic_source_column"].split("|")
            available = all(col in mimic_header for col in cols)
            base.update(spec)
            base["track_v_eligible"] = "true" if available else "false"
            base["track_f_required"] = "false" if available else "true"
            base["category"] = "track_v_eligible" if available else "not_external_v1"
            if not available:
                missing = [col for col in cols if col not in mimic_header]
                base["exclusion_reason"] = f"Missing MIMIC vendor columns: {'|'.join(missing)}"
        else:
            category, track_f_required, reason = classify_non_track_v_concept(concept.family, concept.concept_id)
            base["category"] = category
            base["track_f_required"] = track_f_required
            base["exclusion_reason"] = reason
        rows.append(base)
    return rows


def build_cohort_gate_summary() -> list[dict[str, str]]:
    track_f_full = DEFAULT_OUT_DIR / "track_f_full" / "waveform_concepts_by_record.csv"
    track_f_closure = DEFAULT_OUT_DIR / "track_f_closure" / "closure_transfer_track_f.csv"
    track_f_smoke = DEFAULT_OUT_DIR / "waveform_smoke_1k" / "waveform_family_gate.csv"
    challenge_g4 = DEFAULT_OUT_DIR / "challenge_native_task_feasibility.csv"

    def track_f_status() -> str:
        if track_f_full.exists() and track_f_closure.exists():
            return "track_f_full_extraction_and_closure_complete"
        if track_f_smoke.exists():
            return "track_f_smoke_complete"
        return "pending_waveform_extraction_smoke"

    def challenge_task_status() -> str:
        return "yes" if challenge_g4.exists() else "pending_g4"

    sae_status = summarize_sae_transfer_status()
    return [
        {
            "cohort_track": "MIMIC-V",
            "waveform_available": "yes" if MIMIC_WAVEFORM_FILES_DIR.exists() else "no",
            "vendor_measurements_available": "yes" if MIMIC_MEASUREMENT_CSV.exists() else "no",
            "vendor_measurement_scope": "interval/rate/axis plus QTc-like and axis-difference QRS-T angle",
            "task_labels_available": "yes" if MIMIC_DIAGNOSES_ICD_CSV_GZ.exists() else "no",
            "primary_label_source": "MIMIC-IV hosp ICD-linked labels",
            "track_status": "passed_with_restricted_scope",
            "sae_transfer_status": "not_applicable_track_v",
        },
        {
            "cohort_track": "MIMIC-F",
            "waveform_available": "yes" if MIMIC_WAVEFORM_FILES_DIR.exists() else "no",
            "vendor_measurements_available": "not_required",
            "vendor_measurement_scope": "not_applicable",
            "task_labels_available": "yes" if MIMIC_DIAGNOSES_ICD_CSV_GZ.exists() else "no",
            "primary_label_source": "MIMIC-IV hosp ICD-linked labels",
            "track_status": track_f_status(),
            "sae_transfer_status": sae_status.get("MIMIC-F", "not_run_second_wave"),
        },
        {
            "cohort_track": "Chapman-F",
            "waveform_available": "yes",
            "vendor_measurements_available": "not_required",
            "vendor_measurement_scope": "not_applicable",
            "task_labels_available": challenge_task_status(),
            "primary_label_source": "native challenge labels",
            "track_status": track_f_status(),
            "sae_transfer_status": sae_status.get("Chapman-F", "not_run_second_wave"),
        },
        {
            "cohort_track": "CPSC-F",
            "waveform_available": "yes",
            "vendor_measurements_available": "not_required",
            "vendor_measurement_scope": "not_applicable",
            "task_labels_available": challenge_task_status(),
            "primary_label_source": "native challenge labels",
            "track_status": track_f_status(),
            "sae_transfer_status": sae_status.get("CPSC-F", "not_run_second_wave"),
        },
        {
            "cohort_track": "Ningbo-F",
            "waveform_available": "yes",
            "vendor_measurements_available": "not_required",
            "vendor_measurement_scope": "not_applicable",
            "task_labels_available": challenge_task_status(),
            "primary_label_source": "native challenge labels",
            "track_status": track_f_status(),
            "sae_transfer_status": sae_status.get("Ningbo-F", "not_run_second_wave"),
        },
    ]


def summarize_sae_transfer_status() -> dict[str, str]:
    path = DEFAULT_OUT_DIR / "external_sae" / "external_sae_recon_gate.csv"
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    by_cohort: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_cohort.setdefault(row.get("external_cohort", ""), []).append(row)

    statuses: dict[str, str] = {}
    for cohort, cohort_rows in by_cohort.items():
        if not cohort:
            continue
        if any(row.get("recon_gate_pass") == "true" for row in cohort_rows):
            statuses[cohort] = "recon_gate_passed"
            continue
        if any(row.get("recon_gate_status") == "fail_external_recon_below_floor" for row in cohort_rows):
            statuses[cohort] = "csfm_recon_gate_failed_below_floor"
            continue
        if all(row.get("external_activation_status") == "missing_external_activation_cache" for row in cohort_rows):
            statuses[cohort] = "missing_external_activation_cache"
            continue
        statuses[cohort] = "recon_gate_no_pass"
    return statuses


def build_external_task_feasibility_template() -> list[dict[str, str]]:
    rows = []
    task_specs = [
        (
            "MIMIC-V",
            "af_rhythm_icd",
            "ICD-linked AF/AFlutter labels",
            "primary_independent",
            "rate/rhythm covered by RR/HR proxy",
            "pending_icd_admission_time_linkage",
        ),
        (
            "MIMIC-V",
            "bbb_conduction_icd",
            "ICD-linked conduction/BBB labels",
            "primary_independent",
            "conduction covered by PR/PQ/QRS duration",
            "pending_icd_admission_time_linkage",
        ),
        (
            "MIMIC-V",
            "qt_interval_icd",
            "ICD-linked long-QT/QT-prolongation labels",
            "secondary_semi_independent",
            "sensitivity only: QT/QTc-like interval is measurement-proximal to this label",
            "sensitivity_label_measurement_proximal",
        ),
        (
            "MIMIC-V",
            "mi_ischemia_icd",
            "ICD-linked MI/ischemia labels",
            "primary_independent",
            "out_of_scope_missing_ST_T_morphology",
            "task_out_of_scope_due_to_missing_measurement_family",
        ),
        (
            "MIMIC-V",
            "hypertrophy_icd",
            "ICD-linked hypertrophy labels",
            "primary_independent",
            "out_of_scope_missing_amplitude_morphology",
            "task_out_of_scope_due_to_missing_measurement_family",
        ),
        (
            "MIMIC-V",
            "report_note_sensitivity",
            "MIMIC-IV-ECG report/note-derived labels",
            "secondary_semi_independent",
            "label-only source; never measurement source",
            "secondary_sensitivity_only",
        ),
    ]
    for cohort, task, source, tier, covered, reason in task_specs:
        rows.append(
            {
                "cohort": cohort,
                "task": task,
                "label_source": source,
                "label_independence_tier": tier,
                "positive_count": "",
                "negative_count": "",
                "split_unit": "patient_preferred_pending_linkage",
                "eligible_for_auroc": "false",
                "task_measurement_family_covered": covered,
                "exclusion_reason": reason,
            }
        )
    return rows


ICD_TASK_RULES = {
    "af_rhythm_icd": {
        "label_source": "ICD-linked AF/AFlutter labels",
        "label_independence_tier": "primary_independent",
        "task_measurement_family_covered": "rate/rhythm covered by RR/HR proxy",
        "primary_track_v": True,
        "prefixes_9": ("42731", "42732"),
        "prefixes_10": ("I48",),
    },
    "bbb_conduction_icd": {
        "label_source": "ICD-linked conduction/BBB labels",
        "label_independence_tier": "primary_independent",
        "task_measurement_family_covered": "conduction covered by PR/PQ/QRS duration",
        "primary_track_v": True,
        "prefixes_9": ("426",),
        "prefixes_10": ("I44", "I45"),
    },
    "qt_interval_icd": {
        "label_source": "ICD-linked long-QT/QT-prolongation labels",
        "label_independence_tier": "secondary_semi_independent",
        "task_measurement_family_covered": "sensitivity only: QT/QTc-like interval is measurement-proximal to this label",
        "primary_track_v": False,
        "sensitivity_track_v": True,
        "prefixes_9": ("42682",),
        "prefixes_10": ("I4581",),
    },
    "mi_ischemia_icd": {
        "label_source": "ICD-linked MI/ischemia labels",
        "label_independence_tier": "primary_independent",
        "task_measurement_family_covered": "out_of_scope_missing_ST_T_morphology",
        "primary_track_v": False,
        "prefixes_9": ("410", "411", "412", "413", "414"),
        "prefixes_10": ("I20", "I21", "I22", "I23", "I24", "I25"),
    },
    "hypertrophy_icd": {
        "label_source": "ICD-linked hypertrophy labels",
        "label_independence_tier": "primary_independent",
        "task_measurement_family_covered": "out_of_scope_missing_amplitude_morphology",
        "primary_track_v": False,
        "prefixes_9": ("4293", "425"),
        "prefixes_10": ("I51", "I42"),
    },
}


def parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def read_admissions_by_subject(
    path: Path = MIMIC_ADMISSIONS_CSV_GZ,
) -> dict[str, list[tuple[datetime, datetime, str]]]:
    by_subject: dict[str, list[tuple[datetime, datetime, str]]] = {}
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            subject_id = row.get("subject_id", "")
            hadm_id = row.get("hadm_id", "")
            admit = parse_time(row.get("admittime", ""))
            discharge = parse_time(row.get("dischtime", ""))
            if not subject_id or not hadm_id or admit is None or discharge is None:
                continue
            by_subject.setdefault(subject_id, []).append((admit, discharge, hadm_id))
    for admissions in by_subject.values():
        admissions.sort(key=lambda x: x[0])
    return by_subject


def read_task_positive_hadm_ids(
    path: Path = MIMIC_DIAGNOSES_ICD_CSV_GZ,
) -> dict[str, set[str]]:
    positives = {task: set() for task in ICD_TASK_RULES}
    with gzip.open(path, "rt", newline="") as f:
        for row in csv.DictReader(f):
            hadm_id = row.get("hadm_id", "")
            code = (row.get("icd_code", "") or "").replace(".", "").upper()
            version = row.get("icd_version", "")
            if not hadm_id or not code:
                continue
            version_key = "prefixes_10" if version == "10" else "prefixes_9"
            for task, rule in ICD_TASK_RULES.items():
                if code.startswith(tuple(rule[version_key])):
                    positives[task].add(hadm_id)
    return positives


def find_hadm_for_ecg(
    subject_id: str,
    ecg_time: datetime,
    admissions_by_subject: dict[str, list[tuple[datetime, datetime, str]]],
) -> str | None:
    for admit, discharge, hadm_id in admissions_by_subject.get(subject_id, []):
        if admit <= ecg_time <= discharge:
            return hadm_id
    return None


def build_external_task_feasibility_from_icd(
    min_positive: int = 50,
    min_negative: int = 50,
    max_records: int | None = None,
) -> list[dict[str, str]]:
    admissions_by_subject = read_admissions_by_subject()
    task_positive_hadm_ids = read_task_positive_hadm_ids()
    linked_hadm_ids: set[str] = set()
    linked_subject_ids: set[str] = set()
    records_seen = 0
    records_linked = 0

    with MIMIC_RECORD_LIST_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            if max_records is not None and records_seen >= max_records:
                break
            records_seen += 1
            subject_id = row.get("subject_id", "")
            ecg_time = parse_time(row.get("ecg_time", ""))
            if not subject_id or ecg_time is None:
                continue
            hadm_id = find_hadm_for_ecg(subject_id, ecg_time, admissions_by_subject)
            if hadm_id is None:
                continue
            records_linked += 1
            linked_hadm_ids.add(hadm_id)
            linked_subject_ids.add(subject_id)

    rows = []
    for task, rule in ICD_TASK_RULES.items():
        positive_hadm = linked_hadm_ids & task_positive_hadm_ids[task]
        positive_count = len(positive_hadm)
        negative_count = len(linked_hadm_ids) - positive_count
        in_scope = bool(rule["primary_track_v"])
        sensitivity = bool(rule.get("sensitivity_track_v", False))
        enough = positive_count >= min_positive and negative_count >= min_negative
        eligible = (in_scope or sensitivity) and enough
        if not in_scope:
            reason = (
                "sensitivity_label_measurement_proximal"
                if sensitivity
                else "task_out_of_scope_due_to_missing_measurement_family"
            )
        elif not enough:
            reason = "insufficient_positive_or_negative_count"
        else:
            reason = ""
        rows.append(
            {
                "cohort": "MIMIC-V",
                "task": task,
                "label_source": str(rule["label_source"]),
                "label_independence_tier": str(rule["label_independence_tier"]),
                "positive_count": str(positive_count),
                "negative_count": str(negative_count),
                "split_unit": "patient_level_available",
                "eligible_for_auroc": "true" if eligible else "false",
                "task_measurement_family_covered": str(rule["task_measurement_family_covered"]),
                "exclusion_reason": reason,
                "records_seen": str(records_seen),
                "records_linked_to_admission": str(records_linked),
                "linked_hadm_count": str(len(linked_hadm_ids)),
                "linked_subject_count": str(len(linked_subject_ids)),
                "min_positive": str(min_positive),
                "min_negative": str(min_negative),
            }
        )

    rows.append(
        {
            "cohort": "MIMIC-V",
            "task": "report_note_sensitivity",
            "label_source": "MIMIC-IV-ECG report/note-derived labels",
            "label_independence_tier": "secondary_semi_independent",
            "positive_count": "",
            "negative_count": "",
            "split_unit": "patient_level_available",
            "eligible_for_auroc": "false",
            "task_measurement_family_covered": "label-only source; never measurement source",
            "exclusion_reason": "secondary_sensitivity_only",
            "records_seen": str(records_seen),
            "records_linked_to_admission": str(records_linked),
            "linked_hadm_count": str(len(linked_hadm_ids)),
            "linked_subject_count": str(len(linked_subject_ids)),
            "min_positive": str(min_positive),
            "min_negative": str(min_negative),
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def task_feasibility_report_lines(task_feasibility: list[dict[str, str]], source: str) -> list[str]:
    eligible = [row for row in task_feasibility if row.get("eligible_for_auroc") == "true"]
    with_counts = [
        row
        for row in task_feasibility
        if row.get("positive_count") not in {"", None}
        and row.get("negative_count") not in {"", None}
    ]
    lines = [
        f"- Source: {source}",
        f"- Rows: {len(task_feasibility)}",
    ]
    if with_counts:
        records_seen = next((row.get("records_seen", "") for row in with_counts if row.get("records_seen")), "")
        records_linked = next(
            (row.get("records_linked_to_admission", "") for row in with_counts if row.get("records_linked_to_admission")),
            "",
        )
        if records_seen:
            lines.append(f"- ECG records seen: {records_seen}")
        if records_linked:
            lines.append(f"- ECG records linked to admission: {records_linked}")
        lines.append(f"- AUROC-eligible rows: {len(eligible)}")
        if eligible:
            lines.append("- Eligible tasks: " + ", ".join(f"{row.get('cohort', '')}/{row['task']}" for row in eligible))
        out_of_scope = [
            row["task"]
            for row in task_feasibility
            if row.get("exclusion_reason") == "task_out_of_scope_due_to_missing_measurement_family"
        ]
        if out_of_scope:
            lines.append("- Out-of-scope due missing measurement family: " + ", ".join(out_of_scope))
    else:
        lines.extend(
            [
                "- AUROC eligibility is intentionally false until ICD admission-time linkage and positive/negative counts are computed.",
                "- MIMIC-V primary tasks should stay within interval/rate/axis-covered families.",
            ]
        )
    return lines


def render_report(
    mimic_audit: list[dict[str, str]],
    crosswalk: list[dict[str, str]],
    gate_summary: list[dict[str, str]],
    task_feasibility: list[dict[str, str]],
    task_feasibility_source: str = "template",
) -> str:
    category_counts: dict[str, int] = {}
    for row in crosswalk:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    track_v = [row for row in crosswalk if row["category"] == "track_v_eligible"]
    lines = [
        "# Multi-Cohort Gate Report",
        "",
        "## MIMIC-IV-ECG Local Inputs",
        "",
        f"- ECG root: `{MIMIC_ECG_ROOT}`",
        f"- `machine_measurements.csv`: {'present' if MIMIC_MEASUREMENT_CSV.exists() else 'missing'}",
        f"- `record_list.csv`: {'present' if MIMIC_RECORD_LIST_CSV.exists() else 'missing'}",
        f"- `waveform_note_links.csv`: {'present' if MIMIC_WAVEFORM_NOTE_LINKS_CSV.exists() else 'missing'}",
        f"- waveform `files/`: {'present' if MIMIC_WAVEFORM_FILES_DIR.exists() else 'missing'}",
        f"- MIMIC-IV hosp ICD diagnoses: {'present' if MIMIC_DIAGNOSES_ICD_CSV_GZ.exists() else 'missing'}",
        "",
        "## G1: MIMIC Vendor Measurement Audit",
        "",
        f"- Columns audited: {len(mimic_audit)}",
        f"- Track-V eligible vendor columns: {sum(1 for row in mimic_audit if row['track_v_eligible'] == 'yes')}",
        "- Scope: interval/rate/axis only; report text is label-only and never a measurement source.",
        "",
        "## G2: Track V Concept Crosswalk",
        "",
        f"- Frozen main concepts audited: {len(crosswalk)}",
        f"- Track V eligible concepts: {category_counts.get('track_v_eligible', 0)}",
        f"- Track F only concepts: {category_counts.get('track_f_only', 0)}",
        f"- Not external v1 concepts: {category_counts.get('not_external_v1', 0)}",
        "",
        "Track V eligible concept IDs:",
        "",
        ", ".join(row["ptbxl_concept"] for row in track_v),
        "",
        "## G4: External Task Feasibility",
        "",
        *task_feasibility_report_lines(task_feasibility, task_feasibility_source),
        "",
        "## G3: Track F Waveform Extraction Status",
        "",
        *track_f_status_report_lines(),
        "",
        "## External SAE Reconstruction Gate",
        "",
        *sae_gate_report_lines(),
        "",
        "## Cohort Track Status",
        "",
    ]
    for row in gate_summary:
        lines.append(f"- {row['cohort_track']}: {row['track_status']} / SAE: {row['sae_transfer_status']}")
    lines.append("")
    return "\n".join(lines)


def track_f_status_report_lines() -> list[str]:
    full_concepts = DEFAULT_OUT_DIR / "track_f_full" / "waveform_concepts_by_record.csv"
    full_quality = DEFAULT_OUT_DIR / "track_f_full" / "waveform_concept_quality.csv"
    closure = DEFAULT_OUT_DIR / "track_f_closure" / "closure_transfer_track_f.csv"
    smoke = DEFAULT_OUT_DIR / "waveform_smoke_1k" / "waveform_family_gate.csv"
    lines = [
        f"- 1000-record/cohort smoke gate: {'present' if smoke.exists() else 'missing'}",
        f"- full waveform concept table: {'present' if full_concepts.exists() else 'missing'}",
        f"- full waveform concept quality table: {'present' if full_quality.exists() else 'missing'}",
        f"- Track F closure transfer table: {'present' if closure.exists() else 'missing'}",
    ]
    if full_concepts.exists():
        with full_concepts.open() as f:
            row_count = max(sum(1 for _ in f) - 1, 0)
        lines.append(f"- full waveform concept rows: {row_count}")
    if full_quality.exists():
        rows = read_csv_rows(full_quality)
        usable = [
            row
            for row in rows
            if row.get("gate_status") in {"full_candidate", "partial_full_candidate"}
            or row.get("final_gate_status") in {"full_candidate", "partial_full_candidate"}
        ]
        if usable:
            concepts = sorted({row.get("concept", "") for row in usable if row.get("concept")})
            lines.append("- gate-passed or sensitivity concepts: " + ", ".join(concepts))
    if closure.exists():
        rows = read_csv_rows(closure)
        primary = [row for row in rows if row.get("status") == "ok" and row.get("task_scope", "").startswith("primary")]
        lines.append(f"- primary Track F closure rows: {len(primary)}")
    return lines


def sae_gate_report_lines() -> list[str]:
    recon = DEFAULT_OUT_DIR / "external_sae" / "external_sae_recon_gate.csv"
    steering = DEFAULT_OUT_DIR / "external_sae" / "external_sae_steering_audit.csv"
    if not recon.exists():
        return [
            "- external SAE recon gate table: missing",
            "- steering transfer is not evaluated without the reconstruction gate.",
        ]
    rows = read_csv_rows(recon)
    status_counts: dict[str, int] = {}
    activation_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.get("recon_gate_status", "")] = status_counts.get(row.get("recon_gate_status", ""), 0) + 1
        activation_counts[row.get("external_activation_status", "")] = (
            activation_counts.get(row.get("external_activation_status", ""), 0) + 1
        )
    passes = sum(row.get("recon_gate_pass") == "true" for row in rows)
    lines = [
        f"- external SAE recon gate rows: {len(rows)}",
        f"- recon-gate passes: {passes}",
        "- recon statuses: " + "; ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()) if k),
        "- activation statuses: " + "; ".join(f"{k}: {v}" for k, v in sorted(activation_counts.items()) if k),
    ]
    if steering.exists():
        steering_rows = read_csv_rows(steering)
        skipped = sum(row.get("status") == "skipped_no_recon_gate_pass" for row in steering_rows)
        allowed = sum(row.get("steering_claim_allowed") == "true" for row in steering_rows)
        lines.append(f"- steering audit rows: {len(steering_rows)}; skipped_no_recon_gate_pass: {skipped}; claim-allowed rows: {allowed}")
    if passes == 0:
        lines.append("- claim discipline: external SAE steering claims remain disallowed because no row passed reconstruction.")
    return lines


def build_multicohort_gates(
    out_dir: Path = DEFAULT_OUT_DIR,
    overwrite_task_feasibility_template: bool = False,
) -> MulticohortPaths:
    paths = multicohort_paths(out_dir)
    mimic_audit = build_mimic_measurement_audit()
    crosswalk = build_concept_crosswalk_track_v()
    gate_summary = build_cohort_gate_summary()
    task_feasibility_source = "new template"
    if paths.external_task_feasibility.exists() and not overwrite_task_feasibility_template:
        task_feasibility = read_csv_rows(paths.external_task_feasibility)
        task_feasibility_source = f"existing file preserved at {paths.external_task_feasibility}"
    else:
        task_feasibility = build_external_task_feasibility_template()

    write_csv(paths.mimic_measurement_audit, mimic_audit)
    write_csv(paths.concept_crosswalk_track_v, crosswalk)
    write_csv(paths.cohort_gate_summary, gate_summary)
    if overwrite_task_feasibility_template or not paths.external_task_feasibility.exists():
        write_csv(paths.external_task_feasibility, task_feasibility)
    paths.report.write_text(
        render_report(
            mimic_audit,
            crosswalk,
            gate_summary,
            task_feasibility,
            task_feasibility_source,
        )
    )
    return paths
