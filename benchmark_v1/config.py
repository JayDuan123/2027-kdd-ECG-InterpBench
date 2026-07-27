from __future__ import annotations

from dataclasses import dataclass
import csv
import os
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(
    os.environ.get("ECG_INTERPBENCH_WORKSPACE", ROOT.parent)
).expanduser().resolve()
DATA_ROOT = Path(
    os.environ.get("ECG_INTERPBENCH_DATA_ROOT", WORKSPACE / "data")
).expanduser().resolve()
PTBXL_WAVEFORM_ROOT = Path(
    os.environ.get("PTBXL_ROOT", DATA_ROOT / "ptb-xl")
).expanduser().resolve()
PTBXL_METADATA_ROOT = PTBXL_WAVEFORM_ROOT / "1.0.3"
PTBXL_PLUS_ROOT = Path(
    os.environ.get("PTBXL_PLUS_ROOT", DATA_ROOT / "1.0.1")
).expanduser().resolve()

CONCEPTS_CSV = ROOT / "configs" / "concepts.csv"
TASKS_CSV = ROOT / "configs" / "tasks.csv"
MODEL_GATE_CSV = ROOT / "configs" / "model_gate.csv"

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

FEATURE_FILES = {
    "12sl_features.csv": PTBXL_PLUS_ROOT / "features" / "12sl_features.csv",
    "ecgdeli_features.csv": PTBXL_PLUS_ROOT / "features" / "ecgdeli_features.csv",
    "unig_features.csv": PTBXL_PLUS_ROOT / "features" / "unig_features.csv",
}

LABEL_FILES = {
    "ptbxl_statements": PTBXL_PLUS_ROOT / "labels" / "ptbxl_statements.csv",
    "12sl_statements": PTBXL_PLUS_ROOT / "labels" / "12sl_statements.csv",
    "ptbxl_database": PTBXL_METADATA_ROOT / "ptbxl_database.csv",
    "scp_statements": PTBXL_METADATA_ROOT / "scp_statements.csv",
}

LOCAL_MODEL_ARTIFACTS = {
    "ECG-FM": [
        WORKSPACE / "ECG-FM" / "ckpts" / "mimic_iv_ecg_physionet_pretrained.pt",
        WORKSPACE / "ECG-FM" / "ckpts" / "mimic_iv_ecg_physionet_pretrained.yaml",
    ],
    "ECG-JEPA": [
        WORKSPACE / "ECG_JEPA" / "weights" / "multiblock_epoch100.pth",
        WORKSPACE / "ECG_JEPA" / "ecg_jepa.py",
    ],
    "CSFM ECG branch": [
        WORKSPACE / "Cardiac-Sensing-FM" / "pretrained" / "CSFM_tiny.pth",
        WORKSPACE / "Cardiac-Sensing-FM" / "network" / "model.py",
    ],
    "ST-MEM": [],
    "ECGFounder": [
        WORKSPACE / "ECGFounder" / "checkpoint" / "12_lead_ECGFounder.pth",
        WORKSPACE / "ECGFounder" / "net1d.py",
    ],
    "HuBERT-ECG": [
        WORKSPACE / "HuBERT-ECG" / "checkpoints" / "hubert-ecg-base" / "model.safetensors",
        WORKSPACE / "HuBERT-ECG" / "hubert_ecg" / "modeling.py",
    ],
    "CARDIAC-FM ECG branch": [
        WORKSPACE / "CARDIAC-FM" / "checkpoints" / "hf" / "model_epoch_8.pth",
        WORKSPACE / "CARDIAC-FM" / "cardiac_fm" / "model.py",
        WORKSPACE / "ECG-FM" / "ckpts" / "mimic_iv_ecg_physionet_pretrained.pt",
    ],
}


@dataclass(frozen=True)
class Concept:
    concept_id: str
    family: str
    display_name: str
    aggregation: str
    source_file: str
    source_columns: str
    main: str
    notes: str


@dataclass(frozen=True)
class Task:
    task_id: str
    task_family: str
    diagnostic_label: str
    source: str
    label_type: str
    main: str
    notes: str


@dataclass(frozen=True)
class ModelGate:
    model: str
    checkpoint: str
    architecture: str
    activation_access: str
    continuation: str
    input_protocol: str
    head_protocol: str
    modality: str
    lead_protocol: str
    time_length: str
    status: str
    notes: str


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_concepts(path: Path = CONCEPTS_CSV) -> list[Concept]:
    return [Concept(**row) for row in read_csv_dicts(path)]


def load_tasks(path: Path = TASKS_CSV) -> list[Task]:
    return [Task(**row) for row in read_csv_dicts(path)]


def load_model_gates(path: Path = MODEL_GATE_CSV) -> list[ModelGate]:
    return [ModelGate(**row) for row in read_csv_dicts(path)]


def csv_header(path: Path) -> set[str]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        return set(next(reader))


def expand_source_columns(spec: str) -> list[str]:
    cols: list[str] = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        if "{lead}" in part:
            cols.extend(part.replace("{lead}", lead) for lead in LEADS)
        else:
            cols.append(part)
    return cols


def existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [p for p in paths if p.exists()]
