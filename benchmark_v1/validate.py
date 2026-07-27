from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
from pathlib import Path

from .config import (
    FEATURE_FILES,
    LABEL_FILES,
    LOCAL_MODEL_ARTIFACTS,
    ROOT,
    csv_header,
    existing_paths,
    expand_source_columns,
    load_concepts,
    load_model_gates,
    load_tasks,
)


FORBIDDEN_DIAGNOSTIC_TOKENS = {
    "AF",
    "MI",
    "STTC",
    "CD",
    "HYP",
    "NORM",
}

MAIN_GATE_PASS = {
    "checkpoint": {"yes"},
    "architecture": {"transformer", "vit", "vit-style"},
    "activation_access": {"yes"},
    "continuation": {"yes"},
    "head_protocol": {"yes"},
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: list[str]


def tokenize_upper(text: str) -> set[str]:
    return {tok.upper() for tok in re.split(r"[^A-Za-z0-9]+", text) if tok}


def check_required_files() -> CheckResult:
    details: list[str] = []
    ok = True
    for label, path in {**FEATURE_FILES, **LABEL_FILES}.items():
        exists = path.exists()
        ok = ok and exists
        details.append(f"{label}: {'FOUND' if exists else 'MISSING'} {path}")
    return CheckResult("local PTB-XL+ files", ok, details)


def check_concepts() -> CheckResult:
    concepts = load_concepts()
    details: list[str] = []
    ok = True

    ids = [c.concept_id for c in concepts]
    duplicates = [k for k, v in Counter(ids).items() if v > 1]
    if duplicates:
        ok = False
        details.append(f"duplicate concept_id values: {duplicates}")

    main_concepts = [c for c in concepts if c.main.lower() == "yes"]
    details.append(f"main concept count: {len(main_concepts)}")
    if not 45 <= len(main_concepts) <= 60:
        ok = False
        details.append("main concept count should stay near the preregistered target of 50")

    families = Counter(c.family for c in main_concepts)
    details.append("main concepts by family: " + ", ".join(f"{k}={v}" for k, v in sorted(families.items())))

    for c in concepts:
        scanned = " ".join([c.concept_id, c.display_name, c.source_columns])
        leaked = tokenize_upper(scanned) & FORBIDDEN_DIAGNOSTIC_TOKENS
        if leaked:
            ok = False
            details.append(f"diagnostic token in concept {c.concept_id}: {sorted(leaked)}")

    headers: dict[str, set[str]] = {}
    for c in concepts:
        path = FEATURE_FILES.get(c.source_file)
        if path is None:
            ok = False
            details.append(f"{c.concept_id}: unknown source_file {c.source_file}")
            continue
        if not path.exists():
            ok = False
            details.append(f"{c.concept_id}: missing source file {path}")
            continue
        headers.setdefault(c.source_file, csv_header(path))
        missing = [col for col in expand_source_columns(c.source_columns) if col not in headers[c.source_file]]
        if missing:
            ok = False
            details.append(f"{c.concept_id}: missing source columns in {c.source_file}: {missing[:8]}")

    return CheckResult("concept registry", ok, details)


def check_tasks() -> CheckResult:
    tasks = load_tasks()
    concepts = load_concepts()
    ok = True
    details: list[str] = []

    task_ids = [t.task_id for t in tasks]
    duplicates = [k for k, v in Counter(task_ids).items() if v > 1]
    if duplicates:
        ok = False
        details.append(f"duplicate task_id values: {duplicates}")

    concept_text = defaultdict(set)
    for c in concepts:
        for token in tokenize_upper(" ".join([c.concept_id, c.display_name, c.source_columns])):
            concept_text[token].add(c.concept_id)

    for t in tasks:
        for token in tokenize_upper(t.diagnostic_label):
            if token in concept_text:
                ok = False
                details.append(f"task/concept leakage: task {t.task_id} token {token} appears in {sorted(concept_text[token])}")

    main_tasks = [t.task_id for t in tasks if t.main.lower() == "yes"]
    conditional = [t.task_id for t in tasks if t.main.lower() == "conditional"]
    details.append(f"main tasks: {', '.join(main_tasks)}")
    details.append(f"conditional tasks: {', '.join(conditional) if conditional else 'none'}")
    return CheckResult("task registry", ok, details)


def check_model_gate() -> CheckResult:
    gates = load_model_gates()
    ok = True
    details: list[str] = []

    for gate in gates:
        artifacts = LOCAL_MODEL_ARTIFACTS.get(gate.model, [])
        found = existing_paths(artifacts)
        details.append(f"{gate.model}: local artifacts {len(found)}/{len(artifacts)}; status={gate.status}")
        if gate.checkpoint == "yes" and artifacts and len(found) != len(artifacts):
            ok = False
            details.append(f"{gate.model}: checkpoint marked yes but expected local artifacts are missing")
        if "main" in gate.status and "candidate" in gate.status:
            failed = []
            for field, allowed in MAIN_GATE_PASS.items():
                value = getattr(gate, field)
                if value.lower() not in allowed:
                    failed.append(f"{field}={value}")
            if failed:
                ok = False
                details.append(f"{gate.model}: main candidate has failed gates: {', '.join(failed)}")

    return CheckResult("model gate", ok, details)


def run_checks() -> list[CheckResult]:
    return [
        check_required_files(),
        check_concepts(),
        check_tasks(),
        check_model_gate(),
    ]


def render_markdown(results: list[CheckResult]) -> str:
    passed = sum(1 for r in results if r.ok)
    lines = [
        "# ECG FM Benchmark v1 Validation Report",
        "",
        f"Checks passed: {passed}/{len(results)}",
        "",
    ]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"## {status}: {result.name}")
        lines.append("")
        for detail in result.details:
            lines.append(f"- {detail}")
        lines.append("")
    return "\n".join(lines)


def write_report(path: Path | None = None) -> Path:
    results = run_checks()
    out = path or (ROOT / "results" / "validation_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(results), encoding="utf-8")
    return out
