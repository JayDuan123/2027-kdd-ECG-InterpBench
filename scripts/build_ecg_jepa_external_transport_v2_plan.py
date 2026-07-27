#!/usr/bin/env python
"""Regenerate ECG-JEPA external smoke activations after finite-input hardening."""
from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "results/activations_external_pooled_smoke/plan/sample_manifest.csv"
OUT = ROOT / "results/activations_external_pooled_smoke_v2"
PLAN = OUT / "plan"
COHORTS = ("chapman", "cpsc", "ningbo", "mimic")


def main() -> None:
    from scripts.plan_external_activation_extraction import command, index_command

    PLAN.mkdir(parents=True, exist_ok=True)
    commands = []
    indexes = []
    for cohort in COHORTS:
        for offset in range(0, 512, 32):
            commands.append(
                command(
                    model="ecg_jepa", cohort=cohort, offset=offset, limit=32, layers="0",
                    activation_out_dir=OUT, manifest=SOURCE, device="cuda"
                )
            )
        indexes.append(index_command("ecg_jepa", cohort, OUT))
    (PLAN / "all_commands.txt").write_text("\n".join(commands) + "\n")
    (PLAN / "index_commands.txt").write_text("\n".join(indexes) + "\n")
    print(f"commands={len(commands)} indexes={len(indexes)} out={OUT}")


if __name__ == "__main__":
    main()
