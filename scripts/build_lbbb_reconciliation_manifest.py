#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "results/manifest/split.csv"
DEFAULT_LABELS = Path("/rhf/allocations/wq8/yd68/data/1.0.1/labels/ptbxl_statements.csv")
DEFAULT_OUT = ROOT / "results/sae_reconciliation/lbbb_fig6/manifest.csv"


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--labels-csv", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    a = args()
    split = {r["ecg_id"]: r for r in csv.DictReader(a.split_csv.open())}
    labels = {}
    for row in csv.DictReader(a.labels_csv.open()):
        codes = dict(ast.literal_eval(row["scp_codes"]))
        labels[row["ecg_id"]] = {
            "lbbb": int(float(codes.get("CLBBB", 0.0)) > 0),
            "lbbb_score": float(codes.get("CLBBB", 0.0)),
            "af": int(max(float(codes.get("AFIB", 0.0)), float(codes.get("AFLT", 0.0))) > 0),
        }
    rows = []
    for ecg_id, srow in sorted(split.items(), key=lambda x: int(x[0])):
        if ecg_id not in labels:
            continue
        rows.append({
            "row_index": len(rows),
            "ecg_id": ecg_id,
            "patient_id": srow["patient_id"],
            "split": srow["split"],
            **labels[ecg_id],
        })
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(a.out)
    counts = {}
    for split_name in ("train", "val", "test"):
        subset = [r for r in rows if r["split"] == split_name]
        counts[split_name] = {
            "records": len(subset),
            "lbbb_records": sum(r["lbbb"] for r in subset),
            "lbbb_patients": len({r["patient_id"] for r in subset if r["lbbb"]}),
            "af_records": sum(r["af"] for r in subset),
        }
    meta = {
        "lbbb_definition": "PTB-XL+ scp_codes contains CLBBB with score > 0",
        "af_control_definition": "AFIB or AFLT score > 0",
        "patient_split": str(a.split_csv),
        "counts": counts,
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
