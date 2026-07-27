from __future__ import annotations

import os
import tempfile
from pathlib import Path
import unittest

from benchmark_v1.config import load_concepts, load_model_gates, load_tasks
from benchmark_v1.adapters.ecg_jepa import (
    ECG_JEPA_LEADS,
    build_waveform_index,
    parse_layer_spec,
    parse_header,
    record_name_for_ecg_id,
    reduced_lead_indices,
)
from benchmark_v1.build_manifest import build_manifest
from benchmark_v1.validate import run_checks, write_report


RUN_DATA_TESTS = os.environ.get("ECG_INTERPBENCH_RUN_DATA_TESTS") == "1"


class BenchmarkConfigTests(unittest.TestCase):
    def test_registries_load(self) -> None:
        self.assertGreaterEqual(len(load_concepts()), 45)
        self.assertGreaterEqual(len(load_tasks()), 5)
        self.assertGreaterEqual(len(load_model_gates()), 5)

    @unittest.skipUnless(RUN_DATA_TESTS, "requires local PTB-XL and PTB-XL+ data")
    def test_validation_passes(self) -> None:
        results = run_checks()
        failed = [r.name for r in results if not r.ok]
        self.assertEqual(failed, [])

    @unittest.skipUnless(RUN_DATA_TESTS, "requires local PTB-XL and PTB-XL+ data")
    def test_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_report(Path(tmp) / "report.md")
            text = path.read_text(encoding="utf-8")
            self.assertIn("Checks passed", text)
            self.assertIn("PASS: concept registry", text)

    @unittest.skipUnless(RUN_DATA_TESTS, "requires local PTB-XL and PTB-XL+ data")
    def test_manifest_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = build_manifest(Path(tmp) / "manifest")
            self.assertTrue(paths.concepts_matrix.exists())
            self.assertTrue(paths.tasks_matrix.exists())
            self.assertTrue(paths.split.exists())
            self.assertTrue(paths.concept_summary.exists())
            self.assertTrue(paths.provenance_report.exists())
            self.assertIn("Patient-level split: yes", paths.report.read_text(encoding="utf-8"))
            self.assertIn("exact id match: yes", paths.provenance_report.read_text(encoding="utf-8"))

    @unittest.skipUnless(RUN_DATA_TESTS, "requires local PTB-XL waveform files")
    def test_ecg_jepa_ptbxl_adapter(self) -> None:
        self.assertEqual(record_name_for_ecg_id("1"), "HR00001")
        index = build_waveform_index()
        self.assertIn("HR00001", index)
        header = parse_header(index["HR00001"]["hea"])
        self.assertEqual(header.sample_rate, 500)
        self.assertEqual(header.n_samples, 5000)
        self.assertEqual(header.leads[:2], ["I", "II"])
        self.assertEqual([header.leads[i] for i in reduced_lead_indices(header)], ECG_JEPA_LEADS)
        self.assertEqual(parse_layer_spec("0,5,11"), [0, 5, 11])
        self.assertEqual(parse_layer_spec("all")[:3], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
