from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_v1.input_harmonization import (
    FULL_LEADS,
    INDEPENDENT_LEADS,
    MODEL_INTERFACES,
    _signal_calibration,
    build_model_input,
    final_layer_for_model,
    reconstruct_12_leads,
    select_independent_leads,
    zscore_per_lead,
)


def synthetic_wave(samples: int = 5000) -> np.ndarray:
    time = np.arange(samples, dtype=np.float32) / 500.0
    rows = []
    for index in range(12):
        rows.append(
            (index + 1) * 0.1
            + np.sin(2 * np.pi * (1.0 + index / 10.0) * time)
            + 0.05 * np.sin(2 * np.pi * 180.0 * time + index)
        )
    return np.asarray(rows, dtype=np.float32)


class InputHarmonizationTest(unittest.TestCase):
    def test_reconstruct_dependent_limb_leads(self) -> None:
        wave8 = select_independent_leads(synthetic_wave(64))
        wave12 = reconstruct_12_leads(wave8)
        by_name = {lead: wave12[index] for index, lead in enumerate(FULL_LEADS)}
        np.testing.assert_allclose(by_name["III"], by_name["II"] - by_name["I"])
        np.testing.assert_allclose(by_name["aVR"], -(by_name["I"] + by_name["II"]) / 2)
        np.testing.assert_allclose(by_name["aVL"], by_name["I"] - by_name["II"] / 2)
        np.testing.assert_allclose(by_name["aVF"], by_name["II"] - by_name["I"] / 2)
        self.assertEqual(wave12.shape, (12, 64))

    def test_all_model_protocol_shapes_are_checkpoint_compatible(self) -> None:
        wave = synthetic_wave()
        for model in sorted(MODEL_INTERFACES):
            for protocol in ("native", "lead", "temporal", "joint"):
                with self.subTest(model=model, protocol=protocol):
                    result = build_model_input(wave, model, protocol)
                    spec = MODEL_INTERFACES[model]
                    expected = (int(spec["leads"]) * int(spec["samples"]),) if spec["flatten"] else (
                        int(spec["leads"]),
                        int(spec["samples"]),
                    )
                    self.assertEqual(result.shape, expected)
                    self.assertEqual(result.dtype, np.float32)
                    self.assertTrue(np.isfinite(result).all())

    def test_expected_negative_control_inputs_are_identical(self) -> None:
        wave = synthetic_wave()
        for protocol in ("lead", "temporal", "joint"):
            np.testing.assert_array_equal(
                build_model_input(wave, "ecg_jepa", "native"),
                build_model_input(wave, "ecg_jepa", protocol),
            )
        for model in ("csfm", "hubert_ecg"):
            np.testing.assert_array_equal(
                build_model_input(wave, model, "native"),
                build_model_input(wave, model, "temporal"),
            )
            np.testing.assert_array_equal(
                build_model_input(wave, model, "lead"),
                build_model_input(wave, model, "joint"),
            )

    def test_temporal_harmonization_changes_long_or_cropped_interfaces(self) -> None:
        wave = synthetic_wave()
        for model in ("cardiac_fm", "ecg_fm", "st_mem"):
            native = build_model_input(wave, model, "native")
            temporal = build_model_input(wave, model, "temporal")
            self.assertEqual(native.shape, temporal.shape)
            self.assertFalse(np.array_equal(native, temporal))

    def test_zscore_handles_constant_leads(self) -> None:
        wave = np.ones((len(INDEPENDENT_LEADS), 32), dtype=np.float32)
        result = zscore_per_lead(wave)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_signal_calibration_parser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            header = Path(directory) / "sample.hea"
            header.write_text(
                "sample 2 500 5000\n"
                "sample.mat 16x1+24 1000.0(0)/mV 16 0 0 0 0 I\n"
                "sample.mat 16x1+24 2000(-10)/mV 16 0 0 0 0 II\n",
                encoding="utf-8",
            )
            self.assertEqual(_signal_calibration(header), {"I": (1000.0, 0.0), "II": (2000.0, -10.0)})

    def test_final_layers_match_encoder_depths(self) -> None:
        self.assertEqual(final_layer_for_model("csfm"), 5)
        for model in MODEL_INTERFACES:
            if model != "csfm":
                self.assertEqual(final_layer_for_model(model), 11)


if __name__ == "__main__":
    unittest.main()
