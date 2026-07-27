import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmark_v1.mimic_matched_effect import (
    CONCEPT_SPECS,
    PROTOCOL,
    PROTOCOL_100K,
    aligned_concepts,
    fit_masked_ridge_readout,
    load_indexed_layer,
    masked_feature_concept_correlations,
    normalize_record_id,
    split_for_patient,
)
from scripts.run_mimic_final_layer_matched_effect_worker import (
    feature_firing_counts,
    matched_live_capacity_quality,
)


class MimicMatchedEffectTest(unittest.TestCase):
    def test_100k_protocol_is_isolated_from_4k_protocol(self) -> None:
        self.assertNotEqual(PROTOCOL, PROTOCOL_100K)
        self.assertEqual(PROTOCOL_100K, "mimic_final_layer_matched_effect_100k_v1")

    def test_live_capacity_gate_uses_reconstruction_and_absolute_capacity(self) -> None:
        metrics = {
            "N": 6144,
            "validation": {"reconstruction_r2": 0.95, "dead_fraction": 0.70},
        }
        gate = matched_live_capacity_quality(metrics, 0.90, 768)
        self.assertTrue(gate["pass"])
        self.assertEqual(gate["validation_live_features"], 1843)
        self.assertFalse(matched_live_capacity_quality(metrics, 0.96, 768)["pass"])
        self.assertFalse(matched_live_capacity_quality(metrics, 0.90, 2000)["pass"])

    def test_feature_firing_counts_identifies_train_live_atoms(self) -> None:
        features = np.asarray([[0, 2, 0, 1], [0, 0, 0, 3], [0, 4, 0, 0]])
        np.testing.assert_array_equal(feature_firing_counts(features), [0, 2, 0, 2])

    def test_patient_split_is_deterministic_and_patient_level(self) -> None:
        values = [split_for_patient("100"), split_for_patient("101"), split_for_patient("100")]
        self.assertEqual(values[0], values[2])
        self.assertTrue(set(values).issubset({"train", "val", "test"}))
        self.assertEqual(normalize_record_id("mimic_f:123"), "123")

    def test_concept_panel_excludes_derived_duplicates_and_retains_missing(self) -> None:
        names = [name for name, _family in CONCEPT_SPECS]
        self.assertIn("heart_rate_bpm", names)
        self.assertNotIn("rr_mean_ms", names)
        self.assertNotIn("qtc_bazett_ms", names)
        rows = []
        for index in range(12):
            rows.append(
                {
                    "record_id": str(index),
                    "status": "ok",
                    "rr_mean_ms": str(1000 + index),
                    "qrs_duration_ms": "100",
                    "pr_interval_ms": "" if index == 11 else "160",
                    "qt_like_ms": "400",
                    "r_amp_global_mv": "1.0",
                    "st_amp_global_mv": "0.1",
                    "t_amp_global_mv": "0.3",
                }
            )
        concepts, aligned_names, means, scales, counts = aligned_concepts(
            [str(index) for index in range(12)], rows, np.arange(12) < 8
        )
        self.assertEqual(aligned_names, names)
        self.assertTrue(np.isnan(concepts[11, names.index("pr_interval_ms")]))
        self.assertEqual(counts[names.index("pr_interval_ms")], 11)
        self.assertTrue(np.isfinite(means).all())
        self.assertTrue(np.isfinite(scales).all())

    def test_masked_correlations_do_not_impute_missing_targets(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(200, 5))
        concepts = np.column_stack((features[:, 2], features[:, 4]))
        concepts[::2, 0] = np.nan
        correlations = masked_feature_concept_correlations(features, concepts)
        self.assertEqual(correlations.shape, (5, 2))
        self.assertGreater(abs(correlations[2, 0]), 0.99)
        self.assertGreater(abs(correlations[4, 1]), 0.99)

    def test_masked_ridge_uses_finite_labels_per_split(self) -> None:
        rng = np.random.default_rng(9)
        x_train = rng.normal(size=(120, 6))
        x_val = rng.normal(size=(60, 6))
        x_test = rng.normal(size=(70, 6))

        def targets(values):
            return np.column_stack((values[:, 1] - 0.5 * values[:, 3], values[:, 5]))

        y_train = targets(x_train)
        y_val = targets(x_val)
        y_test = targets(x_test)
        y_train[::4, 0] = np.nan
        y_val[::3, 0] = np.nan
        y_test[::5, 0] = np.nan
        fitted, counts = fit_masked_ridge_readout(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            alphas=(0.1, 1.0, 10.0),
            min_train=20,
            min_validation=10,
            min_test=10,
        )
        self.assertEqual(fitted.coefficients.shape, (6, 2))
        self.assertTrue(np.all(np.abs(fitted.test_correlations) > 0.99))
        self.assertEqual(counts[0, 0], 90)

    def test_generic_indexed_layer_loader_preserves_offset_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_root = root / "index"
            index_root.mkdir()
            shard_rows = []
            expected = []
            for offset, count in ((0, 2), (2, 1)):
                shard = root / f"shard_{offset}"
                shard.mkdir()
                records = shard / "record_ids.csv"
                with records.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["ecg_id", "subject_id"])
                    writer.writeheader()
                    for row_index in range(count):
                        writer.writerow({"ecg_id": f"mimic_f:{offset + row_index}", "subject_id": "1"})
                values = np.arange(offset * 3, (offset + count) * 3, dtype=np.float32).reshape(count, 3)
                np.save(shard / "layer_05.npy", values)
                np.save(shard / "pooled.npy", values)
                expected.append(values)
                shard_rows.append(
                    {
                        "offset": str(offset),
                        "record_ids_file": str(records),
                        "pooled_file": str(shard / "pooled.npy"),
                        "layer_files": "layer_05.npy",
                    }
                )
            with (index_root / "shards.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(shard_rows[0]))
                writer.writeheader()
                writer.writerows(reversed(shard_rows))
            records, values = load_indexed_layer(root, index_root, 5)
            self.assertEqual([normalize_record_id(row["ecg_id"]) for row in records], ["0", "1", "2"])
            np.testing.assert_array_equal(values, np.concatenate(expected))


if __name__ == "__main__":
    unittest.main()
