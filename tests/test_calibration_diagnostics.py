import unittest

import numpy as np

from ran_monitor_ai.evaluation import calibration_diagnostics


class ProbabilityCalibrationTests(unittest.TestCase):
    def test_binary_calibration_uses_brier_and_top_label_ece(self):
        result = calibration_diagnostics([0, 1], [0.6, 0.6], bins=5)
        self.assertAlmostEqual(result["brier_score"], 0.26)
        self.assertAlmostEqual(result["top_label_ece"], 0.1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["samples"], 2)

    def test_multiclass_probabilities_report_reliability_bins(self):
        result = calibration_diagnostics(
            [0, 2, 1],
            [[0.9, 0.05, 0.05], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2]],
            bins=4,
        )
        self.assertEqual(result["classes"], 3)
        self.assertEqual(result["accuracy"], 2 / 3)
        self.assertGreaterEqual(result["brier_score"], 0)
        self.assertGreaterEqual(result["top_label_ece"], 0)
        self.assertEqual(sum(row["count"] for row in result["bins"]), 3)

    def test_rejects_invalid_labels_probabilities_and_bin_counts(self):
        with self.assertRaisesRegex(ValueError, "0–1"):
            calibration_diagnostics([0, 1], [0.2, 1.2])
        with self.assertRaisesRegex(ValueError, "总和为 1"):
            calibration_diagnostics([0, 1], [[0.2, 0.3], [0.8, 0.1]])
        with self.assertRaisesRegex(ValueError, "分箱数"):
            calibration_diagnostics([0, 1], [0.2, 0.8], bins=1)


if __name__ == "__main__":
    unittest.main()
