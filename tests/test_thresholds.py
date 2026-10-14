import unittest

import numpy as np

from ran_monitor_ai.engine import bundle_for_policy
from ran_monitor_ai.train import calibrate_threshold, threshold_profiles


class ValidationThresholdTests(unittest.TestCase):
    def test_recall_profiles_meet_targets_and_sensitive_threshold_is_lower(self):
        labels = np.array([0] * 20 + [1] * 20)
        scores = np.concatenate((np.linspace(.01, .4, 20), np.linspace(.3, .99, 20)))
        profiles = threshold_profiles(labels, scores)
        self.assertGreaterEqual(profiles["balanced"]["validation_recall"], .90)
        self.assertGreaterEqual(profiles["sensitive"]["validation_recall"], .95)
        self.assertLessEqual(profiles["sensitive"]["threshold"], profiles["balanced"]["threshold"])
        self.assertGreaterEqual(profiles["precision"]["validation_precision"], .98)
        base = {"threshold": profiles["balanced"]["threshold"], "threshold_profiles": profiles}
        chosen = bundle_for_policy(base, "sensitive")
        self.assertEqual(chosen["threshold"], profiles["sensitive"]["threshold"])
        self.assertEqual(base["threshold"], profiles["balanced"]["threshold"])

    def test_unreachable_precision_target_has_explicit_fallback(self):
        profiles = threshold_profiles(np.array([0, 1, 0, 1]), np.array([.9, .8, .7, .6]))
        self.assertFalse(profiles["precision"]["target_met"])
        self.assertEqual(profiles["precision"]["threshold"], profiles["balanced"]["threshold"])

    def test_policy_and_calibration_errors_are_actionable(self):
        with self.assertRaisesRegex(ValueError, "策略无效"):
            bundle_for_policy({"threshold": .5}, "unknown")
        with self.assertRaisesRegex(ValueError, "重新训练"):
            bundle_for_policy({"threshold": .5}, "sensitive")
        with self.assertRaises(ValueError):
            threshold_profiles(np.ones(3), np.array([.1, .2, .3]))

    def test_calibrator_uses_requested_validation_objective(self):
        labels = np.array([0, 0, 0, 1, 1])
        scores = np.array([.1, .6, .7, .65, .9])
        low_fpr = calibrate_threshold(labels, scores, "fpr", 0.0)
        self.assertTrue(low_fpr["target_met"])
        self.assertEqual(low_fpr["false_positive_rate"], 0.0)
        high_recall = calibrate_threshold(labels, scores, "recall", .9)
        self.assertTrue(high_recall["target_met"])
        self.assertGreaterEqual(high_recall["recall"], .9)
        self.assertEqual(high_recall["threshold"], .65)

    def test_fpr_target_fallback_is_explicit_when_unreachable(self):
        result = calibrate_threshold(np.array([0, 1]), np.array([1.0, 1.0]), "fpr", 0.0)
        self.assertFalse(result["target_met"])
        self.assertEqual(result["false_positive_rate"], 1.0)

    def test_calibrator_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calibrate_threshold([0, 1], [.2, float("nan")], "fpr", .05)
        with self.assertRaises(ValueError):
            calibrate_threshold([0, 1], [.2, .8], "recall", 1.5)


if __name__ == "__main__":
    unittest.main()
