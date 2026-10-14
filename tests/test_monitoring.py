import unittest

import numpy as np

from ran_monitor_ai.monitoring import assess_score_drift, update_score_drift


class ScoreDriftMonitoringTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.linspace(.05, .45, 100)

    def test_monitor_waits_until_minimum_sample_count(self):
        report = assess_score_drift(self.reference, [.2] * 29)
        self.assertEqual(report["status"], "warming_up")
        self.assertIsNone(report["p_value"])

    def test_same_score_distribution_is_stable(self):
        report = assess_score_drift(self.reference, self.reference)
        self.assertEqual(report["status"], "stable")
        self.assertEqual(report["ks_statistic"], 0.0)

    def test_large_score_shift_is_detected(self):
        report = assess_score_drift(self.reference, [.9] * 50)
        self.assertEqual(report["status"], "drift")
        self.assertGreaterEqual(report["ks_statistic"], .25)
        self.assertLessEqual(report["p_value"], .01)

    def test_stream_is_bounded_and_resets_for_new_model_or_source(self):
        state, first = update_score_drift(None, [.1, .2], self.reference,
                                          model_id="m1", stream_source="upload:a.json")
        self.assertEqual(first["current_samples"], 2)
        state, second = update_score_drift(state, [.3], self.reference,
                                           model_id="m1", stream_source="upload:a.json")
        self.assertEqual(state["observed_total"], 3)
        self.assertEqual(second["current_samples"], 3)
        state, reset = update_score_drift(state, [.4], self.reference,
                                          model_id="m2", stream_source="upload:a.json")
        self.assertEqual(state["observed_total"], 1)
        self.assertEqual(reset["current_samples"], 1)
        self.assertEqual(state["scores"], [.4])

    def test_invalid_scores_are_rejected(self):
        with self.assertRaises(ValueError):
            assess_score_drift([.1], [.2])
        with self.assertRaises(ValueError):
            assess_score_drift(self.reference, [1.2])
        with self.assertRaises(ValueError):
            update_score_drift(None, [], self.reference, model_id="m1", stream_source="test")


if __name__ == "__main__":
    unittest.main()
