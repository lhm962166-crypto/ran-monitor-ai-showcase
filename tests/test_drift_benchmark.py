import json
import unittest

import numpy as np

from ran_monitor_ai.drift_benchmark import benchmark_score_drift


class DriftBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.random.default_rng(7).beta(1.8, 7.0, size=48)

    def test_benchmark_is_reproducible_and_json_safe(self):
        first = benchmark_score_drift(self.reference, model_id="fixture", replicates=20,
                                      horizon=60, batch_size=10, seed=9)
        second = benchmark_score_drift(self.reference, model_id="fixture", replicates=20,
                                       horizon=60, batch_size=10, seed=9)
        self.assertEqual(first, second)
        self.assertEqual(set(first["scenarios"]), {"stable", "mean_shift", "spread_shift", "surge"})
        self.assertIn("synthetic", first["scope_note"].lower())
        json.dumps(first, allow_nan=False)

    def test_reports_operational_false_alert_and_detection_metrics(self):
        result = benchmark_score_drift(self.reference, replicates=20, horizon=60,
                                       batch_size=10, seed=11)
        stable = result["scenarios"]["stable"]
        self.assertGreaterEqual(stable["false_alert_probability"], 0)
        self.assertLessEqual(stable["false_alert_probability"], 1)
        self.assertEqual(len(stable["false_alert_probability_95ci"]), 2)
        for scenario in ("mean_shift", "spread_shift", "surge"):
            report = result["scenarios"][scenario]
            self.assertIn("post_change_detection_probability", report)
            self.assertIn("median_detection_delay_windows", report)
            self.assertGreater(report["change_point_window"], 0)

    def test_rejects_small_reference_and_invalid_protocol(self):
        with self.assertRaisesRegex(ValueError, "至少需要 30"):
            benchmark_score_drift([0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "20–5000"):
            benchmark_score_drift(self.reference, replicates=2)
        with self.assertRaisesRegex(ValueError, "流长度"):
            benchmark_score_drift(self.reference, replicates=20, horizon=20)


if __name__ == "__main__":
    unittest.main()
