import unittest

from ran_monitor_ai.governance import promotion_readiness


class PromotionGateTests(unittest.TestCase):
    def complete_evidence(self):
        return {
            "test_size": 1200,
            "evaluation_protocol": "forward_time_holdout",
            "origin_audit": {"status": "ready", "known_origins": ["synthetic", "controlled_real"],
                             "unknown_windows": 0},
            "temporal_coverage_audit": {
                "forward_evaluation_data_eligible": True,
                "has_explicit_stream_identity": True,
                "has_explicit_event_identity": True,
                "latest_block_contains_anomaly": True,
            },
            "test_metric_uncertainty": {"metrics": {"f1": {"ci95": [0.7, 0.9]}}},
        }

    def test_complete_evidence_passes_for_review_not_automatic_deployment(self):
        result = promotion_readiness(self.complete_evidence())
        self.assertEqual(result["status"], "ready_for_review")
        self.assertTrue(result["promotion_allowed"])
        self.assertIn("人工批准", result["note"])

    def test_missing_origin_or_event_identity_holds_promotion(self):
        metrics = self.complete_evidence()
        metrics["origin_audit"]["unknown_windows"] = 1
        metrics["temporal_coverage_audit"]["has_explicit_event_identity"] = False
        result = promotion_readiness(metrics)
        self.assertEqual(result["status"], "hold")
        self.assertFalse(result["promotion_allowed"])
        failures = {row["key"] for row in result["checks"] if not row["passed"]}
        self.assertIn("origin_coverage", failures)
        self.assertIn("stream_and_event_identity", failures)

    def test_complete_metadata_still_holds_without_forward_time_test(self):
        metrics = self.complete_evidence()
        metrics["evaluation_protocol"] = "overlap_group_stratified_holdout"
        result = promotion_readiness(metrics)
        self.assertFalse(result["promotion_allowed"])
        failed = {row["key"] for row in result["checks"] if not row["passed"]}
        self.assertIn("forward_time_evaluation", failed)

    def test_rejects_invalid_minimum_test_size(self):
        with self.assertRaisesRegex(ValueError, "正整数"):
            promotion_readiness({}, min_test_windows=0)

    def test_invalid_f1_uncertainty_cannot_pass_review_gate(self):
        for interval in ([0.9, 0.7], [-0.1, 0.8], [0.2, 1.1], [float("nan"), 0.9],
                         [True, 0.9], [10**1000, 0.9], "0.7"):
            with self.subTest(interval=interval):
                metrics = self.complete_evidence()
                metrics["test_metric_uncertainty"]["metrics"]["f1"]["ci95"] = interval
                result = promotion_readiness(metrics)
                self.assertFalse(result["promotion_allowed"])
                failed = {row["key"] for row in result["checks"] if not row["passed"]}
                self.assertIn("uncertainty_estimate", failed)


if __name__ == "__main__":
    unittest.main()
