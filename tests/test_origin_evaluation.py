import unittest

from ran_monitor_ai.evaluation import anomaly_origin, evaluate_by_origin


class OriginEvaluationTests(unittest.TestCase):
    def test_origin_metadata_aliases_are_normalized_and_missing_is_unknown(self):
        self.assertEqual(anomaly_origin({"anomaly_origin": "Synthetic"}), "synthetic")
        self.assertEqual(anomaly_origin({"labels": {"origin": "受控真实"}}), "controlled_real")
        self.assertEqual(anomaly_origin({"anomalies": {"origin": "field"}}), "real")
        self.assertEqual(anomaly_origin({}), "unknown")

    def test_audit_reports_groups_and_keeps_unknown_separate(self):
        result = evaluate_by_origin(
            ["Normal", "Interference", "Normal", "Equipment"],
            [.1, .9, .8, .7], .5,
            ["synthetic", "synthetic", "unknown", "real"],
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["unknown_windows"], 1)
        synthetic = next(row for row in result["groups"] if row["origin"] == "synthetic")
        self.assertEqual(synthetic["confusion"], [[1, 0], [0, 1]])
        real = next(row for row in result["groups"] if row["origin"] == "real")
        self.assertEqual(real["recall"], 1.0)

    def test_audit_marks_all_missing_provenance(self):
        result = evaluate_by_origin(["Normal", "Equipment"], [.1, .9], .5, ["unknown", "unknown"])
        self.assertEqual(result["status"], "missing_origin")
        self.assertEqual(result["unknown_windows"], 2)

    def test_audit_rejects_misaligned_data(self):
        with self.assertRaises(ValueError):
            evaluate_by_origin(["Normal"], [.1, .2], .5, ["unknown"])


if __name__ == "__main__":
    unittest.main()
