import unittest

from ran_monitor_ai.model_benchmark import threshold_at_recall


class BaselineThresholdTests(unittest.TestCase):
    def test_chooses_strictest_validation_threshold_that_meets_recall(self):
        result = threshold_at_recall([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], 0.5)
        self.assertEqual(result["threshold"], 0.9)
        self.assertTrue(result["target_met"])
        self.assertEqual(result["validation_recall"], 0.5)
        self.assertEqual(result["validation_confusion"], [[2, 0], [1, 1]])

    def test_rejects_single_class_and_nonfinite_calibration(self):
        with self.assertRaisesRegex(ValueError, "校准需要等长"):
            threshold_at_recall([0, 0], [0.1, 0.2])
        with self.assertRaisesRegex(ValueError, "有限数值"):
            threshold_at_recall([0, 1], [0.1, float("nan")])
        with self.assertRaisesRegex(ValueError, "目标召回率"):
            threshold_at_recall([0, 1], [0.1, 0.9], 0)


if __name__ == "__main__":
    unittest.main()
