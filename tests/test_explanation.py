import unittest

import numpy as np

from ran_monitor_ai.engine import explain_record, score_record
from tests.demo_fixture import get_demo_fixture


class DetectorExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.bundle, _ = get_demo_fixture()

    def test_contributions_reconstruct_probability_for_normal_and_fault_windows(self):
        for record in (self.records[0], self.records[-1]):
            with self.subTest(sample_id=record["sample_id"]):
                explanation = explain_record(record, self.bundle)
                total = explanation["base_value"] + sum(item["contribution"] for item in explanation["features"])
                self.assertAlmostEqual(total, explanation["raw_score"], places=5)
                probability = 1 / (1 + np.exp(-total))
                self.assertAlmostEqual(probability, score_record(record, self.bundle)["score"], places=6)
                self.assertAlmostEqual(sum(group["contribution"] for group in explanation["groups"]),
                                       total - explanation["base_value"], places=5)


if __name__ == "__main__":
    unittest.main()
