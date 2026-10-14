import unittest

from ran_monitor_ai.open_set_experiment import _threshold_at_fpr, _upper_tail_evidence


class EqualFalseAlarmExperimentTests(unittest.TestCase):
    def test_threshold_uses_only_normal_scores_and_respects_requested_budget(self):
        threshold, observed = _threshold_at_fpr([.1, .2, .3, .4], .25)
        self.assertEqual(threshold, .4)
        self.assertEqual(observed, .25)
        zero_threshold, zero_fpr = _threshold_at_fpr([.1, .2, .3, .4], 0.0)
        self.assertGreater(zero_threshold, .4)
        self.assertEqual(zero_fpr, 0.0)

    def test_upper_tail_evidence_is_monotone_with_score(self):
        evidence = _upper_tail_evidence([.1, .2, .3, .4], [.1, .25, .5])
        self.assertLess(evidence[0], evidence[1])
        self.assertLess(evidence[1], evidence[2])


if __name__ == "__main__":
    unittest.main()
