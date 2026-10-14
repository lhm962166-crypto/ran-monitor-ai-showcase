import unittest

import numpy as np

from ran_monitor_ai.diagnosis import (
    choose_classifier_weight,
    evaluate_rank_matrix,
    fit_cause_evidence_ranker,
    fused_ranking_scores,
    prototype_distances,
    rank_cause_candidates,
)


class CauseEvidenceRankingTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["Congestion", "Equipment"]
        self.names = ["RX_Bytes_mean", "DL_BLER_mean"]
        self.training = np.array([
            [0.0, 0.0], [0.2, 0.1], [-0.1, 0.2],
            [10.0, 10.0], [10.2, 9.8], [9.8, 10.1],
        ])
        self.labels = np.array([0, 0, 0, 1, 1, 1])
        self.model = fit_cause_evidence_ranker(
            self.training, self.labels, self.classes, self.names,
        )

    def test_prototypes_identify_independent_kpi_evidence(self):
        distances = prototype_distances([[0.1, 0.1], [10.1, 9.9]], self.model)
        self.assertLess(distances[0, 0], distances[0, 1])
        self.assertLess(distances[1, 1], distances[1, 0])

    def test_validation_weight_selection_is_reproducible_and_prefers_evidence_when_needed(self):
        probabilities = np.array([[.9, .1], [.9, .1]])
        distances = prototype_distances([[10.1, 9.9], [0.1, 0.1]], self.model)
        selected = choose_classifier_weight(probabilities, distances, [1, 0])
        self.assertEqual(selected["classifier_weight"], 0.25)
        scores = fused_ranking_scores(probabilities, distances, selected["classifier_weight"])
        self.assertEqual(evaluate_rank_matrix(scores, [1, 0])["top1_accuracy"], 1.0)

    def test_ui_result_keeps_probability_and_shadow_rank_fields_separate(self):
        probabilities = np.array([.6, .4])
        ranked = rank_cause_candidates(
            probabilities, {"RX_Bytes_mean": 0.1, "DL_BLER_mean": 0.1}, self.model,
        )
        self.assertEqual(ranked["mode"], "shadow")
        self.assertFalse(ranked["probability_claim"])
        self.assertEqual(ranked["candidates"][0]["cause"], "Congestion")
        self.assertEqual(ranked["candidates"][0]["classifier_probability"], .6)

    def test_invalid_or_incomplete_training_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "每个原因类别"):
            fit_cause_evidence_ranker(self.training[:3], self.labels[:3], self.classes, self.names)
        with self.assertRaisesRegex(ValueError, "不匹配"):
            prototype_distances([[1.0]], self.model)


if __name__ == "__main__":
    unittest.main()
