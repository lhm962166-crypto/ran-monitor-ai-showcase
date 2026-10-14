import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ran_monitor_ai.engine import score_records
from ran_monitor_ai.features import extract_features
from ran_monitor_ai.settings import KPIS
from ran_monitor_ai.train import measured_feature_names


class CountingDetector:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict_proba(self, rows):
        self.calls += 1
        scores = np.asarray(self.scores if isinstance(self.scores, list)
                            else [self.scores] * len(rows))
        return np.column_stack((1 - scores, scores))


class CountingClassifier:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.calls = 0
        self.row_count = 0

    def predict_proba(self, rows):
        self.calls += 1
        self.row_count += len(rows)
        return np.tile(self.probabilities, (len(rows), 1))


class BatchInferenceTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((Path(__file__).resolve().parents[1] / "sample_window.json")
                                 .read_text(encoding="utf-8"))
        self.features = extract_features(self.record)

    def bundle(self, scores, probabilities=(.9, .04, .03, .03)):
        names = list(self.features)
        return {
            "detector": CountingDetector(scores),
            "classifier": CountingClassifier(probabilities),
            "threshold": .5, "feature_names": names,
            "kpi_features": measured_feature_names(names),
            "cause_features": measured_feature_names(names),
            "medians": dict.fromkeys(names, 0.),
            "baselines": {"__global__": {
                f"{name}_mean": {"mean": self.features[f"{name}_mean"], "std": 1.}
                for name in KPIS
            }},
            "classes": ["Equipment", "Interference", "Mobility", "Congestion"],
        }

    def test_models_are_called_once_and_only_anomalies_get_classified(self):
        records = [self.record | {"sample_id": name} for name in ("first", "second", "third")]
        bundle = self.bundle([.8, .2, .5])
        results = score_records(records, bundle)
        self.assertEqual([r["sample_id"] for r in results], ["first", "second", "third"])
        self.assertEqual([r["detected"] for r in results], [True, False, True])
        self.assertEqual(bundle["detector"].calls, 1)
        self.assertEqual(bundle["classifier"].calls, 1)
        self.assertEqual(bundle["classifier"].row_count, 2)
        self.assertIsNone(results[1]["alert"])

    def test_normal_batch_skips_root_cause_model(self):
        bundle = self.bundle(.1)
        results = score_records([self.record, self.record], bundle)
        self.assertEqual(bundle["classifier"].calls, 0)
        self.assertFalse(any(r["detected"] for r in results))
        self.assertEqual(score_records([], bundle), [])

    def test_precomputed_features_preserve_scores_and_validate_row_count(self):
        bundle = self.bundle(.8)
        results = score_records([self.record], bundle,
                                feature_rows=pd.DataFrame([self.features]))
        self.assertEqual(results[0]["score"], .8)
        with self.assertRaisesRegex(ValueError, "行数"):
            score_records([self.record], bundle, feature_rows=pd.DataFrame())

    def test_ambiguous_root_cause_requires_human_review(self):
        result = score_records([self.record], self.bundle(.8, (.4, .35, .15, .1)))[0]
        self.assertTrue(result["alert"]["requires_review"])
        self.assertAlmostEqual(result["alert"]["confidence_margin"], .05)
        self.assertEqual(set(result["alert"]["cause_probabilities"]),
                         {"Equipment", "Interference", "Mobility", "Congestion"})
        certain = score_records([self.record], self.bundle(.8))[0]
        # High raw model scores do not replace the missing independent calibration evidence.
        self.assertTrue(certain["alert"]["requires_review"])
        self.assertEqual(certain["alert"]["cause_set_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
