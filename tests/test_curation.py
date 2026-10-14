import json
import unittest
from pathlib import Path

from ran_monitor_ai.curation import compile_reviewed_cause_records
from ran_monitor_ai.features import true_cause, window_identity


class ReviewedFeedbackCurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads((Path(__file__).resolve().parents[1] / "sample_window.json")
                                .read_text(encoding="utf-8"))
        cls.window_id = window_identity(cls.record)

    def review(self, cause="Equipment", **changes):
        return ({"alert_id": "ALT-1", "window_id": self.window_id,
                 "confirmed_cause": cause, "confidence": "high", "reviewer": "reviewer-1",
                 "note": "现场记录核实", "feedback_version": "cause-review-v1",
                 "created_at": "2026-10-13T08:00:00Z"} | changes)

    def test_exact_window_review_is_compiled_with_audit_provenance(self):
        compiled, report = compile_reviewed_cause_records([self.record], [self.review()])
        self.assertEqual(len(compiled), 1)
        self.assertEqual(compiled[0]["reviewed_cause"], "Equipment")
        self.assertEqual(true_cause(compiled[0]), "Equipment")
        self.assertEqual(compiled[0]["review_metadata"]["source_window_id"], self.window_id)
        self.assertFalse(report["training_eligible"])
        self.assertTrue(report["ready_for_data_owner_review"])

    def test_low_confidence_feedback_is_excluded_from_reviewed_label_set(self):
        compiled, report = compile_reviewed_cause_records(
            [self.record], [self.review(confidence="low")],
        )
        self.assertEqual(compiled, [])
        self.assertEqual(report["skipped_rows"]["low_or_undetermined_confidence"], 1)

    def test_conflicting_cause_reviews_are_reported_and_not_compiled(self):
        compiled, report = compile_reviewed_cause_records(
            [self.record], [self.review(), self.review("Interference")],
        )
        self.assertEqual(compiled, [])
        self.assertEqual(report["conflicting_window_ids"], [self.window_id])
        self.assertFalse(report["ready_for_data_owner_review"])


if __name__ == "__main__":
    unittest.main()
