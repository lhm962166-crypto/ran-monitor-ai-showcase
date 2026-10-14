import unittest

import numpy as np

from ran_monitor_ai.data_audit import temporal_coverage_audit


class TemporalCoverageAuditTests(unittest.TestCase):
    def make_records(self, *, include_entity=False):
        records = []
        for index in range(10):
            minute = index
            record = {
                "start_time": f"2026-01-01T00:{minute:02d}:00",
                "end_time": f"2026-01-01T00:{minute:02d}:30",
            }
            if include_entity:
                record["cell_id"] = "cell-a" if index < 5 else "cell-b"
                if index >= 6:
                    record["event_id"] = "incident-1"
            records.append(record)
        return records

    def test_reports_temporal_blocks_and_cautions_without_stream_identity(self):
        records = self.make_records()
        labels = np.array(["Normal"] * 6 + ["Equipment"] * 4)
        result = temporal_coverage_audit(records, labels, np.arange(10), blocks=5)
        self.assertEqual(result["windows"], 10)
        self.assertEqual(result["overlap_groups"], 10)
        self.assertEqual(len(result["chronological_blocks"]), 5)
        self.assertFalse(result["has_explicit_stream_identity"])
        self.assertFalse(result["forward_evaluation_data_eligible"])
        self.assertTrue(result["latest_block_contains_anomaly"])

    def test_full_entity_identity_and_labeled_future_block_enable_audit(self):
        records = self.make_records(include_entity=True)
        labels = np.array(["Normal"] * 6 + ["Equipment"] * 4)
        result = temporal_coverage_audit(records, labels, np.arange(10), blocks=5)
        self.assertTrue(result["has_explicit_stream_identity"])
        self.assertTrue(result["has_explicit_event_identity"])
        self.assertTrue(result["forward_evaluation_data_eligible"])
        self.assertEqual(result["entity_identity_coverage"]["cell_id"]["fraction"], 1.0)

    def test_scenario_id_alone_does_not_count_as_ran_entity_identity(self):
        records = self.make_records()
        for record in records:
            record["scenario_id"] = "scenario-a"
        labels = np.array(["Normal"] * 6 + ["Equipment"] * 4)
        result = temporal_coverage_audit(records, labels, np.arange(10), blocks=5)
        self.assertFalse(result["has_explicit_stream_identity"])
        self.assertNotIn("scenario_id", result["entity_identity_coverage"])
        self.assertFalse(result["forward_evaluation_data_eligible"])

    def test_rejects_misaligned_and_invalid_block_arguments(self):
        with self.assertRaisesRegex(ValueError, "等长"):
            temporal_coverage_audit(self.make_records(), ["Normal"], np.arange(10))
        with self.assertRaisesRegex(ValueError, "分段数"):
            temporal_coverage_audit(self.make_records(), ["Normal"] * 10, np.arange(10), blocks=1)


if __name__ == "__main__":
    unittest.main()
