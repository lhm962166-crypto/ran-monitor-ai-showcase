import json
import tempfile
import unittest
from pathlib import Path

from ran_monitor_ai.demo import synthetic_records, write_synthetic_dataset
from ran_monitor_ai.evaluation import anomaly_origin
from ran_monitor_ai.features import true_cause, validate_batch


class SyntheticDemoTests(unittest.TestCase):
    def test_generated_records_are_reproducible_complete_and_explicitly_synthetic(self):
        first = synthetic_records(per_cause=10, seed=17)
        second = synthetic_records(per_cause=10, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertEqual({true_cause(row) for row in first},
                         {"Normal", "Interference", "Congestion", "Equipment", "Mobility"})
        self.assertEqual({anomaly_origin(row) for row in first}, {"synthetic"})
        checked = validate_batch(first, max_records=100)
        self.assertEqual(len(checked), 50)
        self.assertTrue(all(len(values) == 128 for row in checked for values in row["KPIs"].values()))

    def test_writer_creates_jsonl_without_using_project_data_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            target = write_synthetic_dataset(Path(folder) / "nested" / "demo.jsonl", per_cause=10)
            rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 50)
            self.assertTrue(all(row["data_source"] == "synthetic_generated_demo" for row in rows))


if __name__ == "__main__":
    unittest.main()
