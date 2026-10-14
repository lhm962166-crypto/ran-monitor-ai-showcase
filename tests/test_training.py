"""Model snapshots and independent evaluation partitions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

import ran_monitor_ai.train as training
from ran_monitor_ai.evaluation import evaluate_two_stage


class ModelArtifactTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.data = self.root / "windows.jsonl"
        self.data.write_text('{"sample_id": 1}\n', encoding="utf-8")
        self.artifact = self.root / "model.joblib"
        self.paths = patch.multiple(training, ARTIFACT_PATH=self.artifact)
        self.paths.start()
        self.data_path = patch.object(training, "get_data_path", return_value=self.data)
        self.data_path.start()
        self.builder = patch.object(training, "build_bundle", side_effect=lambda *_args, **_kwargs: ({"detector": "fixture"}, {"samples": 1}))
        self.build = self.builder.start()

    def tearDown(self):
        self.builder.stop()
        self.data_path.stop()
        self.paths.stop()
        self.folder.cleanup()

    def test_snapshot_reuses_current_model_and_metrics_are_embedded(self):
        first, metrics = training.train()
        (self.root / "metrics.json").write_text('{"samples": 999}', encoding="utf-8")
        second, actual = training.train()
        self.assertEqual(self.build.call_count, 1)
        self.assertEqual(actual, metrics)
        self.assertEqual(first["metadata"], second["metadata"])
        self.assertEqual(actual["samples"], 1)

    def test_data_change_invalidates_previous_snapshot(self):
        first, _ = training.train()
        self.data.write_text('{"sample_id": 2}\n', encoding="utf-8")
        second, _ = training.train()
        self.assertEqual(self.build.call_count, 2)
        self.assertNotEqual(first["metadata"]["fingerprint"], second["metadata"]["fingerprint"])

    def test_legacy_and_truncated_artifacts_are_rebuilt(self):
        for contents in (None, b"incomplete model"):
            with self.subTest(contents=contents):
                if contents is None:
                    joblib.dump({"detector": "legacy"}, self.artifact)
                else:
                    self.artifact.write_bytes(contents)
                bundle, _ = training.train()
                self.assertEqual(bundle["metadata"]["format_version"], training.MODEL_FORMAT_VERSION)
        self.assertEqual(self.build.call_count, 2)

    def test_failed_replacement_preserves_readable_previous_model(self):
        training.train()
        previous = self.artifact.read_bytes()
        with patch.object(training.os, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                training.train(force=True)
        self.assertEqual(self.artifact.read_bytes(), previous)
        self.assertFalse(list(self.root.glob(".model-*.tmp")))
        self.assertIsInstance(joblib.load(self.artifact), dict)

    def test_missing_external_dataset_reports_how_to_configure_it(self):
        missing = self.root / "not-downloaded.jsonl"
        with patch.object(training, "get_data_path", return_value=missing):
            with self.assertRaisesRegex(FileNotFoundError, "RAN_MONITOR_DATA_PATH"):
                training.load_records()


class IndependentPartitionTests(unittest.TestCase):
    @staticmethod
    def _window(minute, *, end_minute=None, stream="cell-a", event=None, cause="Normal"):
        end_minute = minute + 1 if end_minute is None else end_minute
        row = {
            "start_time": f"2026-01-01T00:{minute:02}:00",
            "end_time": f"2026-01-01T00:{end_minute:02}:00",
            "stream_id": stream,
            "anomalies": {"exists": cause != "Normal", "type": "Jamming" if cause != "Normal" else None},
        }
        if event is not None:
            row["event_id"] = event
        return row

    def test_disjoint_windows_from_one_event_stay_in_one_group(self):
        records = [self._window(0, stream="cell-a", event="incident-7"),
                   self._window(20, stream="cell-b", event="incident-7"),
                   self._window(40, event="incident-8")]
        groups = training.group_overlapping_windows(records)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[1], groups[2])

    def test_event_identity_aliases_are_respected_in_nested_metadata(self):
        records = [self._window(0), self._window(20)]
        records[0]["metadata"] = {"fault_event_id": "event-42"}
        records[1]["labels"] = {"anomaly_event_id": "event-42"}
        groups = training.group_overlapping_windows(records)
        self.assertEqual(groups[0], groups[1])

    def test_overlapping_windows_in_one_stream_join_despite_different_events(self):
        records = [self._window(0, end_minute=5, event="incident-a"),
                   self._window(4, end_minute=8, event="incident-b"),
                   self._window(20, event="incident-c")]
        groups = training.group_overlapping_windows(records)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[1], groups[2])

    def test_unknown_stream_overlap_joins_named_stream_conservatively(self):
        records = [self._window(0, end_minute=5, stream=None, event="incident-a"),
                   self._window(4, end_minute=8, stream="cell-a", event="incident-b"),
                   self._window(20, stream="cell-b", event="incident-c")]
        records[0].pop("stream_id")
        groups = training.group_overlapping_windows(records)
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[1], groups[2])

    def test_event_and_overlap_links_are_closed_transitively(self):
        records = [self._window(0, end_minute=10, stream="cell-a", event="incident-a"),
                   self._window(5, end_minute=15, stream="cell-a", event="incident-b"),
                   self._window(30, end_minute=40, stream="cell-b", event="incident-a")]
        groups = training.group_overlapping_windows(records)
        self.assertEqual(len(set(groups.tolist())), 1)

    def test_cause_weights_balance_classes_and_overlap_groups(self):
        groups = np.array(["a", "a", "b", "c", "c", "d"])
        labels = np.array([0, 0, 0, 1, 1, 1])
        weights = training.overlap_group_class_weights(groups, labels)
        self.assertAlmostEqual(float(weights[0]), float(weights[1]))
        self.assertAlmostEqual(float(weights[0] + weights[1]), float(weights[2]))
        self.assertAlmostEqual(float(weights[3] + weights[4]), float(weights[5]))
        self.assertAlmostEqual(float(weights[labels == 0].sum()), float(weights[labels == 1].sum()))
        self.assertAlmostEqual(float(weights.mean()), 1.)

    def test_cause_weights_reject_misaligned_or_missing_group_identity(self):
        with self.assertRaisesRegex(ValueError, "输入无效"):
            training.overlap_group_class_weights(["a", "b"], [0])
        with self.assertRaisesRegex(ValueError, "输入无效"):
            training.overlap_group_class_weights(["a", None], [0, 1])

    def test_too_few_time_groups_are_rejected_without_random_fallback(self):
        records = [{"start_time": "2026-01-01T00:00:00", "end_time": "2026-01-01T00:01:00"}] * 20
        labels = np.array(["Normal", "Interference"] * 10)
        with self.assertRaisesRegex(ValueError, "至少需要 3 个不重叠时间组"):
            training.split_time_groups(records, (labels != "Normal").astype(int), labels)

    def test_three_independent_groups_can_each_cover_both_labels(self):
        records = []
        for minute in range(3):
            records.extend([{"start_time": f"2026-01-01T00:0{minute}:00",
                             "end_time": f"2026-01-01T00:0{minute}:30"}] * 2)
        labels = np.array(["Normal", "Interference"] * 3)
        partitions = training.split_time_groups(records, (labels != "Normal").astype(int), labels)
        groups = training.group_overlapping_windows(records)
        self.assertTrue(all(len(part) == 2 for part in partitions[:3]))
        self.assertEqual(len({int(groups[part[0]]) for part in partitions[:3]}), 3)

    def test_training_rejects_unknown_intervals_and_mixed_timezones(self):
        labels = np.array(["Normal", "Interference"])
        with self.assertRaisesRegex(ValueError, "无法保证独立划分"):
            training.split_time_groups([{}, {}], np.array([0, 1]), labels)
        records = [{"start_time": "2026-01-01T00:00:00", "end_time": "2026-01-01T00:01:00"},
                   {"start_time": "2026-01-01T00:00:00+00:00", "end_time": "2026-01-01T00:01:00+00:00"}]
        with self.assertRaisesRegex(ValueError, "统一时区"):
            training.split_time_groups(records, np.array([0, 1]), labels)

    def test_forward_split_rejects_an_event_group_that_crosses_its_time_cut(self):
        records = [self._window(0, end_minute=1, event="repeated-event")]
        records.extend(self._window(minute, end_minute=minute + 1, stream=f"cell-{minute}",
                                    event=f"event-{minute}", cause=("Interference" if minute % 2 else "Normal"))
                       for minute in range(1, 9))
        records.append(self._window(9, end_minute=10, stream="cell-late", event="repeated-event",
                                    cause="Interference"))
        labels = np.array(["Normal" if row["anomalies"]["exists"] is False else "Interference"
                           for row in records])
        with self.assertRaisesRegex(ValueError, "未能覆盖必要类别"):
            training.split_forward_time_groups(records, (labels != "Normal").astype(int), labels)

    def test_frozen_development_split_is_bound_to_data_and_keeps_groups_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            data_path = Path(folder) / "windows.jsonl"
            records = []
            for minute in range(3):
                for cause in ("Normal", "Interference"):
                    records.append({
                        "sample_id": f"{minute}-{cause}",
                        "start_time": f"2026-01-01T00:0{minute}:00",
                        "end_time": f"2026-01-01T00:0{minute}:30",
                        "anomalies": {"exists": cause != "Normal", "type": "Jamming"},
                    })
            data_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
            manifest = training.freeze_development_split(data_path)
            partitions = training.load_frozen_development_split(records, data_path)
            self.assertEqual(manifest["dataset_role"], "development_only_already_inspected")
            self.assertEqual([len(part) for part in partitions[:3]], [2, 2, 2])
            self.assertIn("不代表最终", partitions[3])

            manifest_path = training.frozen_development_split_path(data_path)
            corrupted = json.loads(manifest_path.read_text(encoding="utf-8"))
            corrupted["partitions"]["train"]["indices"][0] = True
            manifest_path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "数据不匹配"):
                training.load_frozen_development_split(records, data_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            data_path.write_text(data_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "数据不匹配"):
                training.load_frozen_development_split(records, data_path)


class EndToEndEvaluationTests(unittest.TestCase):
    def test_detector_misses_and_false_alarms_are_counted_in_workflow(self):
        result = evaluate_two_stage(
            ["Normal", "Mobility", "Interference", "Equipment", "Normal"],
            [0.2, 0.9, 0.1, 0.8, 0.95],
            ["Equipment", "Mobility", "Interference", "Equipment", "Equipment"],
            0.5,
            ["Equipment", "Interference", "Mobility"],
        )
        self.assertEqual(result["classes"], ["Normal", "Equipment", "Interference", "Mobility"])
        self.assertEqual(result["confusion"][0], [1, 1, 0, 0])
        self.assertEqual(result["confusion"][3], [0, 0, 0, 1])
        self.assertEqual(result["missed_faults"], 1)
        self.assertEqual(result["detected_faults"], 2)
        self.assertEqual(result["correctly_classified_faults"], 2)
        self.assertEqual(result["misclassified_detected_faults"], 0)
        self.assertEqual(result["false_alarms"], 1)

    def test_end_to_end_metrics_reject_misaligned_or_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "长度须相同"):
            evaluate_two_stage(["Normal"], [], [], 0.5, ["Equipment"])
        with self.assertRaisesRegex(ValueError, "有限数值"):
            evaluate_two_stage(["Normal"], [float("nan")], ["Equipment"], 0.5, ["Equipment"])


if __name__ == "__main__":
    unittest.main()
