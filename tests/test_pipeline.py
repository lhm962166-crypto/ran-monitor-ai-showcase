import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.engine import score_record
from ran_monitor_ai.exporting import dataframe_csv_bytes
from ran_monitor_ai.features import extract_features, parse_json_payload, summarize_series, true_cause, validate_batch, validate_window
from ran_monitor_ai.settings import KPIS
from ran_monitor_ai.store import recent, save, update_status
from ran_monitor_ai.train import group_overlapping_windows, measured_feature_names, split_time_groups


class FeaturePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "sample_window.json").open(encoding="utf-8") as handle:
            cls.record = json.load(handle)

    def test_window_shape_and_feature_contract(self):
        validated = validate_window(self.record)
        self.assertEqual(set(validated), set(KPIS))
        self.assertTrue(all(len(values) == 128 for values in validated.values()))
        features = extract_features(self.record)
        self.assertEqual(len(features), 194)
        self.assertAlmostEqual(features["RSRP_mean"], sum(validated["RSRP"]) / 128)

    def test_vectorized_statistics_match_independent_reference(self):
        rng = np.random.default_rng(42)
        for values in (np.full(128, 8.), np.linspace(-30, 20, 128), rng.normal(0, 5, 128)):
            with self.subTest(kind=float(values.std())):
                actual = summarize_series(values)
                self.assertAlmostEqual(actual["slope"], np.polyfit(np.arange(128), values, 1)[0], places=10)
                self.assertAlmostEqual(actual["mean"], values.mean(), places=10)
                self.assertAlmostEqual(actual["std"], values.std(), places=10)
                self.assertAlmostEqual(actual["iqr"], np.percentile(values, 75) - np.percentile(values, 25), places=10)
                if values.std() > 1e-9:
                    self.assertAlmostEqual(actual["skew"], skew(values), places=10)
                    self.assertAlmostEqual(actual["kurt"], kurtosis(values), places=10)
                else:
                    self.assertEqual(actual["skew"], 0.)
                    self.assertEqual(actual["kurt"], 0.)

    def test_fault_taxonomy_maps_to_coarse_cause(self):
        self.assertEqual(true_cause(self.record), "Interference")

    def test_batch_validation_names_bad_window_and_enforces_limit(self):
        valid = validate_batch(self.record)
        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["sample_id"], self.record["sample_id"])

        bad = json.loads(json.dumps(self.record))
        bad["sample_id"] = 999
        bad["KPIs"]["UL_SNR"] = ["not-a-number"]
        with self.assertRaisesRegex(ValueError, "窗口 999.*UL_SNR"):
            validate_batch([self.record, bad])
        bad_labels = json.loads(json.dumps(self.record))
        bad_labels["labels"] = "unexpected string"
        with self.assertRaisesRegex(ValueError, "labels 须为 JSON 对象"):
            validate_batch(bad_labels)
        with self.assertRaisesRegex(ValueError, "最多处理 1 个窗口"):
            validate_batch([self.record, self.record], max_records=1)

    def test_uploaded_json_errors_point_to_the_problem(self):
        self.assertEqual(parse_json_payload(b'{"ok": true}'), {"ok": True})
        with self.assertRaisesRegex(ValueError, "第 1 行、第 8 列"):
            parse_json_payload(b'{"ok": nope}')
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            parse_json_payload(b"\xff")

    def test_root_cause_inputs_exclude_target_and_context_labels(self):
        selected = measured_feature_names([
            "RSRP_mean", "DL_BLER_std", "zone_A", "application_File", "mobility", "congestion",
        ])
        self.assertEqual(selected, ["RSRP_mean", "DL_BLER_std"])

    def test_overlapping_time_windows_stay_in_the_same_split_group(self):
        groups = group_overlapping_windows([
            {"start_time": "2026-01-01T00:00:00", "end_time": "2026-01-01T00:00:12.8"},
            {"start_time": "2026-01-01T00:00:06", "end_time": "2026-01-01T00:00:18.8"},
            {"start_time": "2026-01-01T02:00:10+02:00", "end_time": "2026-01-01T02:00:18+02:00"},
            {"start_time": "2026-01-01T00:01:00", "end_time": "2026-01-01T00:01:12.8"},
        ])
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])

    def test_time_group_split_never_shares_overlapping_windows(self):
        records = []
        for group in range(10):
            start = f"2026-01-01T00:{group:02}:00"
            end = f"2026-01-01T00:{group:02}:12"
            records.extend([
                {"start_time": start, "end_time": end},
                {"start_time": start, "end_time": end},
            ])
        labels = np.array(["Normal", "Interference"] * 10)
        anomaly = (labels != "Normal").astype(int)
        train_idx, val_idx, test_idx, note = split_time_groups(records, anomaly, labels)
        groups = group_overlapping_windows(records)
        partitions = [set(groups[index] for index in part) for part in (train_idx, val_idx, test_idx)]
        self.assertFalse(partitions[0] & partitions[1])
        self.assertFalse(partitions[0] & partitions[2])
        self.assertFalse(partitions[1] & partitions[2])
        self.assertIn("时间窗口分组", note)

    def test_csv_export_escapes_formula_text_but_preserves_numbers(self):
        frame = pd.DataFrame({"sample_id": ["  =1+1", "safe"], "score": [-0.5, 1.0]})
        csv = dataframe_csv_bytes(frame).decode("utf-8-sig")
        self.assertIn("'  =1+1", csv)
        self.assertIn("-0.5", csv)
        self.assertIn("safe", csv)

    def test_each_repeated_detection_gets_a_unique_event_id(self):
        class AlwaysAnomaly:
            def predict_proba(self, _row):
                return np.array([[.01, .99]])

        class EquipmentClassifier:
            feature_count = None

            def predict_proba(self, _row):
                self.feature_count = _row.shape[1]
                return np.array([[1., 0., 0., 0.]])

        feature_names = list(extract_features(self.record))
        cause_features = measured_feature_names(feature_names)
        classifier = EquipmentClassifier()
        baseline = {
            f"{name}_mean": {"mean": float(np.mean(self.record["KPIs"][name])), "std": 1.}
            for name in KPIS
        }
        evidence_ranker = {
            "method": "test_prototype",
            "classes": ["Equipment", "Interference", "Mobility", "Congestion"],
            "feature_names": cause_features,
            "scale": [1.] * len(cause_features),
            "prototypes": [[0.] * len(cause_features) for _ in range(4)],
            "classifier_weight": 1.0,
        }
        bundle = {
            "detector": AlwaysAnomaly(), "classifier": classifier,
            "threshold": .5, "feature_names": feature_names,
            "cause_features": cause_features,
            "cause_evidence_ranker": evidence_ranker,
            "kpi_features": [name for name in feature_names
                             if not name.startswith(("zone_", "application_"))
                             and name not in {"mobility", "congestion"}],
            "medians": {name: 0. for name in feature_names},
            "baselines": {"__global__": baseline},
            "classes": ["Equipment", "Interference", "Mobility", "Congestion"],
        }
        first = score_record(self.record, bundle)
        second = score_record(self.record, bundle)
        self.assertTrue(first["detected"] and second["detected"])
        self.assertNotEqual(first["alert"]["alert_id"], second["alert"]["alert_id"])
        self.assertEqual(first["alert"]["severity"], "P4")
        self.assertIn("未观察到明确 KPI 影响证据", first["alert"]["severity_reason"])
        self.assertEqual(first["alert"]["confidence_level"], "高")
        self.assertEqual(first["alert"]["cause_ranking_shadow"]["mode"], "shadow")
        self.assertFalse(first["alert"]["cause_ranking_shadow"]["probability_claim"])
        self.assertEqual(classifier.feature_count, len(cause_features))


class AlertHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "alerts.sqlite3"

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def result(alert_id):
        return {
            "sample_id": 42,
            "detected": True,
            "score": .98,
            "alert": {
                "alert_id": alert_id,
                "severity": "P1",
                "cause": "Equipment",
                "affected_kpis": [],
            },
        }

    def test_repeated_detection_creates_distinct_history_events(self):
        save(self.result("ALT-one"), self.db_path)
        save(self.result("ALT-two"), self.db_path)

        alerts = recent(path=self.db_path)
        self.assertEqual(len(alerts), 2)
        self.assertTrue(all(alert["status"] == "OPEN" for alert in alerts))

    def test_alert_can_be_acknowledged_resolved_and_reopened(self):
        save(self.result("ALT-flow"), self.db_path)

        self.assertTrue(update_status("ALT-flow", "ACKNOWLEDGED", self.db_path))
        acknowledged = recent(path=self.db_path)[0]
        self.assertEqual(acknowledged["status"], "ACKNOWLEDGED")
        self.assertIsNotNone(acknowledged["acknowledged_at"])

        self.assertTrue(update_status("ALT-flow", "RESOLVED", self.db_path))
        resolved = recent(path=self.db_path)[0]
        self.assertEqual(resolved["status"], "RESOLVED")
        self.assertIsNotNone(resolved["resolved_at"])

        self.assertTrue(update_status("ALT-flow", "OPEN", self.db_path))
        reopened = recent(path=self.db_path)[0]
        self.assertEqual(reopened["status"], "OPEN")
        self.assertIsNone(reopened["acknowledged_at"])
        self.assertIsNone(reopened["resolved_at"])
        self.assertFalse(update_status("missing", "OPEN", self.db_path))
        with self.assertRaises(ValueError):
            update_status("ALT-flow", "INVALID", self.db_path)

    def test_existing_database_is_upgraded_without_losing_alerts(self):
        db = sqlite3.connect(self.db_path)
        try:
            db.execute("""CREATE TABLE alerts (
                alert_id TEXT PRIMARY KEY, sample_id TEXT, severity TEXT NOT NULL,
                cause TEXT NOT NULL, score REAL NOT NULL, payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""")
            db.execute("""INSERT INTO alerts VALUES
                ('ALT-old', '7', 'P2', 'Interference', .9, '{}', '2026-10-01T00:00:00+00:00')""")
            db.commit()
        finally:
            db.close()

        alerts = recent(path=self.db_path)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["alert_id"], "ALT-old")
        self.assertEqual(alerts[0]["status"], "OPEN")
        db = sqlite3.connect(self.db_path)
        try:
            columns = {row[1] for row in db.execute("PRAGMA table_info(alerts)")}
        finally:
            db.close()
        self.assertTrue({"status", "acknowledged_at", "resolved_at"}.issubset(columns))


if __name__ == "__main__":
    unittest.main()
