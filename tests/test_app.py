"""UI workflow tests use their own SQLite database and never send notifications."""
import os
import sqlite3
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from ran_monitor_ai.engine import score_records
from ran_monitor_ai.features import true_cause
from ran_monitor_ai.governance import promotion_readiness
from ran_monitor_ai.store import alert_events, query_alerts
from tests.demo_fixture import get_demo_fixture

ROOT = Path(__file__).resolve().parents[1]


def keyed(elements, key):
    return next(element for element in elements if element.key == key)


class DashboardWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        all_records, cls.bundle, cls.metrics = get_demo_fixture()
        records = [record for record in all_records if true_cause(record) != "Normal"]
        results = score_records(records, cls.bundle)
        cls.abnormal_id = next(record["sample_id"] for record, result in zip(records, results)
                               if result["detected"])

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.db_path = Path(self.folder.name) / "ui.sqlite3"
        self.environment = patch.dict(os.environ, {"RAN_MONITOR_DB_PATH": str(self.db_path)})
        self.environment.start()
        self.notifications = patch("ran_monitor_ai.notify.channels_ready",
                                   return_value={"email": False, "whatsapp": False})
        self.notifications.start()
        self.delivery = patch("ran_monitor_ai.notify.deliver_digest", side_effect=AssertionError("Unexpected notification"))
        self.delivery.start()
        self.app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()

    def tearDown(self):
        self.delivery.stop()
        self.notifications.stop()
        self.environment.stop()
        self.folder.cleanup()

    def assert_app_ok(self):
        self.assertEqual(len(self.app.exception), 0, [x.value for x in self.app.exception])

    def test_preview_detection_can_avoid_persisting_history_and_clear_results(self):
        self.assert_app_ok()
        self.assertEqual(len(self.app.tabs), 7)
        keyed(self.app.checkbox, "persist_alerts").set_value(False)
        keyed(self.app.selectbox, "live_sample_id").set_value(self.abnormal_id)
        keyed(self.app.button, "run_detection").click()
        self.app.run()
        self.assert_app_ok()
        self.assertTrue(self.app.session_state["last_batch"][0]["detected"])
        result_columns = {
            str(column)
            for frame in self.app.dataframe
            for column in frame.value.columns
        }
        self.assertIn("原因原始输出（未校准）", result_columns)
        self.assertEqual(query_alerts(path=self.db_path)[1], 0)
        keyed(self.app.button, "clear_detection_results").click()
        self.app.run()
        self.assert_app_ok()
        self.assertFalse(any(button.key == "clear_detection_results" for button in self.app.button))

    def test_detection_starts_persisted_score_drift_monitor(self):
        keyed(self.app.selectbox, "live_sample_id").set_value(self.abnormal_id)
        keyed(self.app.button, "run_detection").click()
        self.app.run()
        self.assert_app_ok()
        drift = self.app.session_state["last_detection_meta"]["score_drift"]
        self.assertEqual(drift["status"], "warming_up")
        self.assertEqual(drift["persistence"], "persistent")

    def test_detection_and_disposition_workflow_is_saved_and_audited(self):
        keyed(self.app.selectbox, "live_sample_id").set_value(self.abnormal_id)
        keyed(self.app.button, "run_detection").click()
        self.app.run()
        self.assert_app_ok()
        records, count = query_alerts(path=self.db_path)
        self.assertEqual(count, 1)
        alert_id = records[0]["alert_id"]
        keyed(self.app.text_area, f"history_note_{alert_id}").set_value("已核实指标，转交运维团队")
        keyed(self.app.button, "history_action_ACKNOWLEDGED").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(query_alerts(status="ACKNOWLEDGED", path=self.db_path)[1], 1)
        self.assertEqual(alert_events(alert_id, self.db_path)[0]["note"], "已核实指标，转交运维团队")
        keyed(self.app.button, "history_action_RESOLVED").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(query_alerts(status="RESOLVED", path=self.db_path)[1], 1)
        keyed(self.app.button, "history_action_OPEN").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(query_alerts(status="OPEN", path=self.db_path)[1], 1)
        self.assertEqual(len(alert_events(alert_id, self.db_path)), 4)

    def test_missing_upload_and_empty_filter_have_clear_feedback(self):
        keyed(self.app.radio, "data_source").set_value("上传 JSON")
        keyed(self.app.button, "run_detection").click()
        self.app.run()
        self.assert_app_ok()
        self.assertTrue(any("请先选择 JSON" in message.value for message in self.app.error))
        for severity in ("P1", "P2", "P3", "P4"):
            keyed(self.app.checkbox, f"severity_{severity}").set_value(False)
        self.app.run()
        self.assert_app_ok()
        self.assertTrue(any("当前筛选条件下没有告警" in message.value for message in self.app.info))
        self.assertEqual(query_alerts(path=self.db_path)[1], 0)

    def test_explanation_and_normal_comparison_render_without_errors(self):
        keyed(self.app.checkbox, "compare_normal_window").set_value(True)
        keyed(self.app.button, "inspector_explain").click()
        self.app.run()
        self.assert_app_ok()
        self.assertTrue(self.app.session_state["inspector_explanation"]["data"]["features"])
        self.assertTrue(any(select.key == "normal_reference_id" for select in self.app.selectbox))

    def test_replay_start_pause_resume_keeps_preview_isolated(self):
        keyed(self.app.button, "replay_start").click()
        self.app.run()
        self.assert_app_ok()
        replay = self.app.session_state["replay_run"]
        self.assertEqual(replay.cursor, 1)
        keyed(self.app.button, "replay_pause").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(replay.state, "paused")
        self.assertEqual(replay.cursor, 1)
        keyed(self.app.button, "replay_resume").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(replay.cursor, 2)
        self.assertEqual(query_alerts(path=self.db_path)[1], 0)

    def test_independent_note_and_filtered_history_export(self):
        keyed(self.app.selectbox, "live_sample_id").set_value(self.abnormal_id)
        keyed(self.app.button, "run_detection").click()
        self.app.run()
        alert_id = query_alerts(path=self.db_path)[0][0]["alert_id"]
        keyed(self.app.text_area, f"history_note_{alert_id}").set_value("已查看原始 KPI，等待现场复核")
        self.app.run()
        self.assert_app_ok()
        keyed(self.app.button, "history_add_note").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(query_alerts(path=self.db_path)[0][0]["status"], "OPEN")
        self.assertEqual(alert_events(alert_id, self.db_path)[0]["event_type"], "NOTE")
        keyed(self.app.button, "prepare_history_export").click()
        self.app.run()
        self.assert_app_ok()
        self.assertEqual(self.app.session_state["history_export"]["size"], 1)

    def test_unavailable_history_does_not_remove_detection_evaluation_or_replay(self):
        with patch("ran_monitor_ai.ui_alerts.history_summary", side_effect=sqlite3.OperationalError("read-only database")):
            self.app.run()
        self.assert_app_ok()
        self.assertTrue(any("告警数据库暂不可用" in message.value for message in self.app.error))
        self.assertTrue(any(button.key == "run_detection" for button in self.app.button))
        self.assertTrue(any(button.key == "replay_start" for button in self.app.button))
        self.assertTrue(any(input.key == "calibration_goal" for input in self.app.selectbox))

    def test_validation_calibration_can_be_applied_to_live_detection(self):
        keyed(self.app.button, "apply_calibrated_threshold").click()
        self.app.run()
        self.assert_app_ok()
        self.assertIn("custom_detection_threshold", self.app.session_state)
        self.assertIn("自定义校准", keyed(self.app.selectbox, "detection_policy").options)
        self.assertTrue(any("现在可以选择“自定义校准”" in item.value for item in self.app.success))

    def test_model_evaluation_discloses_synthetic_scope(self):
        self.assert_app_ok()
        self.assertTrue(any("合成演示模式" in item.value for item in self.app.info))
        self.assertTrue(any("AP " in item.value for item in self.app.caption))

    def test_model_evaluation_distinguishes_metadata_from_forward_test_protocol(self):
        bundle, original_metrics = self.bundle, self.metrics
        metrics = deepcopy(original_metrics)
        temporal = metrics["temporal_coverage_audit"]
        temporal.update({
            "forward_evaluation_data_eligible": True,
            "has_explicit_stream_identity": True,
            "has_explicit_event_identity": True,
            "latest_block_contains_anomaly": True,
        })
        metrics["evaluation_protocol"] = "overlap_group_stratified_holdout"
        self.assertEqual(metrics["evaluation_protocol"], "overlap_group_stratified_holdout")
        metrics["promotion_readiness"] = promotion_readiness(metrics)

        import streamlit as st
        st.cache_resource.clear()
        try:
            with patch("ran_monitor_ai.train.train", return_value=(bundle, metrics)):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0, [x.value for x in app.exception])
            self.assertTrue(any("当前模型仍采用按重叠时间组的分层留出" in item.value
                                for item in app.info), [item.value for item in app.info])
        finally:
            st.cache_resource.clear()

    def test_dashboard_runs_same_split_unsupervised_baseline_comparison(self):
        keyed(self.app.button, "run_detector_comparison").click()
        self.app.run()
        self.assert_app_ok()
        result = self.app.session_state["detector_comparison"]
        self.assertIn("xgboost_supervised", result["models"])
        self.assertIn("isolation_forest_normal_only", result["models"])
        self.assertEqual(result["split_sizes"]["test"], self.metrics["test_size"])


if __name__ == "__main__":
    unittest.main()
