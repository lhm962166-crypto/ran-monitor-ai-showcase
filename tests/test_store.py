import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from ran_monitor_ai.store import (add_note, alert_events, backup_database, connect, export_alerts, history_summary,
                                  add_feedback, feedback_rows, incident_events, query_alerts, query_incident,
                                  recent, save_batch, update_incident_status, update_status)


class TransactionalHistoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.path = Path(self.folder.name) / "nested" / "alerts.sqlite3"

    def tearDown(self):
        self.folder.cleanup()

    @staticmethod
    def result(name, *, sample_id="cell-1", severity="P1", cause="Equipment"):
        return {"sample_id": sample_id, "detected": True, "score": .9,
                "alert": {"alert_id": name, "severity": severity, "cause": cause}}

    @staticmethod
    def stream_result(name, *, minute, detected=True, request_id=None):
        start = f"2026-01-01T00:{minute:02}:00+00:00"
        end = f"2026-01-01T00:{minute + 1:02}:00+00:00"
        return {
            "sample_id": f"cell-1-{minute}", "detected": detected, "score": .9 if detected else .1,
            "request_id": request_id, "window_id": f"window-{minute}", "stream_id": "cell-1",
            "model_id": "model-v1", "start_time": start, "end_time": end,
            "alert": ({"alert_id": name, "severity": "P2", "cause": "Interference",
                       "model_id": "model-v1"} if detected else None),
        }

    def test_batch_failure_rolls_back_every_new_alert(self):
        invalid = self.result("bad", cause=None)
        with self.assertRaises(ValueError):
            save_batch([self.result("valid"), invalid], self.path)
        self.assertEqual(recent(path=self.path), [])
        self.assertEqual(history_summary(self.path)["total"], 0)

    def test_duplicate_event_is_idempotent_in_alerts_and_audit(self):
        result = self.result("one")
        self.assertEqual(save_batch([result, result], self.path), 1)
        self.assertEqual(save_batch([result], self.path), 0)
        self.assertEqual(len(alert_events("one", self.path)), 1)

    def test_filters_pagination_and_literal_wildcard_search(self):
        save_batch([
            self.result("a", sample_id="cell%1"),
            self.result("b", sample_id="cellX1", severity="P2", cause="Interference"),
            self.result("c", sample_id="cell_2", severity="P3", cause="Congestion"),
        ], self.path)
        matches, count = query_alerts(search="%", path=self.path)
        self.assertEqual(count, 1)
        self.assertEqual(matches[0]["sample_id"], "cell%1")
        self.assertEqual(query_alerts(search="%' OR 1=1 --", path=self.path)[1], 0)
        self.assertEqual(query_alerts(severities=[], path=self.path)[1], 0)
        self.assertEqual(query_alerts(severities=["P2"], causes=["Interference"], path=self.path)[1], 1)
        page1, count = query_alerts(limit=1, offset=0, path=self.path)
        page2, _ = query_alerts(limit=1, offset=1, path=self.path)
        self.assertEqual(count, 3)
        self.assertNotEqual(page1[0]["alert_id"], page2[0]["alert_id"])

    def test_disposition_notes_are_audited_and_repeated_status_does_not_duplicate(self):
        save_batch([self.result("one")], self.path)
        update_status("one", "ACKNOWLEDGED", self.path, note="已通知现场团队")
        update_status("one", "ACKNOWLEDGED", self.path, note="重复请求")
        events = alert_events("one", self.path)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["previous_status"], "OPEN")
        self.assertEqual(events[0]["note"], "已通知现场团队")
        update_status("one", "RESOLVED", self.path)
        with self.assertRaises(ValueError):
            update_status("one", "ACKNOWLEDGED", self.path)
        self.assertEqual(query_alerts(status="RESOLVED", path=self.path)[1], 1)

    def test_concurrent_batches_preserve_every_event(self):
        connect(self.path).close()
        with ThreadPoolExecutor(max_workers=4) as pool:
            saved = list(pool.map(lambda i: save_batch([self.result(f"event-{i}")], self.path), range(12)))
        self.assertEqual(sum(saved), 12)
        self.assertEqual(history_summary(self.path)["critical_unresolved"], 12)

    def test_database_path_can_be_configured_without_changing_source(self):
        with patch.dict(os.environ, {"RAN_MONITOR_DB_PATH": str(self.path)}):
            save_batch([self.result("configured")])
            self.assertEqual(recent()[0]["alert_id"], "configured")
        self.assertEqual(recent(path=self.path)[0]["alert_id"], "configured")

    def test_note_is_independent_of_status_and_requires_nonempty_text(self):
        save_batch([self.result("one")], self.path)
        self.assertTrue(add_note("one", "  已核实频谱  ", self.path))
        self.assertEqual(recent(path=self.path)[0]["status"], "OPEN")
        event = alert_events("one", self.path)[0]
        self.assertEqual(event["event_type"], "NOTE")
        self.assertEqual(event["note"], "已核实频谱")
        self.assertFalse(add_note("missing", "记录", self.path))
        with self.assertRaises(ValueError):
            add_note("one", " ", self.path)

    def test_filtered_export_crosses_pages_and_reports_truncation(self):
        save_batch([self.result(f"event-{i}", severity="P1" if i % 2 else "P2") for i in range(30)], self.path)
        rows, count = export_alerts(severities=["P1"], limit=10, path=self.path)
        self.assertEqual(len(rows), 10)
        self.assertEqual(count, 15)
        self.assertTrue(all(row["severity"] == "P1" for row in rows))
        self.assertEqual(len(export_alerts(path=self.path)[0]), 30)

    def test_corrupt_payload_errors_name_the_record_without_exposing_contents(self):
        save_batch([self.result("broken")], self.path)
        db = connect(self.path)
        db.execute("UPDATE alerts SET payload=? WHERE alert_id='broken'", ('["private contents"]',))
        db.commit()
        db.close()
        with self.assertRaisesRegex(ValueError, "broken.*详情数据无效") as error:
            query_alerts(path=self.path)
        self.assertNotIn("private contents", str(error.exception))

    def test_database_backup_preserves_alerts_and_notes_as_a_snapshot(self):
        save_batch([self.result("one")], self.path)
        add_note("one", "检查记录", self.path)
        backup = self.path.parent / "backup.sqlite3"
        backup_database(backup, self.path)
        save_batch([self.result("two")], self.path)
        self.assertEqual(query_alerts(path=backup)[1], 1)
        self.assertEqual(query_alerts(path=self.path)[1], 2)
        self.assertEqual(alert_events("one", backup)[0]["note"], "检查记录")
        with self.assertRaisesRegex(ValueError, "数据库不同"):
            backup_database(self.path, self.path)
        self.assertFalse(list(backup.parent.glob(".ran-backup-*.sqlite3")))

    def test_stream_alerts_merge_and_three_advancing_normal_windows_recover_incident(self):
        first = self.stream_result("alert-1", minute=0, request_id="request-1")
        duplicate = self.stream_result("alert-1", minute=0, request_id="request-1")
        self.assertEqual(save_batch([first], self.path), 1)
        self.assertEqual(save_batch([duplicate], self.path), 0)
        self.assertEqual(save_batch([self.stream_result("alert-2", minute=1)], self.path), 1)

        alerts, count = query_alerts(path=self.path)
        self.assertEqual(count, 2)
        incident_id = alerts[0]["incident_id"]
        self.assertEqual(alerts[1]["incident_id"], incident_id)
        self.assertEqual(query_incident(incident_id, self.path)["window_count"], 2)

        for minute in (2, 3):
            self.assertEqual(save_batch([self.stream_result(
                f"normal-{minute}", minute=minute, detected=False,
            )], self.path), 0)
            self.assertEqual(query_incident(incident_id, self.path)["status"], "OPEN")
        # A repeated window ID is idempotent and cannot complete the recovery streak.
        self.assertEqual(save_batch([self.stream_result(
            "normal-duplicate", minute=3, detected=False,
        )], self.path), 0)
        self.assertEqual(query_incident(incident_id, self.path)["normal_streak"], 2)
        self.assertEqual(save_batch([self.stream_result(
            "normal-4", minute=4, detected=False,
        )], self.path), 0)
        self.assertEqual(query_incident(incident_id, self.path)["status"], "RECOVERED")
        timeline = incident_events(incident_id, self.path)
        self.assertEqual(timeline[0]["event_type"], "RECOVERY_CONFIRMED")
        self.assertEqual(timeline[0]["payload"]["consecutive_normal_windows"], 3)

    def test_windows_without_explicit_stream_identity_remain_separate_incidents(self):
        first = self.result("unlinked-1")
        second = self.result("unlinked-2")
        first["start_time"] = second["start_time"] = "2026-01-01T00:00:00Z"
        first["end_time"] = second["end_time"] = "2026-01-01T00:01:00Z"
        self.assertEqual(save_batch([first, second], self.path), 2)
        alerts, _ = query_alerts(path=self.path)
        self.assertNotEqual(alerts[0]["incident_id"], alerts[1]["incident_id"])

    def test_feedback_export_links_human_label_to_window_and_keeps_alert_state(self):
        result = self.stream_result("reviewed-alert", minute=0)
        self.assertEqual(save_batch([result], self.path), 1)
        self.assertTrue(add_feedback("reviewed-alert", "Interference", "high", "reviewer-1",
                                     "频谱记录确认外部干扰", self.path))
        rows = feedback_rows(self.path)
        self.assertEqual(rows[0]["window_id"], "window-0")
        self.assertEqual(rows[0]["predicted_cause"], "Interference")
        self.assertEqual(rows[0]["reviewer"], "reviewer-1")
        self.assertEqual(recent(path=self.path)[0]["status"], "OPEN")
        self.assertFalse(add_feedback("missing", "Equipment", "medium", "reviewer-1", "核实依据", self.path))


if __name__ == "__main__":
    unittest.main()
