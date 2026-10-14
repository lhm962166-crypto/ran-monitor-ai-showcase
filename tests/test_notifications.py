import unittest
from unittest.mock import patch

from ran_monitor_ai.notify import deliver_digest, notification_digest, send_email, send_whatsapp


def result(index, level="P1"):
    return {"sample_id": index, "detected": True, "score": .9,
            "alert": {"alert_id": f"ALT-{index}", "severity": level, "cause_zh": "设备故障",
                      "action": "检查设备", "team": "运维"}}


class DigestNotificationTests(unittest.TestCase):
    def test_digest_filters_levels_and_limits_details(self):
        results = [result(index) for index in range(30)] + [result(100, "P4")]
        title, body, short = notification_digest(results, {"P1"})
        self.assertIn("30 条", title)
        self.assertIn("另有 10 条", body)
        self.assertNotIn("ALT-100", body)
        self.assertLess(len(short), 1200)

    def test_batch_sends_once_per_selected_channel_without_network(self):
        with patch("ran_monitor_ai.notify._send_email_message") as email, patch("ran_monitor_ai.notify._send_whatsapp_body") as whatsapp:
            outcome = deliver_digest([result(index) for index in range(100)], {"email", "whatsapp"}, {"P1"})
            self.assertEqual(email.call_count, 1)
            self.assertEqual(whatsapp.call_count, 1)
            self.assertTrue(all(success for _, success, _ in outcome))

    def test_direct_notifications_disclose_uncalibrated_cause_probability(self):
        alert = result(7)["alert"] | {"confidence": .73}
        with patch("ran_monitor_ai.notify._send_email_message") as email, patch(
                "ran_monitor_ai.notify._send_whatsapp_body") as whatsapp:
            send_email(alert, .91)
            send_whatsapp(alert, .91)
        self.assertIn("模型输出概率，未校准", email.call_args.args[1])
        self.assertIn("模型输出概率（未校准）", whatsapp.call_args.args[0])

    def test_empty_selection_sends_nothing_and_channel_failures_are_independent(self):
        with patch("ran_monitor_ai.notify._send_email_message", side_effect=OSError("offline")) as email, patch("ran_monitor_ai.notify._send_whatsapp_body") as whatsapp:
            self.assertEqual(deliver_digest([result(1, "P4")], {"email"}, {"P1"}), [])
            email.assert_not_called()
            outcome = deliver_digest([result(1)], {"email", "whatsapp"}, {"P1"})
            self.assertFalse(outcome[0][1])
            self.assertTrue(outcome[1][1])
            whatsapp.assert_called_once()


if __name__ == "__main__":
    unittest.main()
