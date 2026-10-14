import unittest
from unittest.mock import patch

from ran_monitor_ai.replay import ReplayRun, replay_windows


class ReplayStateTests(unittest.TestCase):
    def setUp(self):
        self.records = [{"sample_id": index, "anomalies": {"exists": index > 0, "type": "Jamming"}}
                        for index in range(4)]

    def test_mixed_queue_and_cause_filter_use_existing_records_only(self):
        queue = replay_windows(self.records, limit=3)
        self.assertEqual(len(queue), 3)
        self.assertEqual(queue[0]["sample_id"], 0)
        self.assertTrue(all(record["sample_id"] > 0 for record in replay_windows(self.records, cause="Interference")))

    def test_timer_pause_resume_and_completion_do_not_duplicate_windows(self):
        replay = ReplayRun(self.records, interval=2)
        with patch("ran_monitor_ai.replay.score_record", side_effect=lambda record, _: {"sample_id": record["sample_id"]}):
            self.assertIsNone(replay.advance({}, now=0))
            replay.resume()
            self.assertEqual(replay.advance({}, now=10)["sample_id"], 0)
            self.assertIsNone(replay.advance({}, now=11))
            replay.pause()
            self.assertIsNone(replay.advance({}, now=12))
            replay.resume()
            self.assertEqual(replay.advance({}, now=20)["sample_id"], 1)
            replay.advance({}, now=22)
            replay.advance({}, now=24)
            self.assertEqual(replay.state, "completed")
            self.assertIsNone(replay.advance({}, now=26))
            self.assertEqual([result["sample_id"] for result in replay.results], [0, 1, 2, 3])

    def test_failed_inference_stops_replay_without_advancing_cursor(self):
        replay = ReplayRun(self.records)
        replay.resume()
        with patch("ran_monitor_ai.replay.score_record", side_effect=ValueError("bad window")):
            with self.assertRaises(ValueError):
                replay.advance({}, now=0)
        self.assertEqual(replay.state, "failed")
        self.assertEqual(replay.cursor, 0)
        self.assertEqual(replay.results, [])


if __name__ == "__main__":
    unittest.main()
