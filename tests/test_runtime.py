import os
import unittest
from unittest.mock import patch

from ran_monitor_ai.runtime import configure_threads


class NativeThreadDefaultsTests(unittest.TestCase):
    def test_defaults_are_bounded_and_existing_preferences_are_preserved(self):
        with patch.dict(os.environ, {"OPENBLAS_NUM_THREADS": "6"}, clear=True), patch("os.cpu_count", return_value=24):
            settings = configure_threads(4)
            self.assertEqual(settings["OPENBLAS_NUM_THREADS"], "6")
            self.assertEqual(settings["OMP_NUM_THREADS"], "4")
        with patch.dict(os.environ, {}, clear=True), patch("os.cpu_count", return_value=2):
            self.assertEqual(configure_threads()["OMP_NUM_THREADS"], "2")

    def test_invalid_limits_are_rejected(self):
        for limit in (0, 33, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                configure_threads(limit)


if __name__ == "__main__":
    unittest.main()
