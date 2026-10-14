import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ran_monitor_ai.cli import main
from ran_monitor_ai.configuration import load_env_file, read_env
from ran_monitor_ai.store import query_alerts
from tests.demo_fixture import get_demo_fixture


class LocalToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        get_demo_fixture()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        source = Path(__file__).resolve().parents[1] / "sample_window.json"
        self.input = self.root / "window.json"
        self.input.write_bytes(source.read_bytes())
        self.db = self.root / "isolated.sqlite3"
        self.environment = patch.dict(os.environ, {"RAN_MONITOR_DB_PATH": str(self.db)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.folder.cleanup()

    def run_cli(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_detection_outputs_a_json_array_and_does_not_save_by_default(self):
        code, output, _ = self.run_cli(["detect", str(self.input)])
        self.assertEqual(code, 0)
        results = json.loads(output)
        self.assertEqual(len(results), 1)
        self.assertIn("detected", results[0])
        self.assertEqual(query_alerts(path=self.db)[1], 0)

    def test_output_formats_and_input_preservation(self):
        previous = self.input.read_bytes()
        for kind in ("csv", "html"):
            target = self.root / f"result.{kind}"
            code, _, _ = self.run_cli(["detect", str(self.input), "--format", kind, "--output", str(target)])
            self.assertEqual(code, 0)
            self.assertGreater(target.stat().st_size, 0)
        code, _, message = self.run_cli(["detect", str(self.input), "--output", str(self.input)])
        self.assertEqual(code, 1)
        self.assertIn("保留原始窗口", message)
        self.assertEqual(self.input.read_bytes(), previous)

    def test_invalid_input_returns_nonzero_without_output_or_alerts(self):
        self.input.write_text('{"KPIs": {}}', encoding="utf-8")
        target = self.root / "result.json"
        code, _, message = self.run_cli(["detect", str(self.input), "--output", str(target), "--save-alerts"])
        self.assertEqual(code, 1)
        self.assertIn("RSRP", message)
        self.assertFalse(target.exists())
        self.assertEqual(query_alerts(path=self.db)[1], 0)

    def test_demo_launch_uses_isolated_synthetic_runtime_and_loopback(self):
        runtime = self.root / "demo-runtime"
        runtime.mkdir()
        with patch.dict(os.environ, {"SMTP_USER": "ignored@example.invalid", "TWILIO_AUTH_TOKEN": "ignored"}), \
                patch("ran_monitor_ai.cli.tempfile.mkdtemp", return_value=str(runtime)), \
                patch("ran_monitor_ai.cli.subprocess.call", return_value=0) as launch, \
                patch("ran_monitor_ai.cli.load_env_file") as load_env:
            code, _, _ = self.run_cli(["demo", "--port", "8509"])
        self.assertEqual(code, 0)
        load_env.assert_not_called()
        command = launch.call_args.args[0]
        environment = launch.call_args.kwargs["env"]
        self.assertIn("127.0.0.1", command)
        self.assertIn("8509", command)
        self.assertIn("--server.headless", command)
        self.assertIn("--browser.gatherUsageStats", command)
        self.assertEqual(environment["RAN_MONITOR_DEMO"], "1")
        self.assertTrue(Path(environment["RAN_MONITOR_DATA_PATH"]).is_file())
        self.assertTrue(environment["RAN_MONITOR_ARTIFACT_PATH"].startswith(str(runtime)))
        self.assertTrue(environment["RAN_MONITOR_DB_PATH"].startswith(str(runtime)))
        self.assertNotIn("SMTP_USER", environment)
        self.assertNotIn("TWILIO_AUTH_TOKEN", environment)

    def test_demo_rejects_nonlocal_unusable_port_before_launch(self):
        with patch("ran_monitor_ai.cli.subprocess.call") as launch:
            code, _, message = self.run_cli(["demo", "--port", "80"])
        self.assertEqual(code, 1)
        self.assertIn("1024–65535", message)
        launch.assert_not_called()

    def test_env_keeps_hash_characters_and_quoted_windows_paths(self):
        target = self.root / ".env"
        target.write_text('PASSWORD="private#value" # comment\nDB_PATH="C:\\local\\alerts.sqlite3"\nVALUE=abc#def\nOTHER=plain # comment\n', encoding="utf-8")
        values = read_env(target)
        self.assertEqual(values["PASSWORD"], "private#value")
        self.assertEqual(values["DB_PATH"], "C:\\local\\alerts.sqlite3")
        self.assertEqual(values["VALUE"], "abc#def")
        self.assertEqual(values["OTHER"], "plain")
        with patch.dict(os.environ, {"PASSWORD": "existing"}):
            load_env_file(target)
            self.assertEqual(os.environ["PASSWORD"], "existing")

    def test_invalid_config_errors_do_not_expose_values(self):
        target = self.root / ".env"
        target.write_text('PASSWORD="do-not-disclose', encoding="utf-8")
        with self.assertRaises(ValueError) as error:
            read_env(target)
        self.assertNotIn("do-not-disclose", str(error.exception))

    def test_drift_benchmark_cli_returns_reproducible_protocol_and_scenarios(self):
        code, output, _ = self.run_cli([
            "drift-benchmark", "--runs", "20", "--horizon", "60",
            "--batch-size", "10", "--seed", "11",
        ])
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(report["seed"], 11)
        self.assertEqual(report["replicates_per_scenario"], 20)
        self.assertEqual(set(report["scenarios"]), {"stable", "mean_shift", "spread_shift", "surge"})
        self.assertIn("do not measure RAN field accuracy", report["scope_note"])

    def test_detector_comparison_uses_an_unsupervised_normal_only_baseline(self):
        code, output, _ = self.run_cli([
            "compare-detectors", "--iforest-trees", "50", "--target-recall", "0.9", "--seed", "13",
        ])
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertEqual(set(report["models"]), {"xgboost_supervised", "isolation_forest_normal_only"})
        forest = report["models"]["isolation_forest_normal_only"]
        self.assertGreater(forest["fit_normal_windows"], 0)
        self.assertIn("no fault labels used during fit", forest["fit_scope"])
        self.assertEqual(forest["calibration"]["target_recall"], 0.9)
        self.assertEqual(report["target_validation_recall"], 0.9)
        self.assertIn("group_bootstrap_95ci", forest["test"])
        self.assertIn("average_precision", forest["test"])

    def test_promotion_check_returns_hold_code_with_explainable_reasons(self):
        code, output, _ = self.run_cli(["promotion-check"])
        self.assertEqual(code, 2)
        report = json.loads(output)
        self.assertFalse(report["promotion_allowed"])
        self.assertEqual(report["status"], "hold")
        checks = {row["key"]: row["passed"] for row in report["checks"]}
        self.assertFalse(checks["origin_coverage"], "synthetic-only data must not satisfy controlled-real evidence")
        self.assertFalse(checks["test_sample_size"])
        self.assertTrue(checks["stream_and_event_identity"])
        self.assertTrue(checks["forward_time_evaluation"])

    def test_evaluate_exports_probability_calibration_diagnostics(self):
        code, output, _ = self.run_cli(["evaluate"])
        self.assertEqual(code, 0)
        report = json.loads(output)
        self.assertIn("brier_score", report["detector_probability_diagnostics"])
        self.assertIn("top_label_ece", report["detector_probability_diagnostics"])
        self.assertIn("brier_score", report["cause_probability_diagnostics"])
        self.assertGreaterEqual(report["test_average_precision"], 0.0)
        self.assertIn("average_precision", report["test_metric_uncertainty"]["metrics"])
        self.assertIn("promotion_readiness", report)


if __name__ == "__main__":
    unittest.main()
