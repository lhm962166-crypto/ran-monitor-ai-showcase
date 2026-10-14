import json
import unittest
from pathlib import Path

from ran_monitor_ai.charts import kpi_figure
from ran_monitor_ai.features import extract_features, parse_json_payload, validate_batch
from ran_monitor_ai.reporting import detection_report_html

ROOT = Path(__file__).resolve().parents[1]


class InputAndExportTests(unittest.TestCase):
    def setUp(self):
        self.record = json.loads((ROOT / "sample_window.json").read_text(encoding="utf-8"))

    def test_nonstandard_numbers_and_duplicate_json_fields_are_rejected(self):
        for raw in ('{"score": NaN}', '{"score": Infinity}', '{"score": -Infinity}',
                    '{"KPIs": {}, "KPIs": {}}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_json_payload(raw)

    def test_boolean_kpis_and_structured_sample_ids_are_rejected(self):
        self.record["KPIs"]["RSRP"][0] = True
        with self.assertRaisesRegex(ValueError, "布尔值"):
            validate_batch(self.record)
        self.record["KPIs"]["RSRP"][0] = -72
        self.record["sample_id"] = {"unexpected": "object"}
        with self.assertRaisesRegex(ValueError, "sample_id"):
            validate_batch(self.record)

    def test_finite_but_overflowing_inputs_have_an_actionable_error(self):
        self.record["KPIs"]["RSRP"] = [1e308] * 128
        with self.assertRaisesRegex(ValueError, "数值范围过大"):
            extract_features(self.record)

    def test_offline_report_escapes_untrusted_text_and_needs_no_external_assets(self):
        attack = '<script>alert("x")</script>'
        alert = {"severity": "P1", "cause_zh": attack, "confidence": .9, "alert_id": "ALT-safe",
                 "action": attack, "team": "测试", "requires_review": True, "affected_kpis": [],
                 "cause_probabilities": {"Equipment": .9, "Interference": .04,
                                          "Congestion": .03, "Mobility": .03},
                 "cause_ranking_shadow": {"candidates": [{
                     "rank": 1, "cause": "Equipment", "classifier_probability": .9,
                     "kpi_prototype_distance": .25, "ranking_score": .75,
                 }]}}
        result = {"sample_id": attack, "detected": True, "score": .95, "threshold": .7, "alert": alert}
        document = detection_report_html([result], metadata={"source": attack}).decode("utf-8")
        self.assertNotIn("<script>", document)
        self.assertIn("&lt;script&gt;", document)
        self.assertIn("根因分类存在不确定性", document)
        self.assertIn("原因原始输出（未校准）", document)
        self.assertIn("原始未校准输出", document)
        self.assertIn("KPI 原型辅助排序（影子实验）", document)
        self.assertIn("实验排序分数（非概率）", document)
        self.assertNotIn("http://", document)
        self.assertNotIn("https://", document)
        self.assertIn("0.9500", document)

    def test_normal_report_and_chart_preserve_values(self):
        document = detection_report_html([{"sample_id": 1, "detected": False, "score": .1,
                                           "threshold": .7, "alert": None}]).decode("utf-8")
        self.assertIn("正常", document)
        values = [1., 3., 2.]
        figure = kpi_figure("RSRP", values, {"mean": 2, "std": .5}, comparison=[2., 2., 2.])
        self.assertEqual(list(figure.data[-1].y), values)
        self.assertEqual(len(figure.data), 2)
        self.assertEqual(figure.layout.shapes[0].y0, .5)
        self.assertEqual(figure.layout.shapes[0].y1, 3.5)


if __name__ == "__main__":
    unittest.main()
