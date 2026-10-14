"""Join explicitly reviewed alert causes to exact source KPI windows."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.curation import compile_reviewed_cause_records


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"源窗口文件为空：{path}")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True,
                        help="保留原始 KPI 的 JSONL；须与告警窗口身份一致")
    parser.add_argument("--feedback-json", type=Path, required=True,
                        help="告警中心导出的人工反馈 JSON 数组")
    parser.add_argument("--output", type=Path, required=True,
                        help="输出 reviewed-cause JSONL，不会被训练流程自动载入")
    parser.add_argument("--report", type=Path,
                        help="可选的编译审计报告 JSON 路径")
    args = parser.parse_args(argv)
    reviews = json.loads(args.feedback_json.read_text(encoding="utf-8"))
    if isinstance(reviews, dict):
        reviews = reviews.get("feedback")
    compiled, report = compile_reviewed_cause_records(read_jsonl(args.source_jsonl), reviews)
    if not compiled:
        raise ValueError("没有可关联的高/中可信度人工复核窗口；未创建输出文件。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in compiled:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
                               encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
