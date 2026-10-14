"""Sustained local inference/storage checks against an isolated test database.

No notification code is imported. This does not write to the user's alert history.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.cli import atomic_output
from ran_monitor_ai.engine import score_records
from ran_monitor_ai.diagnostics import process_memory_mb
from ran_monitor_ai.store import add_note, alert_events, history_summary, query_alerts, save_batch, update_status
from ran_monitor_ai.train import load_records, train


def main():
    parser = argparse.ArgumentParser(description="持续运行检查（使用独立测试数据库）")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "soak-latest.json")
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval < 0:
        parser.error("seconds 须大于零，interval 不能为负。")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    database = ROOT / "artifacts" / f"soak-{run_id}.sqlite3"
    bundle, _ = train()
    records = load_records()
    started = time.monotonic()
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "status": "running", "rounds": 0,
              "windows": 0, "alerts": 0, "failures": [], "database": str(database),
              "model_id": bundle.get("metadata", {}).get("model_id")}
    durations = []
    expected = {}
    tracemalloc.start()
    next_progress = started
    try:
        while time.monotonic() - started < args.seconds:
            round_started = time.monotonic()
            offset = (report["rounds"] * 79) % len(records)
            batch = [records[(offset + i) % len(records)] for i in range(100)]
            results = score_records(batch, bundle)
            durations.append(time.monotonic() - round_started)
            for record, result in zip(batch, results):
                key = record["sample_id"]
                signature = (result["score"], result["detected"], (result.get("alert") or {}).get("cause"))
                if key in expected and expected[key] != signature:
                    raise AssertionError(f"窗口 {key} 重复检测出现不一致")
                expected[key] = signature
            saved = save_batch(results, database)
            if saved != sum(result["detected"] for result in results):
                raise AssertionError("批量保存数量与异常数量不一致")
            if save_batch(results, database) != 0:
                raise AssertionError("重复保存产生额外记录")
            report["alerts"] += saved
            report["rounds"] += 1
            report["windows"] += len(batch)
            abnormal = next((result for result in results if result["detected"]), None)
            if abnormal:
                alert_id = abnormal["alert"]["alert_id"]
                update_status(alert_id, "ACKNOWLEDGED", database, note="持续运行检查：确认")
                add_note(alert_id, "持续运行检查：排查记录", database)
                update_status(alert_id, "RESOLVED", database, note="持续运行检查：解决")
                if len(alert_events(alert_id, database)) != 4:
                    raise AssertionError("告警处置记录数量不正确")
            summary = history_summary(database)
            if summary["total"] != report["alerts"]:
                raise AssertionError("历史告警数量与累计保存数量不一致")
            _, count = query_alerts(limit=1, path=database)
            if count != report["alerts"]:
                raise AssertionError("分页统计遗漏告警")
            current, peak = tracemalloc.get_traced_memory()
            report.update(elapsed_seconds=round(time.monotonic() - started, 2),
                          median_batch_seconds=round(statistics.median(durations), 4),
                          max_batch_seconds=round(max(durations), 4),
                          python_memory_mb=round(current / 1024**2, 2),
                          python_peak_mb=round(peak / 1024**2, 2))
            report.update(process_memory_mb())
            atomic_output(args.output, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
            if time.monotonic() >= next_progress:
                print(json.dumps({key: report[key] for key in ("rounds", "windows", "alerts", "elapsed_seconds", "python_memory_mb")}), flush=True)
                next_progress = time.monotonic() + 60
            remaining = args.seconds - (time.monotonic() - started)
            time.sleep(max(0, min(args.interval - (time.monotonic() - round_started), remaining)))
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        raise
    except Exception as error:
        report["status"] = "failed"
        report["failures"].append(f"{type(error).__name__}: {error}")
        raise
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 2)
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_output(args.output, json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
        tracemalloc.stop()
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
