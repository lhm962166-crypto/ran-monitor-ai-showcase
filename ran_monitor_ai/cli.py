"""Local batch detection and evaluation without the Streamlit server."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import subprocess
from pathlib import Path
from time import perf_counter

from .runtime import configure_threads
configure_threads()

import pandas as pd

from .engine import bundle_for_policy, score_records
from .configuration import load_env_file
from .drift_benchmark import benchmark_score_drift
from .exporting import dataframe_csv_bytes
from .features import parse_json_payload, validate_batch
from .governance import promotion_readiness
from .model_benchmark import compare_with_isolation_forest
from .reporting import detection_report_html
from .store import backup_database, save_batch
from .train import ROOT, load_records, train


def atomic_output(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".ran-export-", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def result_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {"窗口编号": result.get("sample_id"), "判定": "异常" if result["detected"] else "正常",
         "评分": result["score"], "阈值": result["threshold"],
         "级别": (result.get("alert") or {}).get("severity", "—"),
         "原因": (result.get("alert") or {}).get("cause_zh", "—"),
         "需复核": bool((result.get("alert") or {}).get("requires_review", False))} for result in results
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAN Monitor AI 本地批量检测工具")
    commands = parser.add_subparsers(dest="command", required=True)
    detect = commands.add_parser("detect", help="检测单个窗口或 JSON 窗口数组")
    detect.add_argument("input", type=Path, help="UTF-8 JSON 文件（最多 10 MB / 100 窗口）")
    detect.add_argument("--output", type=Path, help="输出文件；不填写时打印结果")
    detect.add_argument("--format", choices=["json", "csv", "html"], default="json")
    detect.add_argument("--save-alerts", action="store_true", help="保存异常到本机数据库")
    detect.add_argument("--db", type=Path, help="可选告警数据库路径")
    detect.add_argument("--policy", choices=["balanced", "sensitive", "precision"], default="balanced",
                        help="验证集校准的检测策略：均衡 / 高召回 / 高精确")
    evaluation = commands.add_parser("evaluate", help="导出当前模型评估指标")
    evaluation.add_argument("--output", type=Path)
    drift = commands.add_parser("drift-benchmark", help="用固定种子评测分数漂移监测的误报与发现延迟")
    drift.add_argument("--runs", type=int, default=200, help="每种场景重复次数（20–5000）")
    drift.add_argument("--horizon", type=int, default=200, help="每条模拟流的窗口数（60–10000）")
    drift.add_argument("--batch-size", type=int, default=10, help="每次在线更新的窗口数（1–100）")
    drift.add_argument("--seed", type=int, default=42, help="可复现实验随机种子")
    drift.add_argument("--output", type=Path, help="JSON 结果路径；不填写时打印到终端")
    comparison = commands.add_parser("compare-detectors", help="同一数据切分上比较监督模型与无监督基线")
    comparison.add_argument("--iforest-trees", type=int, default=300, help="Isolation Forest 树数（50–2000）")
    comparison.add_argument("--target-recall", type=float, default=0.90, help="验证集校准目标召回率")
    comparison.add_argument("--seed", type=int, default=42, help="可复现实验随机种子")
    comparison.add_argument("--output", type=Path, help="JSON 结果路径；不填写时打印到终端")
    promotion = commands.add_parser("promotion-check", help="评估模型是否具备晋级评审所需证据")
    promotion.add_argument("--min-test-windows", type=int, default=500, help="最低测试窗口数")
    promotion.add_argument("--required-origins", default="synthetic,controlled_real",
                           help="必须覆盖的数据来源，逗号分隔")
    promotion.add_argument("--output", type=Path, help="JSON 结果路径；暂缓时仍保存报告并返回退出码 2")
    backup = commands.add_parser("backup", help="备份告警数据库及处置记录")
    backup.add_argument("--output", type=Path, required=True)
    backup.add_argument("--db", type=Path, help="源数据库路径；默认按本地配置读取")
    demo = commands.add_parser("demo", help="启动仅使用合成数据的本机交互演示")
    demo.add_argument("--port", type=int, default=8504, help="本机服务端口（1024–65535）")
    args = parser.parse_args(argv)
    exit_code = 0
    try:
        if args.command == "demo":
            if not 1024 <= args.port <= 65535:
                raise ValueError("演示端口须在 1024–65535 范围内。")
            from .demo import write_synthetic_dataset

            runtime = Path(tempfile.mkdtemp(prefix="ran-monitor-ai-demo-"))
            data_path = write_synthetic_dataset(runtime / "data" / "windows.jsonl")
            environment = os.environ.copy()
            for name in ("SMTP_USER", "SMTP_PASSWORD", "SMTP_HOST", "SMTP_PORT", "ALERT_EMAIL_TO",
                         "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM",
                         "TWILIO_WHATSAPP_TO"):
                environment.pop(name, None)
            environment.update({
                "RAN_MONITOR_DEMO": "1",
                "RAN_MONITOR_DATA_PATH": str(data_path),
                "RAN_MONITOR_ARTIFACT_PATH": str(runtime / "artifacts" / "model.joblib"),
                "RAN_MONITOR_DB_PATH": str(runtime / "alerts.sqlite3"),
            })
            print(f"合成数据和告警数据库保存在临时目录：{runtime}", file=sys.stderr)
            command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                       "--server.address", "127.0.0.1", "--server.port", str(args.port),
                       "--server.headless", "true", "--browser.gatherUsageStats", "false"]
            return subprocess.call(command, cwd=ROOT, env=environment)
        load_env_file(ROOT / ".env")
        if args.command == "backup":
            target = backup_database(args.output, args.db)
            print(f"备份完成：{target}", file=sys.stderr)
            return 0
        if args.command == "evaluate":
            bundle, metrics = train()
            payload = json.dumps(metrics | {"model_metadata": bundle.get("metadata", {})},
                                 ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        elif args.command == "drift-benchmark":
            bundle, _ = train()
            payload = json.dumps(benchmark_score_drift(
                bundle["calibration"]["scores"],
                model_id=bundle.get("metadata", {}).get("model_id", "unknown"),
                replicates=args.runs, horizon=args.horizon,
                batch_size=args.batch_size, seed=args.seed,
            ), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        elif args.command == "compare-detectors":
            bundle, _ = train()
            payload = json.dumps(compare_with_isolation_forest(
                load_records(), bundle, target_recall=args.target_recall,
                n_estimators=args.iforest_trees, seed=args.seed,
            ), ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        elif args.command == "promotion-check":
            _, metrics = train()
            decision = promotion_readiness(
                metrics, required_origins=args.required_origins,
                min_test_windows=args.min_test_windows,
            )
            payload = json.dumps(decision, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            exit_code = 0 if decision["promotion_allowed"] else 2
        else:
            if args.output and args.output.resolve() == args.input.resolve():
                raise ValueError("输出文件须与输入文件不同，以保留原始窗口。")
            if args.db and not args.save_alerts:
                raise ValueError("指定 --db 时请同时使用 --save-alerts。")
            if args.input.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("输入文件超过 10 MB 限制。")
            records = validate_batch(parse_json_payload(args.input.read_bytes()), max_records=100)
            bundle, _ = train()
            bundle = bundle_for_policy(bundle, args.policy)
            started = perf_counter()
            results = score_records(records, bundle)
            elapsed = perf_counter() - started
            if args.format == "html":
                payload = detection_report_html(results, metadata={"source": args.input.name},
                                                model_metadata=bundle.get("metadata"))
            elif args.format == "csv":
                payload = dataframe_csv_bytes(result_table(results))
            else:
                payload = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
            if args.save_alerts:
                saved = save_batch(results, args.db)
                print(f"已保存 {saved} 条新告警。", file=sys.stderr)
            print(f"检测完成：{len(records)} 个窗口，{sum(result['detected'] for result in results)} 个异常，{elapsed:.3f} 秒。",
                  file=sys.stderr)
        if args.output:
            atomic_output(args.output, payload)
            print(f"结果：{args.output.resolve()}", file=sys.stderr)
        else:
            sys.stdout.write(payload.decode("utf-8-sig") + "\n")
        return exit_code
    except (ValueError, OSError, sqlite3.Error) as error:
        print(f"处理失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
