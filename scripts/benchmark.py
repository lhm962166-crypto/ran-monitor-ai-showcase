"""Reproducible local latency and feature-equivalence benchmark."""
from __future__ import annotations

import json
import argparse
import importlib.metadata
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ran_monitor_ai.runtime import configure_threads
configure_threads()

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from ran_monitor_ai.cli import atomic_output
from ran_monitor_ai.diagnostics import process_memory_mb
from ran_monitor_ai.engine import score_records
from ran_monitor_ai.features import extract_features
from ran_monitor_ai.train import load_records, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="可复现的特征与批量推理性能基准")
    parser.add_argument("--batch-size", type=int, default=100, help="每批窗口数（1–100）")
    parser.add_argument("--warmups", type=int, default=3, help="预热次数（0–20）")
    parser.add_argument("--repeats", type=int, default=20, help="计时重复次数（3–100）")
    parser.add_argument("--seed", type=int, default=42, help="可复现抽样种子")
    args = parser.parse_args(argv)
    if not 1 <= args.batch_size <= 100:
        parser.error("batch-size 须为 1–100。")
    if not 0 <= args.warmups <= 20:
        parser.error("warmups 须为 0–20。")
    if not 3 <= args.repeats <= 100:
        parser.error("repeats 须为 3–100。")
    records = load_records()
    bundle, _ = train()
    feature_durations = []
    features = None
    for _ in range(max(1, args.warmups)):
        started = perf_counter()
        features = pd.DataFrame([extract_features(record) for record in records])
        feature_durations.append(perf_counter() - started)
    feature_seconds = statistics.median(feature_durations)
    indices = np.random.default_rng(args.seed).choice(
        len(records), size=min(args.batch_size, len(records)), replace=False)
    batch = [records[index] for index in indices]
    for _ in range(args.warmups):
        score_records(batch, bundle)
    durations = []
    for _ in range(args.repeats):
        started = perf_counter()
        score_records(batch, bundle)
        durations.append(perf_counter() - started)
    batch_median = statistics.median(durations)
    report = {"feature_windows": len(records), "feature_seconds": feature_seconds,
              "feature_repeats": len(feature_durations),
              "batch_windows": len(batch), "batch_median_seconds": batch_median,
              "batch_min_seconds": min(durations), "batch_max_seconds": max(durations),
              "batch_p95_seconds": float(np.percentile(durations, 95)),
              "batch_windows_per_second": len(batch) / batch_median,
              "batch_median_ms_per_window": batch_median * 1000 / len(batch),
              "warmups": args.warmups, "runs": len(durations), "seed": args.seed,
              "model_id": bundle["metadata"]["model_id"],
              "environment": {
                  "os": platform.platform(), "python": platform.python_version(),
                  "machine": platform.machine(),
                  "processor": platform.processor() or platform.machine(),
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
              },
              "memory": process_memory_mb(),
              "thread_pools": [{"api": pool["user_api"], "threads": pool["num_threads"]} for pool in threadpool_info()]}
    baseline_path = ROOT / "artifacts" / "performance_baseline.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if len(baseline.get("features", [])) == len(records):
            report["baseline_snapshot_note"] = (
                "Saved pre-vectorization one-shot timing; speedup is directional because the old snapshot "
                "does not contain repeated measurements or machine metadata."
            )
            old = pd.DataFrame(baseline["features"]).reindex(columns=features.columns)
            report["maximum_feature_difference"] = float(np.max(np.abs(features.to_numpy() - old.to_numpy())))
            report["features_equivalent"] = bool(np.allclose(features, old, rtol=1e-10, atol=1e-9))
            original_predictions = score_records(records, bundle, feature_rows=old)
            current_predictions = score_records(records, bundle, feature_rows=features)
            report["changed_decisions"] = sum(
                (left["detected"], (left.get("alert") or {}).get("cause"))
                != (right["detected"], (right.get("alert") or {}).get("cause"))
                for left, right in zip(original_predictions, current_predictions)
            )
            report["baseline_feature_seconds"] = baseline["feature_seconds"]
            report["feature_speedup"] = baseline["feature_seconds"] / feature_seconds
            if baseline.get("score_windows") == len(batch) and baseline.get("score_seconds", 0) > 0:
                report["baseline_batch_seconds"] = baseline["score_seconds"]
                report["batch_speedup"] = baseline["score_seconds"] / batch_median
    atomic_output(ROOT / "artifacts" / "benchmark-latest.json", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
