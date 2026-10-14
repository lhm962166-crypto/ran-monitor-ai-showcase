"""Verify that fixed data, code, and seeds reproduce the model snapshot."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.cli import atomic_output
from ran_monitor_ai.engine import score_records
from ran_monitor_ai.train import load_records, train


def _canonical_digest(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _prediction_signature(results: list[dict]) -> list[dict]:
    return [{
        "sample_id": row.get("sample_id"),
        "score": row["score"],
        "detected": row["detected"],
        "cause": (row.get("alert") or {}).get("cause"),
        "cause_probabilities": (row.get("alert") or {}).get("cause_probabilities"),
    } for row in results]


def main() -> int:
    records = load_records()
    stride = max(1, len(records) // 20)
    probes = records[::stride][:20]

    started = perf_counter()
    first_bundle, first_metrics = train(force=True)
    first_signature = _prediction_signature(score_records(probes, first_bundle))
    second_bundle, second_metrics = train(force=True)
    second_signature = _prediction_signature(score_records(probes, second_bundle))
    elapsed = round(perf_counter() - started, 3)

    first_digest = _canonical_digest(first_metrics)
    second_digest = _canonical_digest(second_metrics)
    same_model_id = (first_bundle.get("metadata", {}).get("model_id")
                     == second_bundle.get("metadata", {}).get("model_id"))
    same_metrics = first_digest == second_digest
    same_predictions = first_signature == second_signature
    report = {
        "status": "passed" if same_model_id and same_metrics and same_predictions else "failed",
        "model_id": second_bundle.get("metadata", {}).get("model_id"),
        "probe_windows": len(probes),
        "same_model_id_after_rebuild": same_model_id,
        "same_metrics_after_rebuild": same_metrics,
        "metrics_sha256": second_digest,
        "same_probe_predictions_after_rebuild": same_predictions,
        "elapsed_seconds": elapsed,
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "xgboost")},
        },
        "scope_note": "固定数据、代码和库版本下的本地确定性检查；不证明不同硬件或依赖版本间逐位一致。",
    }
    output = ROOT / "artifacts" / "reproducibility-latest.json"
    atomic_output(output, json.dumps(report, ensure_ascii=False, indent=2,
                                     allow_nan=False).encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
