"""Evidence-based checks for whether a model is ready for promotion review."""
from __future__ import annotations

import math


def _valid_unit_interval(interval) -> bool:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return False
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in interval):
        return False
    try:
        lower, upper = (float(value) for value in interval)
    except (OverflowError, ValueError):
        return False
    if not math.isfinite(lower) or not math.isfinite(upper):
        return False
    return 0 <= lower <= upper <= 1


def promotion_readiness(metrics: dict, *, required_origins=("synthetic", "controlled_real"),
                        min_test_windows: int = 500) -> dict:
    """Hold promotion when evaluation evidence is too small or provenance is missing."""
    if not isinstance(metrics, dict):
        raise ValueError("晋级门禁需要模型评估指标对象。")
    if (not isinstance(min_test_windows, int) or isinstance(min_test_windows, bool)
            or min_test_windows < 1):
        raise ValueError("晋级门禁最小测试窗口数须为正整数。")
    if isinstance(required_origins, str):
        required_origins = tuple(item.strip() for item in required_origins.split(",") if item.strip())
    else:
        required_origins = tuple(str(item).strip() for item in required_origins if str(item).strip())

    origin = metrics.get("origin_audit", {})
    known_origins = set(origin.get("known_origins", []))
    unknown_windows = int(origin.get("unknown_windows", 0) or 0)
    temporal = metrics.get("temporal_coverage_audit", {})
    uncertainty = metrics.get("test_metric_uncertainty", {})
    evaluation_protocol = str(metrics.get("evaluation_protocol", "unknown"))
    f1_interval = (uncertainty.get("metrics", {}).get("f1", {}).get("ci95"))
    test_size = int(metrics.get("test_size", 0) or 0)
    has_stream_identity = bool(temporal.get("has_explicit_stream_identity"))
    has_event_identity = bool(temporal.get("has_explicit_event_identity"))
    latest_anomaly = bool(temporal.get("latest_block_contains_anomaly"))

    checks = [
        {
            "key": "test_sample_size",
            "passed": test_size >= min_test_windows,
            "observed": test_size,
            "required": min_test_windows,
            "message": (f"测试集窗口数 {test_size}，要求至少 {min_test_windows}。"),
        },
        {
            "key": "origin_coverage",
            "passed": (origin.get("status") == "ready" and unknown_windows == 0
                       and set(required_origins).issubset(known_origins)),
            "known_origins": sorted(known_origins),
            "required_origins": list(required_origins),
            "unknown_windows": unknown_windows,
            "message": "测试数据来源须完整标注，并覆盖要求的来源类型。",
        },
        {
            "key": "stream_and_event_identity",
            "passed": bool(temporal.get("forward_evaluation_data_eligible") and has_stream_identity
                           and has_event_identity and latest_anomaly),
            "has_stream_identity": has_stream_identity,
            "has_event_identity": has_event_identity,
            "latest_block_contains_anomaly": latest_anomaly,
            "message": "前向评测数据需要完整基站/数据流 ID、异常事件 ID，并在最新时间块保留异常样本。",
        },
        {
            "key": "forward_time_evaluation",
            "passed": evaluation_protocol == "forward_time_holdout",
            "observed_protocol": evaluation_protocol,
            "required_protocol": "forward_time_holdout",
            "message": "模型晋级证据必须来自按时间向前留出的测试集；随机分层留出不能证明未来泛化。",
        },
        {
            "key": "uncertainty_estimate",
            "passed": _valid_unit_interval(f1_interval),
            "f1_ci95": f1_interval,
            "message": "测试集需要报告有限、递增且位于 0–1 范围内的分组自助法 F1 区间。",
        },
    ]
    failures = [row["message"] for row in checks if not row["passed"]]
    return {
        "status": "ready_for_review" if not failures else "hold",
        "promotion_allowed": not failures,
        "required_origins": list(required_origins),
        "minimum_test_windows": min_test_windows,
        "checks": checks,
        "blocking_reasons": failures,
        "note": "此门禁检查评测证据是否完整，不替代业务风险审查、灰度发布或人工批准。",
    }
