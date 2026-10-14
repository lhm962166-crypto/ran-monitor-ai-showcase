"""Explicit, auditable joining of human cause reviews to source KPI windows."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy

from .features import window_identity
from .settings import CAUSE_ZH


def compile_reviewed_cause_records(records: list[dict], reviews: list[dict]) -> tuple[list[dict], dict]:
    """Create a reviewed dataset from exact source windows; never train automatically."""
    if not isinstance(records, list) or not isinstance(reviews, list):
        raise ValueError("数据窗口与人工反馈都须为数组。")
    windows = {}
    for record in records:
        identity = window_identity(record)
        if identity in windows:
            raise ValueError(f"源数据中存在重复窗口身份 {identity}，无法安全关联人工反馈。")
        windows[identity] = record

    grouped = defaultdict(list)
    skipped = Counter()
    for review in reviews:
        if not isinstance(review, dict):
            skipped["invalid_review"] += 1
            continue
        window_id = review.get("window_id")
        cause = review.get("confirmed_cause")
        if review.get("feedback_version") != "cause-review-v1":
            skipped["unsupported_feedback_version"] += 1
            continue
        if cause not in CAUSE_ZH:
            skipped["unsupported_cause"] += 1
            continue
        if review.get("confidence") not in {"high", "medium"}:
            skipped["low_or_undetermined_confidence"] += 1
            continue
        if not str(review.get("reviewer") or "").strip() or not str(review.get("note") or "").strip():
            skipped["missing_review_audit_fields"] += 1
            continue
        if window_id not in windows:
            skipped["source_window_not_found"] += 1
            continue
        grouped[window_id].append(review)

    compiled = []
    conflicts = []
    for window_id, entries in grouped.items():
        causes = {entry["confirmed_cause"] for entry in entries}
        if len(causes) > 1:
            conflicts.append(window_id)
            continue
        # Multiple agreeing reviews remain in the audit record; the latest is the label source.
        chosen = max(entries, key=lambda entry: str(entry.get("created_at") or ""))
        record = deepcopy(windows[window_id])
        record["reviewed_cause"] = chosen["confirmed_cause"]
        record["review_metadata"] = {
            "dataset_schema": "reviewed-cause-window-v1",
            "source_window_id": window_id,
            "reviewer": str(chosen["reviewer"]).strip()[:120],
            "confidence": chosen["confidence"],
            "feedback_version": chosen["feedback_version"],
            "feedback_count": len(entries),
            "feedback_ids": [str(entry.get("alert_id", "")) for entry in entries],
        }
        compiled.append(record)

    report = {
        "schema": "reviewed-cause-compile-report-v1",
        "source_windows": len(records),
        "feedback_rows": len(reviews),
        "compiled_windows": len(compiled),
        "cause_counts": {cause: sum(row["reviewed_cause"] == cause for row in compiled)
                         for cause in ("Congestion", "Equipment", "Interference", "Mobility")},
        "conflicting_window_ids": sorted(conflicts),
        "skipped_rows": dict(sorted(skipped.items())),
        "ready_for_data_owner_review": bool(compiled) and not conflicts,
        "training_eligible": False,
        "training_scope_note": (
            "这是人工复核标签与精确原始窗口的关联产物；必须由数据负责人审查并安排到训练分区，"
            "不能覆盖已冻结测试集，也不会被应用自动载入。"
        ),
    }
    return compiled, report
