"""Audit temporal and entity coverage before interpreting held-out metrics."""
from __future__ import annotations

from datetime import datetime

import numpy as np


# A scenario describes a test condition, not a persistent RAN entity or stream.
# It must not satisfy the identity requirement for future-per-entity evaluation.
ENTITY_FIELDS = ("cell_id", "site_id", "entity_id", "network_element_id", "stream_id")
EVENT_FIELDS = ("event_id", "incident_id", "anomaly_event_id", "fault_event_id")
CONTRACT_FIELDS = {
    "sampling_rate": ("sampling_rate",),
    "measurement_units": ("units", "kpi_units", "measurement_units"),
    "schema_version": ("schema_version", "data_schema_version"),
    "data_source": ("data_source", "source", "origin"),
}


def _field_value(record: dict, field: str):
    value = record.get(field)
    if value in (None, ""):
        for container_name in ("metadata", "labels"):
            value = (record.get(container_name) or {}).get(field)
            if value not in (None, ""):
                break
    return value


def _first_value(record: dict, fields: tuple[str, ...]):
    for field in fields:
        value = _field_value(record, field)
        if value not in (None, ""):
            return value
    return None


def temporal_coverage_audit(records: list[dict], causes, group_ids, *, blocks: int = 5) -> dict:
    """Summarize chronological class coverage without claiming a valid future split."""
    labels = np.asarray(causes, dtype=str)
    groups = np.asarray(group_ids)
    if (not records or labels.ndim != 1 or groups.ndim != 1
            or len(records) != len(labels) or len(records) != len(groups)):
        raise ValueError("时间覆盖审计需要等长、非空的记录、标签和时间组。")
    if not isinstance(blocks, int) or not 2 <= blocks <= 20:
        raise ValueError("时间覆盖分段数须为 2–20。")

    group_rows = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        timestamps = [datetime.fromisoformat(str(records[i]["start_time"]).replace("Z", "+00:00"))
                      for i in indices]
        group_rows.append((min(timestamps), group, indices))
    group_rows.sort(key=lambda row: row[0])
    splits = np.array_split(np.arange(len(group_rows)), min(blocks, len(group_rows)))

    entity_identity_coverage = {}
    for field in ENTITY_FIELDS:
        populated = sum(_field_value(record, field) not in (None, "") for record in records)
        if populated:
            entity_identity_coverage[field] = {"windows": populated, "fraction": populated / len(records)}
    has_explicit_identity = any(row["fraction"] == 1.0 for row in entity_identity_coverage.values())
    anomaly_indices = np.flatnonzero(labels != "Normal")
    event_identity_coverage = {}
    for field in EVENT_FIELDS:
        populated = sum(_field_value(records[int(index)], field) not in (None, "")
                        for index in anomaly_indices)
        if populated:
            event_identity_coverage[field] = {
                "anomaly_windows": populated,
                "fraction": populated / len(anomaly_indices),
            }
    has_explicit_event_identity = bool(len(anomaly_indices)) and any(
        row["fraction"] == 1.0 for row in event_identity_coverage.values())
    class_names = sorted(set(labels))
    timeline = []
    for block_number, split in enumerate(splits, start=1):
        selected_groups = [group_rows[int(position)] for position in split]
        indices = np.concatenate([row[2] for row in selected_groups])
        block_labels = labels[indices]
        class_windows = {name: int(np.sum(block_labels == name)) for name in class_names}
        class_groups = {
            name: int(sum(np.any(labels[row[2]] == name) for row in selected_groups))
            for name in class_names
        }
        end_times = [datetime.fromisoformat(str(records[i]["end_time"]).replace("Z", "+00:00"))
                     for i in indices]
        timeline.append({
            "block": block_number,
            "time_start": min(row[0] for row in selected_groups).isoformat(),
            "time_end": max(end_times).isoformat(),
            "time_groups": len(selected_groups),
            "windows": int(len(indices)),
            "class_windows": class_windows,
            "class_groups": class_groups,
        })

    durations = []
    for record in records:
        start = datetime.fromisoformat(str(record["start_time"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(record["end_time"]).replace("Z", "+00:00"))
        if start.tzinfo is not None:
            start = start.astimezone().replace(tzinfo=None)
            end = end.astimezone().replace(tzinfo=None)
        durations.append((end - start).total_seconds())

    last = timeline[-1]
    contract_coverage = {}
    for name, fields in CONTRACT_FIELDS.items():
        populated = sum(_first_value(record, fields) is not None for record in records)
        contract_coverage[name] = {
            "windows": populated,
            "fraction": populated / len(records),
            "complete": populated == len(records),
        }
    return {
        "windows": len(records),
        "overlap_groups": len(group_rows),
        "largest_overlap_group_windows": max(len(row[2]) for row in group_rows),
        "start_time": min(row[0] for row in group_rows).isoformat(),
        "end_time": max(datetime.fromisoformat(str(record["end_time"]).replace("Z", "+00:00"))
                         for record in records).isoformat(),
        "median_window_duration_seconds": float(np.median(durations)),
        "entity_identity_coverage": entity_identity_coverage,
        "has_explicit_stream_identity": has_explicit_identity,
        "event_identity_coverage": event_identity_coverage,
        "data_contract_coverage": contract_coverage,
        "has_explicit_event_identity": has_explicit_event_identity,
        "chronological_blocks": timeline,
        "latest_block_contains_anomaly": bool(sum(
            count for name, count in last["class_windows"].items() if name != "Normal")),
        "forward_evaluation_data_eligible": bool(
            has_explicit_identity and has_explicit_event_identity and sum(
                count for name, count in last["class_windows"].items() if name != "Normal")
        ),
        "interpretation": (
            "重叠窗口组按时间排序，仅用于覆盖审计。没有基站/流标识和异常事件标识时，"
            "不能证明事件独立，也不能把全局时间切分解释为未来泛化测试。采样率、单位、"
            "模式版本和来源缺失时保留为未知，不根据样本内容补造元数据。"
        ),
    }
