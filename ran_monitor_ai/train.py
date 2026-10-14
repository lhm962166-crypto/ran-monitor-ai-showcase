"""Train project models from an explicitly supplied local dataset."""
from __future__ import annotations

import json
import hashlib
import heapq
import os
import pickle
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .runtime import configure_threads
configure_threads()

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from .features import extract_features, true_cause
from .diagnosis import (choose_classifier_weight, evaluate_rank_matrix,
                        fit_cause_evidence_ranker, fused_ranking_scores,
                        prototype_distances)
from .evaluation import (anomaly_origin, calibration_diagnostics,
                         cluster_bootstrap_binary_metrics, evaluate_by_origin,
                         evaluate_two_stage)
from .data_audit import temporal_coverage_audit
from .governance import promotion_readiness
from .settings import KPIS
from .structure_detector import evaluate_shadow_scores, fit_structure_detector, score_structure
from .uncertainty import (evaluate_candidate_ranking, evaluate_prediction_sets,
                          fit_cause_calibration, temperature_scale)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = ROOT / "data" / "telecomts_demo.jsonl"
DATA_PATH = Path(os.environ.get("RAN_MONITOR_DATA_PATH", DEFAULT_DATA_PATH)).expanduser().resolve()
ARTIFACT_PATH = Path(os.environ.get("RAN_MONITOR_ARTIFACT_PATH", ROOT / "artifacts" / "model.joblib")).expanduser().resolve()
MODEL_FORMAT_VERSION = 9
_TRAIN_LOCK = threading.Lock()


def measured_feature_names(feature_names: list[str]) -> list[str]:
    """Return features derived from KPI values, excluding target/context labels."""
    return [name for name in feature_names
            if not name.startswith(("zone_", "application_"))
            and name not in {"mobility", "congestion"}]


def overlap_group_class_weights(groups, labels) -> np.ndarray:
    """Give each cause equal total weight and each time-overlap group equal weight within its cause."""
    group_values = np.asarray(groups, dtype=object)
    label_values = np.asarray(labels, dtype=object)
    if (group_values.ndim != 1 or label_values.ndim != 1
            or len(group_values) == 0 or len(group_values) != len(label_values)
            or any(group is None for group in group_values.tolist())):
        raise ValueError("原因分类的重叠组权重输入无效。")
    try:
        pairs = list(zip(group_values.tolist(), label_values.tolist()))
        group_class_counts = Counter(pairs)
        groups_per_class = Counter(label for _, label in group_class_counts)
        weights = np.asarray([
            1.0 / (groups_per_class[label] * group_class_counts[(group, label)])
            for group, label in pairs
        ], dtype=float)
    except (TypeError, ZeroDivisionError) as error:
        raise ValueError("原因分类的重叠组与类别标识必须可哈希且有效。") from error
    mean_weight = float(weights.mean())
    if not np.isfinite(weights).all() or mean_weight <= 0:
        raise ValueError("原因分类的重叠组权重必须为有限正数。")
    return weights / mean_weight


def group_overlapping_windows(records: list[dict]) -> np.ndarray:
    """Join windows connected by event identity or same-stream time overlap."""
    parent = list(range(len(records)))
    rank = [0] * len(records)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    def identities(record: dict, fields: tuple[str, ...]) -> list[str]:
        labels = record.get("labels") or {}
        metadata = record.get("metadata") or {}
        labels = labels if isinstance(labels, dict) else {}
        metadata = metadata if isinstance(metadata, dict) else {}
        found = []
        for field in fields:
            for source in (record, labels, metadata):
                value = source.get(field)
                if value is not None and str(value).strip() and str(value).strip() not in found:
                    found.append(str(value).strip())
        return found

    event_owner: dict[str, int] = {}
    intervals_by_stream: dict[str, list[tuple[datetime, datetime, int]]] = {}
    unknown_intervals: list[tuple[datetime, datetime, int]] = []
    for index, record in enumerate(records):
        event_ids = identities(record, ("event_id", "incident_id", "anomaly_event_id", "fault_event_id"))
        for event_id in event_ids:
            if event_id in event_owner:
                union(index, event_owner[event_id])
            else:
                event_owner[event_id] = index

        stream_ids = identities(record, ("stream_id", "cell_id", "site_id", "entity_id", "network_element_id"))
        try:
            start = _normalized_time(record["start_time"])
            end = _normalized_time(record["end_time"])
            if end <= start:
                raise ValueError("end_time must follow start_time")
        except (KeyError, TypeError, ValueError):
            # Invalid intervals cannot prove temporal overlap; event identity still links them.
            continue
        if stream_ids:
            for stream_id in stream_ids:
                intervals_by_stream.setdefault(stream_id, []).append((start, end, index))
        else:
            unknown_intervals.append((start, end, index))

    def join_time_overlaps(intervals: list[tuple[datetime, datetime, int]]) -> None:
        intervals.sort(key=lambda item: item[0])
        furthest_end = None
        furthest_index = None
        for start, end, index in intervals:
            if furthest_end is not None and start < furthest_end:
                union(index, furthest_index)
            if furthest_end is None or end > furthest_end:
                furthest_end, furthest_index = end, index

    for intervals in intervals_by_stream.values():
        join_time_overlaps(intervals)
    # Unknown stream IDs are one possible stream, and may also be part of any named
    # stream. Join every overlapping known interval rather than silently separating them.
    join_time_overlaps(unknown_intervals)
    if unknown_intervals and intervals_by_stream:
        known_by_index = {
            index: (start, end, index, False)
            for intervals in intervals_by_stream.values()
            for start, end, index in intervals
        }
        combined = list(known_by_index.values()) + [
            (start, end, index, True) for start, end, index in unknown_intervals
        ]
        active_heap: list[tuple[datetime, int]] = []
        active: dict[int, tuple[datetime, int, bool]] = {}
        token = 0
        for start, end, index, is_unknown in sorted(combined, key=lambda item: item[0]):
            while active_heap and active_heap[0][0] <= start:
                _, expired = heapq.heappop(active_heap)
                active.pop(expired, None)
            if is_unknown:
                # A missing identity could refer to any overlapping named stream.
                for _, other_index, _ in active.values():
                    union(index, other_index)
            else:
                for _, other_index, other_is_unknown in active.values():
                    if other_is_unknown:
                        union(index, other_index)
            heapq.heappush(active_heap, (end, token))
            active[token] = (end, index, is_unknown)
            token += 1

    roots: dict[int, int] = {}
    groups = np.empty(len(records), dtype=int)
    for index in range(len(records)):
        root = find(index)
        if root not in roots:
            roots[root] = len(roots)
        groups[index] = roots[root]
    return groups


def _normalized_time(value) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def split_time_groups(records: list[dict], anomaly_labels: np.ndarray,
                      cause_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Create independent train/validation/test partitions from time-overlap groups."""
    indices = np.arange(len(records))
    timezones = set()
    for index, record in enumerate(records, start=1):
        try:
            start = datetime.fromisoformat(str(record["start_time"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(record["end_time"]).replace("Z", "+00:00"))
            if (start.tzinfo is None) != (end.tzinfo is None) or end <= start:
                raise ValueError("invalid interval")
            timezones.add(start.tzinfo is not None)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"训练窗口 {record.get('sample_id', index)} 缺少有效的开始/结束时间，无法保证独立划分。") from error
    if len(timezones) > 1:
        raise ValueError("训练窗口混用了带时区和不带时区的时间，请统一时区后训练。")
    groups = group_overlapping_windows(records)
    audit = temporal_coverage_audit(records, cause_labels, groups)
    if audit.get("forward_evaluation_data_eligible"):
        return split_forward_time_groups(records, anomaly_labels, cause_labels, groups)
    group_count = len(np.unique(groups))
    if group_count < 3:
        raise ValueError("至少需要 3 个不重叠时间组才能划分训练、验证和测试集；请补充数据。")
    # Keep groups intact even when the data cannot support the default five folds.
    # A random-window fallback would silently reintroduce overlap leakage.
    class_set = set(cause_labels)
    for split_count in range(min(5, group_count), 2, -1):
        for seed in (42, 43, 44, 45):
            try:
                folds = list(StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=seed).split(
                    indices, cause_labels, groups
                ))
            except ValueError:
                continue
            test_idx, val_idx = folds[0][1], folds[1][1]
            train_idx = np.setdiff1d(indices, np.concatenate((test_idx, val_idx)))
            if set(cause_labels[train_idx]) != class_set:
                continue
            if any(len(np.unique(anomaly_labels[part])) != 2 for part in (train_idx, val_idx, test_idx)):
                continue
            note = (
                f"按重叠时间窗口分组后进行 {split_count} 折分层切分，验证集与测试集互不重叠；"
                "演示子集较小，指标仅供功能验证。"
            )
            return train_idx, val_idx, test_idx, note
    raise ValueError("无法在保持时间组独立的同时覆盖正常与故障类别；请补充各类独立窗口。")


def split_forward_time_groups(records: list[dict], anomaly_labels: np.ndarray,
                              cause_labels: np.ndarray, groups: np.ndarray | None = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Make chronological train/validation/test partitions when identity evidence exists."""
    if groups is None:
        groups = group_overlapping_windows(records)
    rows = []
    for group in np.unique(groups):
        indices = np.flatnonzero(groups == group)
        start = min(_normalized_time(records[index]["start_time"]) for index in indices)
        end = max(_normalized_time(records[index]["end_time"]) for index in indices)
        rows.append((start, end, int(group), indices))
    rows.sort(key=lambda item: item[0])
    group_count = len(rows)
    if group_count < 5:
        raise ValueError("身份信息完整，但不重叠时间组不足 5 组，无法形成前向训练/验证/测试集。")
    indices = np.arange(len(records))
    class_set = set(cause_labels)
    candidates = []
    for train_fraction, validation_fraction in ((.6, .2), (.5, .25), (.65, .15)):
        train_end = int(group_count * train_fraction)
        validation_end = int(group_count * (train_fraction + validation_fraction))
        if train_end < 2 or validation_end <= train_end or validation_end >= group_count:
            continue
        parts = [np.concatenate([row[3] for row in rows[left:right]])
                 for left, right in ((0, train_end), (train_end, validation_end),
                                     (validation_end, group_count))]
        # Every sample window must finish before the next forward-time partition begins.
        # Event identity can join disjoint intervals, so checking group start alone is unsafe.
        if (max(row[1] for row in rows[:train_end]) > min(row[0] for row in rows[train_end:validation_end])
                or max(row[1] for row in rows[train_end:validation_end])
                > min(row[0] for row in rows[validation_end:])):
            continue
        train_idx, val_idx, test_idx = parts
        supports = [np.sum(anomaly_labels[part] == 1) for part in parts]
        if (set(cause_labels[train_idx]) != class_set
                or any(len(np.unique(anomaly_labels[part])) != 2 for part in parts)
                or not supports[1] or not supports[2]):
            continue
        candidates.append((sum(len(part) for part in parts), train_idx, val_idx, test_idx,
                           train_fraction, validation_fraction))
    if not candidates:
        raise ValueError("身份信息完整，但按时间向前切分后未能覆盖必要类别；请补充早期训练、验证和最新测试事件。")
    _, train_idx, val_idx, test_idx, train_fraction, validation_fraction = max(candidates, key=lambda row: row[0])
    note = (f"依据显式流/事件身份将重叠时间窗口分组，再按时间向前切分；训练约 {train_fraction:.0%}、"
            f"验证约 {validation_fraction:.0%}、最新测试约 {1 - train_fraction - validation_fraction:.0%}。"
            "该协议适用于此数据快照，不替代跨基站留出。")
    return train_idx, val_idx, test_idx, note


def _dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_data_path() -> Path:
    """Resolve the locally supplied training data without bundling it in Git."""
    configured = os.environ.get("RAN_MONITOR_DATA_PATH")
    return Path(configured).expanduser().resolve() if configured else DEFAULT_DATA_PATH


def frozen_development_split_path(data_path: Path | None = None) -> Path:
    return Path(data_path or get_data_path()).with_name("evaluation_split_dev.json")


def freeze_development_split(data_path: Path | None = None) -> dict:
    """Write a reproducible development partition manifest for an exact data snapshot."""
    path = Path(data_path or get_data_path())
    records = load_records(path)
    causes = np.asarray([true_cause(record) for record in records], dtype=str)
    anomaly = (causes != "Normal").astype(int)
    train_idx, val_idx, test_idx, split_note = split_time_groups(records, anomaly, causes)
    groups = group_overlapping_windows(records)
    manifest = {
        "schema": "development-evaluation-split-v1",
        "dataset_role": "development_only_already_inspected",
        "dataset_file": path.name,
        "dataset_sha256": _dataset_sha256(path),
        "sample_count": len(records),
        "split_note": split_note,
        "partitions": {
            name: {"indices": part.astype(int).tolist(),
                   "windows": int(len(part)),
                   "overlap_groups": int(len(np.unique(groups[part])))}
            for name, part in (("train", train_idx), ("validation", val_idx), ("test", test_idx))
        },
        "protocol_warning": (
            "这是已查看的演示数据开发划分，用于固定代码迭代对照；不构成新的冻结最终测试集，"
            "也不证明跨基站或独立故障事件泛化。"
        ),
    }
    target = frozen_development_split_path(path)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return manifest


def load_frozen_development_split(records: list[dict], data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
    """Load a split only when its exact dataset digest and group isolation still match."""
    manifest_path = frozen_development_split_path(data_path)
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "development-evaluation-split-v1"
                or manifest.get("dataset_role") != "development_only_already_inspected"
                or manifest.get("sample_count") != len(records)
                or manifest.get("dataset_sha256") != _dataset_sha256(data_path)):
            raise ValueError
        partitions = manifest["partitions"]
        raw_indices = [partitions[name]["indices"]
                       for name in ("train", "validation", "test")]
        if any(not isinstance(part, list) or any(
                isinstance(index, bool) or not isinstance(index, int) for index in part
        ) for part in raw_indices):
            raise ValueError
        arrays = [np.asarray(part, dtype=int) for part in raw_indices]
        combined = np.concatenate(arrays)
        if (len(combined) != len(records) or set(combined.tolist()) != set(range(len(records)))
                or any(len(np.unique(part)) != len(part) for part in arrays)):
            raise ValueError
        groups = group_overlapping_windows(records)
        memberships = [set(groups[part].tolist()) for part in arrays]
        if any(memberships[left] & memberships[right]
               for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError
        return (*arrays, "使用 SHA-256 绑定的固定开发集划分；数据身份不足，不代表最终或现网泛化测试。")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OverflowError) as error:
        raise ValueError("固定开发划分清单与数据不匹配或无效；请审查数据变更后重新冻结开发划分。") from error


def load_records(path: Path | None = None) -> list[dict]:
    path = Path(path or get_data_path())
    if not path.is_file():
        raise FileNotFoundError(
            f"未找到训练数据：{path}。请从 TelecomTS 官方数据页单独获取并转换为项目 JSONL 格式，"
            "或设置 RAN_MONITOR_DATA_PATH 指向已获准使用的数据文件。仓库不分发训练样本。"
        )
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"No training examples found at {path}")
    return records


def threshold_profiles(labels: np.ndarray, scores: np.ndarray) -> dict:
    if len(np.unique(labels)) != 2:
        raise ValueError("阈值校准需要同时包含正常和异常的验证窗口。")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    profiles = {}
    for name, target in (("balanced", .90), ("sensitive", .95)):
        eligible = np.flatnonzero(recall[:-1] >= target)
        threshold = float(thresholds[eligible[-1]])
        profiles[name] = {"threshold": threshold, "target_recall": target, "target_met": True}
    eligible = np.flatnonzero(precision[:-1] >= .98)
    profiles["precision"] = {"threshold": float(thresholds[eligible[0]]) if len(eligible) else profiles["balanced"]["threshold"],
                             "target_precision": .98, "target_met": bool(len(eligible))}
    for profile in profiles.values():
        predicted = scores >= profile["threshold"]
        profile.update(validation_precision=float(precision_score(labels, predicted, zero_division=0)),
                       validation_recall=float(recall_score(labels, predicted, zero_division=0)),
                       validation_confusion=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())
    return profiles


def calibrate_threshold(labels, scores, objective: str, target: float) -> dict:
    """Choose a detector threshold using validation data only."""
    y = np.asarray(labels)
    values = np.asarray(scores, dtype=float)
    if y.ndim != 1 or values.ndim != 1 or len(y) == 0 or len(y) != len(values):
        raise ValueError("阈值校准输入须是一维、非空且长度一致。")
    if not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2:
        raise ValueError("阈值校准需要同时包含正常和异常的验证窗口。")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("验证集异常评分须为 0–1 的有限数值。")
    if objective not in {"fpr", "recall"}:
        raise ValueError("校准目标须为 fpr 或 recall。")
    if not np.isfinite(target) or not 0 <= target <= 1:
        raise ValueError("校准目标须为 0–1。")

    candidates = np.unique(np.concatenate(([0.0, 1.0], values)))
    rows = []
    positive = y == 1
    negative = ~positive
    for threshold in candidates:
        predicted = values >= threshold
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & negative))
        fn = int(np.sum(~predicted & positive))
        tn = int(np.sum(~predicted & negative))
        recall = tp / (tp + fn)
        fpr = fp / (fp + tn)
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"threshold": float(threshold), "precision": precision,
                     "recall": recall, "false_positive_rate": fpr, "f1": f1,
                     "confusion": [[tn, fp], [fn, tp]]})

    if objective == "fpr":
        eligible = [row for row in rows if row["false_positive_rate"] <= target]
        target_met = bool(eligible)
        chosen = max(eligible or rows, key=lambda row: (
            row["recall"], -row["false_positive_rate"], row["threshold"]))
    else:
        eligible = [row for row in rows if row["recall"] >= target]
        target_met = bool(eligible)
        if eligible:
            chosen = max(eligible, key=lambda row: row["threshold"])
        else:
            chosen = max(rows, key=lambda row: (
                row["recall"], -row["false_positive_rate"], row["threshold"]))
    return chosen | {"objective": objective, "target": float(target), "target_met": target_met}


def build_bundle(records: list[dict], *, data_path: Path | None = None) -> tuple[dict, dict]:
    rows = [extract_features(record) for record in records]
    data = pd.DataFrame(rows).fillna(0.0)
    y_anomaly = np.array([true_cause(record) != "Normal" for record in records], dtype=int)
    cause_names = sorted({true_cause(record) for record in records if true_cause(record) != "Normal"})
    cause_to_id = {name: index for index, name in enumerate(cause_names)}
    y_cause = np.array([cause_to_id.get(true_cause(record), -1) for record in records], dtype=int)
    if len(np.unique(y_anomaly)) != 2 or set(cause_names) != {"Congestion", "Equipment", "Interference", "Mobility"}:
        raise ValueError("Training subset must include normal windows and all four fault causes.")

    coarse_causes = np.array([true_cause(record) for record in records])
    frozen_split = load_frozen_development_split(records, data_path) if data_path is not None else None
    train_idx, val_idx, test_idx, split_note = (frozen_split if frozen_split is not None else
                                                 split_time_groups(records, y_anomaly, coarse_causes))
    all_time_groups = group_overlapping_windows(records)
    feature_names = list(data.columns)
    # Labels are useful for analysis context but leak the target into training.
    # Both stages must rely only on measured KPI windows.
    kpi_features = measured_feature_names(feature_names)
    cause_features = kpi_features
    medians = data.iloc[train_idx].replace([np.inf, -np.inf], np.nan).median().fillna(0.0)
    clean = data.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    positive = max(1, int(np.sum(y_anomaly[train_idx] == 1)))
    negative = int(np.sum(y_anomaly[train_idx] == 0))
    detector = xgb.XGBClassifier(
        n_estimators=260, max_depth=4, learning_rate=0.07,
        subsample=0.9, colsample_bytree=0.9, n_jobs=4,
        scale_pos_weight=negative / positive, eval_metric="logloss", random_state=42,
    )
    detector.fit(clean.iloc[train_idx][kpi_features], y_anomaly[train_idx])
    val_scores = detector.predict_proba(clean.iloc[val_idx][kpi_features])[:, 1]
    profiles = threshold_profiles(y_anomaly[val_idx], val_scores)
    threshold = profiles["balanced"]["threshold"]

    classes = cause_names
    clf = xgb.XGBClassifier(
        n_estimators=220, max_depth=4, learning_rate=0.07,
        subsample=0.9, colsample_bytree=0.9, objective="multi:softprob",
        num_class=len(classes), eval_metric="mlogloss", random_state=43, n_jobs=4,
    )
    anomaly_train = train_idx[y_anomaly[train_idx] == 1]
    cause_sample_weights = overlap_group_class_weights(
        all_time_groups[anomaly_train], y_cause[anomaly_train],
    )
    clf.fit(clean.iloc[anomaly_train][cause_features], y_cause[anomaly_train],
            sample_weight=cause_sample_weights)
    cause_training_policy = {
        "method": "equal_cause_mass_and_equal_overlap_group_mass",
        "scope": "training_anomaly_windows_only",
        "class_support": {
            cause: {
                "windows": int(np.sum(y_cause[anomaly_train] == class_id)),
                "overlap_groups": int(len(np.unique(
                    all_time_groups[anomaly_train][y_cause[anomaly_train] == class_id]
                ))),
                "effective_weight": float(np.sum(
                    cause_sample_weights[y_cause[anomaly_train] == class_id]
                )),
            }
            for class_id, cause in enumerate(classes)
        },
        "note": "每类总权重相等；同一类别内每个不重叠时间组的总权重相等。",
    }

    evidence_ranker = fit_cause_evidence_ranker(
        clean.iloc[anomaly_train][cause_features].to_numpy(), y_cause[anomaly_train],
        classes, cause_features,
    )

    cause_val_idx = val_idx[y_anomaly[val_idx] == 1]
    cause_val_probabilities = clf.predict_proba(clean.iloc[cause_val_idx][cause_features])
    cause_val_groups = sorted(set(all_time_groups[cause_val_idx].tolist()), key=lambda group: min(
        _normalized_time(records[index]["start_time"])
        for index in cause_val_idx if all_time_groups[index] == group
    ))
    temperature_groups = set(cause_val_groups[::2]) if len(cause_val_groups) >= 2 else set()
    aps_groups = set(cause_val_groups[1::2]) if len(cause_val_groups) >= 2 else set(cause_val_groups)
    temperature_mask = np.asarray([all_time_groups[index] in temperature_groups for index in cause_val_idx])
    aps_mask = np.asarray([all_time_groups[index] in aps_groups for index in cause_val_idx])
    if not np.any(aps_mask):
        aps_mask = np.ones(len(cause_val_idx), dtype=bool)
    cause_calibration = fit_cause_calibration(
        cause_val_probabilities[aps_mask], y_cause[cause_val_idx][aps_mask],
        temperature_probabilities=(cause_val_probabilities[temperature_mask]
                                    if np.any(temperature_mask) else None),
        temperature_labels=(y_cause[cause_val_idx][temperature_mask]
                            if np.any(temperature_mask) else None),
        independent_groups=len(aps_groups), temperature_groups=len(temperature_groups),
    )
    cause_val_evidence_distances = prototype_distances(
        clean.iloc[cause_val_idx][cause_features].to_numpy(), evidence_ranker,
    )
    evidence_weight_selection = choose_classifier_weight(
        cause_val_probabilities, cause_val_evidence_distances, y_cause[cause_val_idx],
    )
    evidence_ranker["classifier_weight"] = evidence_weight_selection["classifier_weight"]

    normal_train_records = [records[index] for index in train_idx if y_anomaly[index] == 0]
    normal_validation_records = [records[index] for index in val_idx if y_anomaly[index] == 0]
    structure_model, structure_report = fit_structure_detector(
        normal_train_records, normal_validation_records,
    )

    normal_train = [i for i in train_idx if y_anomaly[i] == 0]
    baselines = {"__global__": _make_baseline(data.iloc[normal_train])}
    groups: dict[str, list[int]] = {}
    for idx in normal_train:
        labels = records[idx].get("labels") or {}
        if labels.get("zone") and labels.get("application"):
            groups.setdefault(f"zone_{labels['zone']}|application_{labels['application']}", []).append(idx)
    for key, group in groups.items():
        if len(group) >= 3:
            baselines[key] = _make_baseline(data.iloc[group])

    bundle = {
        "detector": detector, "classifier": clf, "threshold": threshold,
        "feature_names": feature_names, "kpi_features": kpi_features,
        "cause_features": cause_features,
        "threshold_profiles": profiles,
        "calibration": {
            "labels": y_anomaly[val_idx].astype(int).tolist(),
            "scores": val_scores.astype(float).tolist(),
            "normal_scores": val_scores[y_anomaly[val_idx] == 0].astype(float).tolist(),
            "origins": [anomaly_origin(records[index]) for index in val_idx],
        },
        "cause_calibration": cause_calibration,
        "cause_evidence_ranker": evidence_ranker,
        "cause_training_policy": cause_training_policy,
        "structure_detector": structure_model,
        "medians": {name: float(value) for name, value in medians.items()},
        "baselines": baselines, "classes": classes,
        "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
    }
    test_scores = detector.predict_proba(clean.iloc[test_idx][kpi_features])[:, 1]
    predicted = test_scores >= threshold
    cause_test_idx = test_idx[y_anomaly[test_idx] == 1]
    cause_probabilities = clf.predict_proba(clean.iloc[cause_test_idx][cause_features])
    cause_predicted = np.argmax(cause_probabilities, axis=1)
    cause_probability_diagnostics = calibration_diagnostics(
        y_cause[cause_test_idx], cause_probabilities,
    )
    cause_test_evidence_distances = prototype_distances(
        clean.iloc[cause_test_idx][cause_features].to_numpy(), evidence_ranker,
    )
    classifier_only_rank_scores = fused_ranking_scores(
        cause_probabilities, cause_test_evidence_distances, 1.0,
    )
    multisignal_rank_scores = fused_ranking_scores(
        cause_probabilities, cause_test_evidence_distances,
        evidence_ranker["classifier_weight"],
    )
    calibrated_cause_probabilities = temperature_scale(
        cause_probabilities, cause_calibration["temperature"],
    )
    calibrated_cause_probability_diagnostics = calibration_diagnostics(
        y_cause[cause_test_idx], calibrated_cause_probabilities,
    )
    cause_set_diagnostics = evaluate_prediction_sets(
        cause_probabilities, y_cause[cause_test_idx], classes, cause_calibration,
    )
    cause_candidate_ranking = evaluate_candidate_ranking(
        cause_probabilities, y_cause[cause_test_idx], budget=min(2, len(classes)),
    )
    pipeline_cause_predictions = np.full(len(test_idx), classes[0], dtype=object)
    detected_indices = np.flatnonzero(predicted)
    if len(detected_indices):
        detected_cause_ids = clf.predict(clean.iloc[test_idx[detected_indices]][cause_features])
        pipeline_cause_predictions[detected_indices] = [classes[int(index)] for index in detected_cause_ids]
    pipeline = evaluate_two_stage(
        coarse_causes[test_idx], test_scores, pipeline_cause_predictions, threshold, classes
    )
    origin_audit = evaluate_by_origin(
        coarse_causes[test_idx], test_scores, threshold,
        [anomaly_origin(records[index]) for index in test_idx],
    )
    test_uncertainty = cluster_bootstrap_binary_metrics(
        y_anomaly[test_idx], test_scores, threshold, all_time_groups[test_idx],
        replicates=500, seed=42,
    )
    detector_probability_diagnostics = calibration_diagnostics(
        y_anomaly[test_idx], test_scores,
    )
    test_structure_scores = np.asarray([
        score_structure(records[index], structure_model)["score"] for index in test_idx
    ], dtype=float)
    structure_shadow_metrics = evaluate_shadow_scores(
        test_structure_scores, y_anomaly[test_idx], structure_model["threshold"],
    )
    metrics = {
        "evaluation_protocol": ("forward_time_holdout" if "按时间向前切分" in split_note
                                else "overlap_group_stratified_holdout"),
        "samples": len(records), "positive_rate": float(y_anomaly.mean()),
        "test_size": len(test_idx), "threshold": threshold,
        "test_precision": float(precision_score(y_anomaly[test_idx], predicted, zero_division=0)),
        "test_recall": float(recall_score(y_anomaly[test_idx], predicted, zero_division=0)),
        "test_f1": float(f1_score(y_anomaly[test_idx], predicted, zero_division=0)),
        "test_auc": float(roc_auc_score(y_anomaly[test_idx], test_scores)),
        "test_average_precision": float(average_precision_score(y_anomaly[test_idx], test_scores)),
        "cause_test_macro_f1": float(f1_score(
            y_cause[cause_test_idx], cause_predicted, labels=np.arange(len(classes)),
            average="macro", zero_division=0)),
        "cause_classes": classes,
        "split_note": split_note,
        "threshold_profiles": profiles,
        "validation_precision": float(precision_score(y_anomaly[val_idx], val_scores >= threshold, zero_division=0)),
        "validation_recall": float(recall_score(y_anomaly[val_idx], val_scores >= threshold, zero_division=0)),
        "detector_confusion": confusion_matrix(y_anomaly[test_idx], predicted, labels=[0, 1]).tolist(),
        "cause_confusion": confusion_matrix(y_cause[cause_test_idx], cause_predicted,
                                             labels=np.arange(len(classes))).tolist(),
        "pipeline": pipeline,
        "origin_audit": origin_audit,
        "test_metric_uncertainty": test_uncertainty,
        "detector_probability_diagnostics": detector_probability_diagnostics,
        "cause_probability_diagnostics": cause_probability_diagnostics,
        "cause_calibration": cause_calibration,
        "cause_training_policy": cause_training_policy,
        "cause_evidence_ranker": evidence_ranker,
        "cause_calibrated_probability_diagnostics": calibrated_cause_probability_diagnostics,
        "cause_prediction_set_diagnostics": cause_set_diagnostics,
        "cause_candidate_ranking": cause_candidate_ranking,
        "cause_multisignal_ranking": {
            "status": "shadow_only",
            "method": evidence_ranker["method"],
            "validation_weight_selection": evidence_weight_selection,
            "classifier_only_test": evaluate_rank_matrix(
                classifier_only_rank_scores, y_cause[cause_test_idx], candidate_budget=2,
            ),
            "multisignal_test": evaluate_rank_matrix(
                multisignal_rank_scores, y_cause[cause_test_idx], candidate_budget=2,
            ),
            "note": "原型由训练故障窗口拟合，融合权重由验证窗口选择；仅作候选排序，不是原因概率或因果结论。",
        },
        "structure_shadow": structure_report,
        "structure_shadow_test": structure_shadow_metrics,
        "partitions": {name: {"size": len(part), "time_groups": len(np.unique(all_time_groups[part])),
                              "classes": {cause: int(np.sum(coarse_causes[part] == cause))
                                          for cause in ["Normal"] + classes}}
                       for name, part in (("train", train_idx), ("validation", val_idx), ("test", test_idx))},
        "temporal_coverage_audit": temporal_coverage_audit(records, coarse_causes, all_time_groups),
    }
    metrics["promotion_readiness"] = promotion_readiness(metrics)
    return bundle, metrics


def _make_baseline(rows: pd.DataFrame) -> dict:
    result = {"__metadata__": {"normal_windows": len(rows)}}
    for kpi in KPIS:
        column = f"{kpi}_mean"
        values = rows[column].astype(float)
        deviation = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[column] = {"mean": float(values.mean()), "std": max(deviation, 1e-6)}
    return result


def model_fingerprint(data_path: Path | None = None) -> str:
    """Invalidate artifacts when data, extraction, training, or dependencies change."""
    data_path = Path(data_path or get_data_path())
    digest = hashlib.sha256()
    source_paths = [data_path, frozen_development_split_path(data_path), Path(__file__),
                 Path(__file__).with_name("features.py"),
                 Path(__file__).with_name("settings.py"), Path(__file__).with_name("evaluation.py"),
                 Path(__file__).with_name("data_audit.py"), Path(__file__).with_name("governance.py"),
                 Path(__file__).with_name("uncertainty.py"),
                 Path(__file__).with_name("diagnosis.py"),
                 Path(__file__).with_name("structure_detector.py"),
                 Path(__file__).with_name("engine.py")]
    for path in source_paths:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\x00")
    import sklearn
    import scipy
    digest.update(f"{MODEL_FORMAT_VERSION}|{xgb.__version__}|{sklearn.__version__}|{np.__version__}|{pd.__version__}|{scipy.__version__}".encode())
    return digest.hexdigest()


def _atomic_write_model(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".model-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(force: bool = False) -> tuple[dict, dict]:
    with _TRAIN_LOCK:
        data_path = get_data_path()
        if not data_path.is_file():
            load_records(data_path)  # Raise an actionable setup message.
        fingerprint = model_fingerprint(data_path)
        if ARTIFACT_PATH.exists() and not force:
            try:
                bundle = joblib.load(ARTIFACT_PATH)
                metadata = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
                if isinstance(metadata, dict) and metadata.get("format_version") == MODEL_FORMAT_VERSION and metadata.get("fingerprint") == fingerprint:
                    return bundle, bundle["metrics"]
            except (EOFError, pickle.UnpicklingError, ImportError, AttributeError, IndexError,
                    ValueError, KeyError, TypeError, OSError):
                # Recover incomplete local artifacts by rebuilding from project data.
                pass
        bundle, metrics = build_bundle(load_records(data_path), data_path=data_path)
        bundle["metadata"] = {"format_version": MODEL_FORMAT_VERSION, "fingerprint": fingerprint,
                              "trained_at": datetime.now(timezone.utc).isoformat(),
                              "model_id": fingerprint[:12]}
        bundle["metrics"] = metrics
        _atomic_write_model(bundle, ARTIFACT_PATH)
        # This is a convenient export. The model itself is the authoritative snapshot.
        ARTIFACT_PATH.with_name("metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return bundle, metrics


if __name__ == "__main__":
    from .configuration import load_env_file
    load_env_file(ROOT / ".env")
    _, report = train(force=True)
    print(json.dumps(report, indent=2, ensure_ascii=False))
