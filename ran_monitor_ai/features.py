"""Raw TelecomTS window validation and feature extraction."""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache

from .runtime import configure_threads
configure_threads()

import numpy as np
from scipy.stats import kurtosis, skew

from .settings import CAUSE_MAP, KPIS, WINDOW_SIZE


def parse_json_payload(raw: bytes | str):
    """Decode an uploaded JSON document and turn parser failures into actionable errors."""
    try:
        document = raw.decode("utf-8-sig") if isinstance(raw, bytes) else raw
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"JSON 字段 {key} 重复；请保留唯一值后重试。")
                result[key] = value
            return result

        def invalid_constant(value):
            raise ValueError(f"JSON 不支持 {value}；请使用有限数值。")

        return json.loads(document, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"JSON 格式错误：第 {error.lineno} 行、第 {error.colno} 列附近无法解析。"
        ) from error
    except UnicodeDecodeError as error:
        raise ValueError("文件不是有效的 UTF-8 JSON；请另存为 UTF-8 编码后重试。") from error
    except RecursionError as error:
        raise ValueError("JSON 嵌套层级过深，请按单个窗口或窗口数组格式整理。") from error


def parse_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "1"}


def window_identity(record: dict) -> str:
    """Build a stable reference for an exact measured window without using labels."""
    labels = record.get("labels") or {}
    metadata = record.get("metadata") or {}
    stream_id = record.get("stream_id") or labels.get("stream_id") or metadata.get("stream_id")
    stream_id = str(stream_id).strip()[:200] if stream_id not in (None, "") else None
    identity = {
        "stream_id": stream_id,
        "sample_id": record.get("sample_id"),
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "sampling_rate": record.get("sampling_rate", metadata.get("sampling_rate")),
        "measurement_units": record.get("measurement_units", metadata.get("measurement_units",
                                 record.get("units", metadata.get("units")))),
        "schema_version": record.get("schema_version", metadata.get("schema_version")),
        "KPIs": record.get("KPIs", {}),
    }
    digest = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str,
    ).encode("utf-8")).hexdigest()[:24].upper()
    return f"WIN-{digest}"


def validate_window(record: dict) -> dict[str, np.ndarray]:
    if not isinstance(record, dict) or not isinstance(record.get("KPIs"), dict):
        raise ValueError("JSON 须包含 KPIs 字典。")
    labels = record.get("labels")
    if labels is not None and not isinstance(labels, dict):
        raise ValueError("labels 须为 JSON 对象。")
    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata 须为 JSON 对象。")
    for container in (record, labels or {}, metadata or {}):
        for field in ("stream_id", "cell_id", "site_id", "entity_id", "network_element_id",
                      "event_id", "incident_id", "anomaly_event_id", "fault_event_id"):
            value = container.get(field)
            if value in (None, ""):
                continue
            if (isinstance(value, bool) or not isinstance(value, (str, int)) or len(str(value)) > 200):
                raise ValueError(f"{field} 须为整数或不超过 200 字的文本。")
    sample_id = record.get("sample_id")
    if sample_id is not None and (isinstance(sample_id, bool) or not isinstance(sample_id, (str, int))
                                 or len(str(sample_id)) > 200):
        raise ValueError("sample_id 须为整数或不超过 200 字的文本。")
    result = {}
    for name in KPIS:
        raw = record["KPIs"].get(name, [])
        if (not isinstance(raw, (list, tuple, np.ndarray))
                or (isinstance(raw, np.ndarray) and raw.ndim != 1)
                or any(isinstance(value, (bool, np.bool_)) for value in raw)):
            raise ValueError(f"{name} 须包含 {WINDOW_SIZE} 个有效数值，不能使用布尔值。")
        try:
            values = np.asarray(raw, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} 须包含 {WINDOW_SIZE} 个有效数值。") from error
        if values.shape != (WINDOW_SIZE,) or not np.isfinite(values).all():
            raise ValueError(f"{name} 须包含 {WINDOW_SIZE} 个有效数值。")
        result[name] = values
    return result


def validate_batch(payload, max_records: int = 100) -> list[dict]:
    """Validate a single window or an all-or-nothing batch before inference."""
    if isinstance(payload, dict):
        items = [payload]
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("JSON 须为一个窗口对象或窗口对象数组。")
    if not items:
        raise ValueError("批量窗口不能为空。")
    if len(items) > max_records:
        raise ValueError(f"单次最多处理 {max_records} 个窗口；当前文件包含 {len(items)} 个。")

    checked = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 个窗口须为 JSON 对象。")
        try:
            validate_window(item)
        except ValueError as error:
            sample_id = item.get("sample_id", f"第 {index} 条")
            raise ValueError(f"窗口 {sample_id}（第 {index} 条）无效：{error}") from error
        normalized = dict(item)
        if normalized.get("sample_id") in (None, ""):
            normalized["sample_id"] = f"upload-{index}"
        checked.append(normalized)
    return checked


@lru_cache(maxsize=8)
def _slope_weights(size: int) -> np.ndarray:
    """Reuse the closed-form linear regression weights for fixed-size windows."""
    centered = np.arange(size, dtype=float) - (size - 1) / 2
    weights = centered / np.dot(centered, centered)
    weights.setflags(write=False)
    return weights


def _summarize_matrix(values: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all KPI summaries together; columns are successive samples."""
    means = values.mean(axis=1)
    deviations = values.std(axis=1)
    q1, medians, q3 = np.percentile(values, [25, 50, 75], axis=1)
    variable = deviations > 1e-9
    skewness = np.zeros(len(values))
    kurt = np.zeros(len(values))
    if variable.any():
        skewness[variable] = skew(values[variable], axis=1)
        kurt[variable] = kurtosis(values[variable], axis=1)
    differences = np.diff(values, axis=1)
    return {
        "mean": means, "std": deviations,
        "min": values.min(axis=1), "max": values.max(axis=1),
        "median": medians, "iqr": q3 - q1,
        "skew": skewness, "kurt": kurt,
        "slope": (values - means[:, None]) @ _slope_weights(values.shape[1]),
        "d1max": np.abs(differences).max(axis=1),
        "d2max": np.abs(np.diff(differences, axis=1)).max(axis=1),
    }


def summarize_series(values: np.ndarray) -> dict[str, float]:
    summaries = _summarize_matrix(np.asarray(values, dtype=float).reshape(1, -1))
    return {name: float(value[0]) for name, value in summaries.items()}


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - np.mean(left)
    right = right - np.mean(right)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


PAIRS = {
    "corr_RSRP_ULSNR": ("RSRP", "UL_SNR"),
    "corr_DLBLER_DLMCS": ("DL_BLER", "DL_MCS"),
    "corr_ULBLER_ULMCS": ("UL_BLER", "UL_MCS"),
    "corr_DLMCS_RXBYTES": ("DL_MCS", "RX_Bytes"),
    "corr_ULMCS_TXBYTES": ("UL_MCS", "TX_Bytes"),
    "corr_PRButilDL_RXBYTES": ("PRB_Utilization_DL", "RX_Bytes"),
    "corr_PRButilUL_TXBYTES": ("PRB_Utilization_UL", "TX_Bytes"),
    "corr_ULBUF_TXBYTES": ("Estimated_UL_Buffer", "TX_Bytes"),
}


def extract_features(record: dict) -> dict[str, float]:
    series = validate_window(record)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            summaries = _summarize_matrix(np.array(list(series.values())))
    except FloatingPointError as error:
        raise ValueError("KPI 数值范围过大，无法稳定计算统计特征；请检查单位与数据范围。") from error
    row = {}
    for index, name in enumerate(series):
        row.update({f"{name}_{key}": float(values[index]) for key, values in summaries.items()})
    row.update({key: correlation(series[a], series[b]) for key, (a, b) in PAIRS.items()})
    rsrp_mean = float(np.mean(series["RSRP"]))
    row["rsrp_throughput_ratio"] = (
        float(np.mean(series["RX_Bytes"])) / abs(rsrp_mean) if rsrp_mean else 0.0
    )
    labels = record.get("labels") or {}
    for value in sorted({"A", "B", "C", "In motion"}):
        row[f"zone_{value}"] = float(labels.get("zone") == value)
    for value in sorted({"File", "Twitch", "Youtube"}):
        row[f"application_{value}"] = float(labels.get("application") == value)
    row["mobility"] = float(parse_flag(labels.get("mobility", False)))
    row["congestion"] = float(parse_flag(labels.get("congestion", False)))
    if not all(np.isfinite(value) for value in row.values()):
        raise ValueError("KPI 特征出现非有限数值，请检查输入单位与数据范围。")
    return row


def true_cause(record: dict) -> str:
    reviewed = record.get("reviewed_cause")
    if reviewed in {"Normal", "Congestion", "Equipment", "Interference", "Mobility"}:
        return reviewed
    anomaly = record.get("anomalies") or {}
    exists = parse_flag(anomaly.get("exists", False))
    return CAUSE_MAP.get(anomaly.get("type"), "Unknown") if exists else "Normal"
