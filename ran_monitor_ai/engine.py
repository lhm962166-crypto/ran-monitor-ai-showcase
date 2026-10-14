"""Two-stage inference and transparent operational severity rules."""
from __future__ import annotations

import uuid

from .runtime import configure_threads
configure_threads()

import numpy as np
import pandas as pd

from .features import extract_features, window_identity
from .diagnosis import rank_cause_candidates
from .settings import ACTION, CAUSE_ZH, KPIS, POLICY_ZH
from .structure_detector import score_structure
from .uncertainty import cause_prediction_set, temperature_scale


def _context(record: dict) -> tuple[str, dict]:
    labels = record.get("labels") or {}
    zone, app = labels.get("zone"), labels.get("application")
    key = f"zone_{zone}|application_{app}" if zone and app else "__global__"
    return key, {"zone": zone, "application": app,
                 "mobility": labels.get("mobility"), "congestion": labels.get("congestion")}


def _baseline(bundle: dict, key: str) -> dict:
    return bundle["baselines"].get(key, bundle["baselines"]["__global__"])


def baseline_for_record(record: dict, bundle: dict) -> tuple[dict, str]:
    key, _ = _context(record)
    exists = key in bundle["baselines"] and key != "__global__"
    return _baseline(bundle, key), "同区域与业务场景" if exists else "全局正常窗口"


def _clean_row(record: dict, bundle: dict) -> pd.Series:
    row = pd.Series(extract_features(record), dtype=float).reindex(bundle["feature_names"])
    return row.replace([np.inf, -np.inf], np.nan).fillna(pd.Series(bundle["medians"])).fillna(0.)


def bundle_for_policy(bundle: dict, policy: str = "balanced") -> dict:
    if policy not in POLICY_ZH:
        raise ValueError("检测策略无效。")
    profiles = bundle.get("threshold_profiles", {})
    if policy != "balanced" and policy not in profiles:
        raise ValueError("当前模型尚未生成该检测策略，请重新训练。")
    threshold = profiles.get(policy, {}).get("threshold", bundle["threshold"])
    return bundle | {"threshold": float(threshold), "detection_policy": policy}


def explain_record(record: dict, bundle: dict) -> dict:
    """Exact tree contributions to the detector's raw log-odds score."""
    import xgboost as xgb
    row = _clean_row(record, bundle)
    names = bundle["kpi_features"]
    matrix = xgb.DMatrix(row[names].to_numpy().reshape(1, -1), feature_names=names)
    booster = bundle["detector"].get_booster()
    contributions = booster.predict(matrix, pred_contribs=True)[0]
    feature_rows = [{"feature": name, "value": float(row[name]), "contribution": float(contribution)}
                    for name, contribution in zip(names, contributions[:-1])]
    feature_rows.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    grouped = {}
    for item in feature_rows:
        family = next((name for name in KPIS if item["feature"].startswith(name + "_")), "关联与组合特征")
        grouped[family] = grouped.get(family, 0.0) + item["contribution"]
    return {"base_value": float(contributions[-1]), "raw_score": float(np.sum(contributions)),
            "features": feature_rows,
            "groups": [{"name": name, "contribution": contribution}
                       for name, contribution in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)]}


def score_record(record: dict, bundle: dict) -> dict:
    row = _clean_row(record, bundle)
    probability = float(bundle["detector"].predict_proba(
        row[bundle["kpi_features"]].to_numpy().reshape(1, -1)
    )[0, 1])
    cause_proba = None
    if probability >= bundle["threshold"]:
        names = bundle.get("cause_features", bundle["feature_names"])
        cause_proba = bundle["classifier"].predict_proba(row[names].to_numpy().reshape(1, -1))[0]
    return _build_result(record, row, probability, cause_proba, bundle,
                         request_id=f"REQ-{uuid.uuid4().hex.upper()}")


def score_records(records: list[dict], bundle: dict,
                  feature_rows: pd.DataFrame | None = None) -> list[dict]:
    """Predict a batch once per model while preserving the input window order."""
    if not records:
        return []
    if feature_rows is None:
        feature_rows = pd.DataFrame([extract_features(record) for record in records])
    elif len(feature_rows) != len(records):
        raise ValueError("特征行数与窗口数不一致。")
    rows = feature_rows.reindex(columns=bundle["feature_names"]).astype(float)
    medians = pd.Series(bundle["medians"])
    rows = rows.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    probabilities = bundle["detector"].predict_proba(rows[bundle["kpi_features"]].to_numpy())[:, 1]
    abnormal_indices = np.flatnonzero(probabilities >= bundle["threshold"])
    request_id = f"REQ-{uuid.uuid4().hex.upper()}"
    cause_rows = {}
    if len(abnormal_indices):
        cause_features = bundle.get("cause_features", bundle["feature_names"])
        predictions = bundle["classifier"].predict_proba(
            rows.iloc[abnormal_indices][cause_features].to_numpy()
        )
        cause_rows = dict(zip(abnormal_indices, predictions))
    return [
        _build_result(record, rows.iloc[index], float(probabilities[index]),
                      cause_rows.get(index), bundle, request_id=request_id)
        for index, record in enumerate(records)
    ]


def _build_result(record: dict, row: pd.Series, probability: float,
                  cause_proba: np.ndarray | None, bundle: dict, *, request_id: str) -> dict:
    detected = probability >= bundle["threshold"]
    labels = record.get("labels") or {}
    metadata = record.get("metadata") or {}
    stream_id = record.get("stream_id") or labels.get("stream_id") or metadata.get("stream_id")
    stream_id = str(stream_id).strip()[:200] if stream_id not in (None, "") else None
    window_id = window_identity(record)
    result = {"sample_id": record.get("sample_id"), "detected": detected,
              "score": probability, "threshold": float(bundle["threshold"]),
              "model_id": bundle.get("metadata", {}).get("model_id"),
              "detection_policy": bundle.get("detection_policy", "balanced"),
              "request_id": request_id, "window_id": window_id, "stream_id": stream_id,
              "start_time": record.get("start_time"), "end_time": record.get("end_time"),
              "alert": None}
    structure_model = bundle.get("structure_detector")
    shadow = score_structure(record, structure_model) if isinstance(structure_model, dict) else None
    result["structure_shadow"] = shadow
    if not detected:
        return result

    class_index = int(np.argmax(cause_proba))
    cause = bundle["classes"][class_index]
    confidence = float(cause_proba[class_index])
    ordered_probabilities = np.sort(cause_proba)
    confidence_margin = float(ordered_probabilities[-1] - ordered_probabilities[-2])
    key, context = _context(record)
    baseline = _baseline(bundle, key)
    affected = []
    for name in KPIS:
        col = f"{name}_mean"
        expected = baseline.get(col)
        if not expected:
            continue
        std = float(expected["std"])
        z = (float(row[col]) - float(expected["mean"])) / std if std > 0 else 0.0
        if abs(z) >= 3:
            affected.append({"name": name, "current": round(float(row[col]), 3),
                             "baseline": round(float(expected["mean"]), 3),
                             "zscore": round(z, 2),
                             "direction": "高于基线" if z > 0 else "低于基线"})
    affected.sort(key=lambda item: abs(item["zscore"]), reverse=True)
    rx = baseline.get("RX_Bytes_mean", {})
    rx_base = float(rx.get("mean", 0)) or 1.0
    drop = max(0.0, (1.0 - float(row["RX_Bytes_mean"]) / rx_base) * 100)
    n = len(affected)
    # Severity represents observed impact, not the classifier's suspected cause.
    if n >= 4:
        severity = "P1"
        severity_reason = f"{n} 项 KPI 偏离正常基线达到 3σ，触发紧急告警。"
    elif drop > 80:
        severity = "P1"
        severity_reason = f"接收流量较正常基线下降 {drop:.1f}%，触发紧急告警。"
    elif any(item["name"].endswith("BLER") and abs(item["zscore"]) > 5 for item in affected):
        severity = "P2"
        severity_reason = "BLER 偏离正常基线超过 5σ，触发严重告警。"
    elif drop > 50 or n >= 2:
        severity = "P2"
        severity_reason = (f"接收流量较正常基线下降 {drop:.1f}%。" if drop > 50
                           else f"{n} 项 KPI 偏离正常基线达到 3σ。")
    elif n == 1:
        severity = "P3"
        severity_reason = "1 项 KPI 偏离正常基线达到 3σ，建议排查并观察。"
    else:
        severity = "P4"
        severity_reason = "检测到统计异常，当前未观察到明确 KPI 影响证据。"
    calibration = bundle.get("cause_calibration", {})
    evidence_ranking = rank_cause_candidates(
        cause_proba, row, bundle.get("cause_evidence_ranker"),
    )
    candidate_rows = cause_prediction_set(cause_proba, bundle["classes"], calibration)
    calibrated_probabilities = temperature_scale(
        np.asarray(cause_proba, dtype=float).reshape(1, -1),
        float(calibration.get("temperature", 1.0)),
    )[0]
    confidence_level = "高" if confidence >= .8 else "中" if confidence >= .6 else "低"
    action, team = ACTION[cause]
    candidate_causes = [
        {"cause": item["cause"], "cause_zh": CAUSE_ZH.get(item["cause"], item["cause"]),
         "calibrated_score": round(float(item["calibrated_score"]), 4),
         "raw_score": round(float(item["raw_score"]), 4)}
        for item in candidate_rows
    ]
    cause_set_status = calibration.get("status", "unavailable")
    requires_review = (len(candidate_causes) != 1 or cause_set_status != "experimental"
                       or confidence < .6 or confidence_margin < .15)
    result["alert"] = {
        "alert_id": f"ALT-{uuid.uuid4().hex.upper()}",
        "request_id": request_id,
        "window_id": window_id,
        "stream_id": stream_id,
        "start_time": record.get("start_time"),
        "end_time": record.get("end_time"),
        "severity": severity,
        "severity_reason": severity_reason,
        "cause": cause,
        "cause_zh": CAUSE_ZH[cause],
        "confidence": round(confidence, 3),
        "confidence_level": confidence_level,
        "confidence_kind": "raw_uncalibrated_model_score",
        "cause_probabilities": {name: round(float(value), 4)
                                for name, value in zip(bundle["classes"], cause_proba)},
        "calibrated_cause_probabilities": {name: round(float(value), 4)
                                            for name, value in zip(bundle["classes"], calibrated_probabilities)},
        "cause_candidates": candidate_causes,
        "cause_ranking_shadow": evidence_ranking,
        "cause_set_method": calibration.get("method", "none"),
        "cause_set_status": cause_set_status,
        "cause_calibration_samples": int(calibration.get("calibration_samples", 0)),
        "cause_calibration_groups": int(calibration.get("independent_time_groups", 0)),
        "temperature_calibration_samples": int(calibration.get("temperature_calibration_samples", 0)),
        "temperature_calibration_groups": int(calibration.get("temperature_calibration_groups", 0)),
        "temperature_fitted": bool(calibration.get("temperature_fitted", False)),
        "cause_set_caveat": calibration.get("caveat", "无校准数据；候选仅按模型排序。"),
        "confidence_margin": round(confidence_margin, 4),
        "requires_review": requires_review,
        "model_id": bundle.get("metadata", {}).get("model_id"),
        "detection_threshold": float(bundle["threshold"]),
        "detection_policy": bundle.get("detection_policy", "balanced"),
        "baseline_scope": baseline_for_record(record, bundle)[1],
        "baseline_sample_count": baseline.get("__metadata__", {}).get("normal_windows"),
        "affected_kpis": affected,
        "throughput_drop_pct": round(drop, 2),
        "structure_shadow": shadow,
        "evidence_package": {
            "detection": {"score": round(probability, 5), "threshold": float(bundle["threshold"]),
                          "policy": bundle.get("detection_policy", "balanced")},
            "impact_kpis": [item["name"] for item in affected],
            "throughput_drop_pct": round(drop, 2),
            "cause_candidates": candidate_causes,
            "cause_ranking_shadow": evidence_ranking,
            "structure_shadow_only": shadow,
            "limitations": ["分类贡献是相关性证据，不代表因果根因。",
                            "结构异常影子分数不参与当前告警判定。"],
        },
        "context": context,
        "action": action,
        "team": team,
    }
    return result
