"""Small-sample cause calibration and experimental APS-style candidate sets.

The bundled dataset contains overlapping windows and only a handful of
validation faults. Outputs from this module are explicitly experimental; they
must not be presented as formal conformal coverage guarantees.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def _probability_matrix(probabilities) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("原因校准需要有限的多分类概率矩阵。")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("原因模型输出须处于 0–1。")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("原因模型输出概率总和必须大于 0。")
    return values / totals


def temperature_scale(probabilities, temperature: float) -> np.ndarray:
    values = _probability_matrix(probabilities)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("温度参数必须是正的有限值。")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probabilities, labels) -> float:
    """Fit a single scalar on held-aside validation anomalies only."""
    values = _probability_matrix(probabilities)
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("温度校准标签与概率行数不一致。")
    if len(targets) < 2 or np.any((targets < 0) | (targets >= values.shape[1])):
        return 1.0

    def objective(log_temperature: float) -> float:
        scaled = temperature_scale(values, math.exp(float(log_temperature)))
        return float(-np.log(np.clip(scaled[np.arange(len(targets)), targets], 1e-12, 1.0)).mean())

    result = minimize_scalar(objective, bounds=(-2.0, 2.0), method="bounded")
    return float(math.exp(result.x)) if result.success and np.isfinite(result.fun) else 1.0


def aps_nonconformity(probabilities, labels) -> np.ndarray:
    """Return deterministic cumulative-probability APS scores (no random ties)."""
    values = _probability_matrix(probabilities)
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("APS 校准标签与概率行数不一致。")
    if np.any((targets < 0) | (targets >= values.shape[1])):
        raise ValueError("APS 校准标签超出原因类别范围。")
    scores = np.empty(len(targets), dtype=float)
    for index, row in enumerate(values):
        order = np.argsort(-row, kind="stable")
        rank = int(np.flatnonzero(order == targets[index])[0])
        scores[index] = float(row[order[:rank + 1]].sum())
    return scores


def fit_cause_calibration(probabilities, labels, *, temperature_probabilities=None,
                          temperature_labels=None, independent_groups: int = 0,
                          temperature_groups: int = 0, alpha: float = 0.10) -> dict:
    """Fit temperature on a disjoint group subset and APS threshold on calibration rows."""
    values = _probability_matrix(probabilities)
    targets = np.asarray(labels, dtype=int)
    if not 0 < alpha < 1:
        raise ValueError("候选集合的 alpha 必须在 0 和 1 之间。")
    if temperature_probabilities is None or temperature_labels is None:
        # Do not double-dip: when independent calibration groups are unavailable,
        # keep the identity transform and disclose the lack of separate data.
        temperature = 1.0
        temperature_sample_count = 0
    else:
        temperature_values = _probability_matrix(temperature_probabilities)
        temperature_targets = np.asarray(temperature_labels, dtype=int)
        temperature = fit_temperature(temperature_values, temperature_targets)
        temperature_sample_count = int(len(temperature_targets))
    scaled = temperature_scale(values, temperature)
    scores = np.sort(aps_nonconformity(scaled, targets))
    n = len(scores)
    rank = min(n, max(1, math.ceil((n + 1) * (1.0 - alpha)))) if n else 0
    threshold = float(scores[rank - 1]) if n else 1.0
    sufficient = n >= 100 and independent_groups >= 50
    return {
        "method": "temperature_scaling_plus_deterministic_aps",
        "temperature": temperature,
        "alpha": float(alpha),
        "aps_threshold": threshold,
        "calibration_samples": n,
        "independent_time_groups": int(independent_groups),
        "temperature_calibration_samples": temperature_sample_count,
        "temperature_calibration_groups": int(temperature_groups),
        "temperature_fitted": bool(temperature_sample_count),
        "status": "experimental" if sufficient else "insufficient_independent_calibration_data",
        "coverage_claim": False,
        "caveat": "校准样本较少或窗口存在依赖；候选集合仅作复核提示，不构成覆盖率保证。",
    }


def cause_prediction_set(probabilities, classes, calibration: dict) -> list[dict]:
    values = _probability_matrix(np.asarray(probabilities, dtype=float).reshape(1, -1))[0]
    if len(classes) != len(values):
        raise ValueError("原因类别与概率数量不一致。")
    calibrated = temperature_scale(values.reshape(1, -1), float(calibration.get("temperature", 1.0)))[0]
    threshold = float(calibration.get("aps_threshold", 1.0))
    order = np.argsort(-calibrated, kind="stable")
    selected, cumulative = [], 0.0
    for index in order:
        selected.append({
            "cause": str(classes[index]),
            "calibrated_score": float(calibrated[index]),
            "raw_score": float(values[index]),
        })
        cumulative += float(calibrated[index])
        if cumulative >= threshold:
            break
    return selected


def evaluate_prediction_sets(probabilities, labels, classes, calibration: dict) -> dict:
    values = _probability_matrix(probabilities)
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("预测集合评估标签与概率行数不一致。")
    sizes, covered, review = [], [], []
    for row, target in zip(values, targets):
        candidate_set = cause_prediction_set(row, classes, calibration)
        names = {item["cause"] for item in candidate_set}
        sizes.append(len(names))
        covered.append(str(classes[int(target)]) in names)
        ordered = np.sort(row)
        margin = float(ordered[-1] - ordered[-2])
        review.append(bool(len(names) != 1 or calibration.get("status") != "experimental"
                           or ordered[-1] < .6 or margin < .15))
    return {
        "samples": len(targets),
        "empirical_coverage": float(np.mean(covered)) if covered else None,
        "mean_set_size": float(np.mean(sizes)) if sizes else None,
        "singleton_rate": float(np.mean(np.asarray(sizes) == 1)) if sizes else None,
        "review_rate": float(np.mean(review)) if review else None,
        "interpretation": "开发/已查看测试集上的描述统计；不构成未来覆盖率保证。",
    }


def evaluate_candidate_ranking(probabilities, labels, *, budget: int = 2) -> dict:
    """Separate fixed-budget cause candidate coverage from within-list ranking."""
    values = _probability_matrix(probabilities)
    targets = np.asarray(labels, dtype=int)
    if targets.ndim != 1 or len(targets) != len(values):
        raise ValueError("候选排序评估标签与概率行数不一致。")
    if not isinstance(budget, int) or not 1 <= budget <= values.shape[1]:
        raise ValueError("候选预算须在原因类别数范围内。")
    if np.any((targets < 0) | (targets >= values.shape[1])):
        raise ValueError("候选排序评估标签超出原因类别范围。")
    order = np.argsort(-values, axis=1, kind="stable")
    covered = np.any(order[:, :budget] == targets[:, None], axis=1)
    top1 = order[:, 0] == targets
    coverage = float(np.mean(covered)) if len(targets) else None
    top1_accuracy = float(np.mean(top1)) if len(targets) else None
    conditional_ranking = (float(np.sum(top1 & covered) / np.sum(covered))
                           if np.any(covered) else None)
    return {
        "candidate_budget": budget,
        "samples": len(targets),
        "candidate_coverage": coverage,
        "candidate_miss_rate": float(1 - coverage) if coverage is not None else None,
        "top1_accuracy": top1_accuracy,
        "top1_accuracy_given_retrieved": conditional_ranking,
        "interpretation": "固定 top-k 仅评估四类故障候选与排序，不代表组件级根因定位。",
    }
