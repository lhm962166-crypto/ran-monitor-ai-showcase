"""Experimental normal-only KPI relationship anomaly channel (shadow mode)."""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from .features import validate_window
from .settings import KPIS


def structure_feature_names() -> list[str]:
    names = []
    for lag in (0, 1, 8):
        names.extend(f"corr_lag{lag}_{left}->{right}"
                     for left_index, left in enumerate(KPIS)
                     for right in KPIS[left_index + 1:])
    names.extend(f"autocorr_lag{lag}_{name}" for lag in (1, 8) for name in KPIS)
    return names


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0)) if denominator else 0.0


def structural_features(record: dict) -> np.ndarray:
    """Measure instantaneous and directional lagged KPI relationships."""
    values = validate_window(record)
    rows = [np.asarray(values[name], dtype=float) for name in KPIS]
    result = []
    for lag in (0, 1, 8):
        for left_index, left in enumerate(rows):
            for right in rows[left_index + 1:]:
                result.append(_correlation(left[:len(left) - lag] if lag else left,
                                           right[lag:] if lag else right))
    for lag in (1, 8):
        result.extend(_correlation(row[:-lag], row[lag:]) for row in rows)
    output = np.asarray(result, dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("关系特征出现非有限数值。")
    return output


def fit_structure_detector(normal_train_records: list[dict], normal_validation_records: list[dict], *,
                           alpha: float = 0.01, max_prototypes: int = 3) -> tuple[dict, dict]:
    """Fit prototypes only on normal training windows; validation sets the shadow threshold."""
    if len(normal_train_records) < 10:
        raise ValueError("结构影子通道至少需要 10 个正常训练窗口。")
    train = np.vstack([structural_features(record) for record in normal_train_records])
    median = np.median(train, axis=0)
    mad = np.median(np.abs(train - median), axis=0) * 1.4826
    scale = np.maximum(mad, 0.08)
    normalized = np.clip((train - median) / scale, -8.0, 8.0)
    prototype_count = min(max_prototypes, max(1, len(train) // 50))
    clustering = KMeans(n_clusters=prototype_count, random_state=42, n_init=10)
    clustering.fit(normalized)
    validation = np.vstack([structural_features(record) for record in normal_validation_records])
    reference_scores = _distances(validation, median, scale, clustering.cluster_centers_)
    threshold = float(np.quantile(reference_scores, 1.0 - alpha, method="higher")) if len(reference_scores) else float("inf")
    bundle = {
        "feature_names": structure_feature_names(),
        "median": median.astype(float).tolist(),
        "scale": scale.astype(float).tolist(),
        "centers": clustering.cluster_centers_.astype(float).tolist(),
        "reference_scores": reference_scores.astype(float).tolist(),
        "threshold": threshold,
        "alpha": float(alpha),
        "prototype_count": int(prototype_count),
        "training_normal_windows": int(len(train)),
        "validation_normal_windows": int(len(validation)),
        "mode": "shadow",
        "coverage_claim": False,
    }
    report = {
        "status": "shadow_only",
        "prototype_count": int(prototype_count),
        "training_normal_windows": int(len(train)),
        "validation_normal_windows": int(len(validation)),
        "threshold": threshold,
        "alpha_target": float(alpha),
        "note": "只用正常训练窗口拟合关系原型；验证集经验阈值不代表现网误报率。",
    }
    return bundle, report


def _distances(values, median, scale, centers) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    normalized = np.clip((matrix - np.asarray(median)) / np.asarray(scale), -8.0, 8.0)
    differences = normalized[:, None, :] - np.asarray(centers)[None, :, :]
    return np.sqrt(np.mean(differences ** 2, axis=2)).min(axis=1)


def score_structure(record: dict, model: dict) -> dict:
    vector = structural_features(record)
    score = float(_distances(vector.reshape(1, -1), model["median"], model["scale"], model["centers"])[0])
    reference = np.asarray(model.get("reference_scores", []), dtype=float)
    tail = float((1 + np.sum(reference >= score)) / (len(reference) + 1)) if len(reference) else None
    names = model.get("feature_names", structure_feature_names())
    normalized = np.abs((vector - np.asarray(model["median"])) / np.asarray(model["scale"]))
    top = np.argsort(-normalized, kind="stable")[:5]
    return {
        "score": score,
        "threshold": float(model.get("threshold", float("inf"))),
        "shadow_flag": bool(score > float(model.get("threshold", float("inf")))),
        "empirical_tail_fraction": tail,
        "top_relation_shifts": [{"relation": names[int(index)], "robust_deviation": float(normalized[index])}
                                 for index in top],
        "mode": "shadow",
        "coverage_claim": False,
    }


def evaluate_shadow_scores(scores, labels, threshold: float) -> dict:
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    predicted = values > float(threshold)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, values)) if np.any(y == 1) else None,
        "promotion_status": "shadow_only",
    }
