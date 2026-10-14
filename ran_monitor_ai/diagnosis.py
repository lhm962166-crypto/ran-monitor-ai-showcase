"""Experimental multi-signal fault-cause ranking (kept separate from probabilities)."""
from __future__ import annotations

import numpy as np


def fit_cause_evidence_ranker(features, labels, classes, feature_names):
    """Fit robust KPI prototypes from training fault windows only."""
    matrix = np.asarray(features, dtype=float)
    targets = np.asarray(labels, dtype=int)
    names = [str(name) for name in feature_names]
    if (matrix.ndim != 2 or targets.ndim != 1 or len(matrix) != len(targets)
            or matrix.shape[1] != len(names) or not len(matrix)
            or not np.isfinite(matrix).all()):
        raise ValueError("多信号原因排序训练输入无效。")
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("多信号原因排序需要唯一类别。")
    if np.any((targets < 0) | (targets >= len(classes))):
        raise ValueError("多信号原因排序训练标签超出类别范围。")
    if set(np.unique(targets)) != set(range(len(classes))):
        raise ValueError("每个原因类别都需要训练窗口才能构造 KPI 证据原型。")

    center = np.median(matrix, axis=0)
    scale = np.median(np.abs(matrix - center), axis=0) * 1.4826
    scale = np.maximum(scale, np.maximum(np.abs(center) * 1e-6, 1e-6))
    prototypes = np.vstack([np.median(matrix[targets == index], axis=0)
                            for index in range(len(classes))])
    return {
        "method": "robust_training_kpi_prototypes",
        "classes": list(classes),
        "feature_names": names,
        "scale": scale.astype(float).tolist(),
        "prototypes": prototypes.astype(float).tolist(),
        "training_windows": int(len(matrix)),
        "training_scope": "training_anomaly_windows_only",
        "classifier_weight": 1.0,
        "mode": "shadow",
        "probability_claim": False,
    }


def prototype_distances(features, model):
    """Return robust scaled distances to each cause prototype."""
    matrix = np.asarray(features, dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    prototypes = np.asarray(model["prototypes"], dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if (matrix.ndim != 2 or matrix.shape[1] != len(scale)
            or prototypes.ndim != 2 or prototypes.shape[1] != matrix.shape[1]
            or not np.isfinite(matrix).all()):
        raise ValueError("多信号原因排序输入特征与训练原型不匹配。")
    return np.sqrt(np.mean(((matrix[:, None, :] - prototypes[None, :, :]) / scale) ** 2, axis=2))


def _rank_scores(values, *, descending=True):
    scores = np.asarray(values, dtype=float)
    order = np.argsort(-scores if descending else scores, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[np.arange(len(order))[:, None], order] = np.arange(scores.shape[1], 0, -1)
    return ranks / max(scores.shape[1], 1)


def fused_ranking_scores(classifier_probabilities, distances, classifier_weight: float):
    """Blend ordinal evidence only; output is a ranking score, never a probability."""
    probabilities = np.asarray(classifier_probabilities, dtype=float)
    distance_values = np.asarray(distances, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(1, -1)
    if distance_values.ndim == 1:
        distance_values = distance_values.reshape(1, -1)
    if (probabilities.shape != distance_values.shape or not len(probabilities)
            or not np.isfinite(probabilities).all() or not np.isfinite(distance_values).all()
            or not np.isfinite(classifier_weight) or not 0 <= classifier_weight <= 1):
        raise ValueError("多信号原因排序评分输入无效。")
    return (classifier_weight * _rank_scores(probabilities, descending=True)
            + (1.0 - classifier_weight) * _rank_scores(distance_values, descending=False))


def choose_classifier_weight(classifier_probabilities, distances, labels,
                             candidates=(0.0, 0.25, 0.5, 0.75, 1.0)) -> dict:
    """Choose a conservative rank blend on validation windows only."""
    probabilities = np.asarray(classifier_probabilities, dtype=float)
    distance_values = np.asarray(distances, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if (probabilities.ndim != 2 or distance_values.shape != probabilities.shape
            or targets.ndim != 1 or len(targets) != len(probabilities)
            or not len(targets) or np.any((targets < 0) | (targets >= probabilities.shape[1]))):
        raise ValueError("多信号排序验证输入无效。")
    valid = sorted({float(value) for value in candidates})
    if not valid or any(not 0 <= value <= 1 for value in valid):
        raise ValueError("排序权重候选必须位于 0–1。")
    reports = []
    for weight in valid:
        scores = fused_ranking_scores(probabilities, distance_values, weight)
        predictions = np.argmax(scores, axis=1)
        reports.append({"classifier_weight": weight,
                        "validation_top1_accuracy": float(np.mean(predictions == targets))})
    # Prefer the existing classifier when validation performance ties.
    best = max(reports, key=lambda row: (row["validation_top1_accuracy"], row["classifier_weight"]))
    return {"classifier_weight": best["classifier_weight"],
            "validation_samples": int(len(targets)), "candidates": reports,
            "selection": "highest_validation_top1_accuracy; ties prefer classifier"}


def evaluate_rank_matrix(scores, labels, *, candidate_budget=2):
    """Report fixed-budget candidate coverage, top-1 accuracy, and reciprocal rank."""
    values = np.asarray(scores, dtype=float)
    targets = np.asarray(labels, dtype=int)
    if (values.ndim != 2 or targets.ndim != 1 or len(values) != len(targets)
            or not len(targets) or np.any((targets < 0) | (targets >= values.shape[1]))
            or not np.isfinite(values).all() or not 1 <= candidate_budget <= values.shape[1]):
        raise ValueError("多信号排序评估输入无效。")
    order = np.argsort(-values, axis=1, kind="stable")
    positions = np.argmax(order == targets[:, None], axis=1) + 1
    return {
        "samples": int(len(targets)),
        "candidate_budget": int(candidate_budget),
        "candidate_coverage": float(np.mean(positions <= candidate_budget)),
        "top1_accuracy": float(np.mean(positions == 1)),
        "mean_reciprocal_rank": float(np.mean(1.0 / positions)),
    }


def rank_cause_candidates(classifier_probabilities, feature_row, model):
    """Create UI evidence with model probabilities and prototype scores kept distinct."""
    if not isinstance(model, dict):
        return None
    names = model["feature_names"]
    vector = np.asarray([float(feature_row[name]) for name in names], dtype=float)
    distances = prototype_distances(vector, model)[0]
    probabilities = np.asarray(classifier_probabilities, dtype=float).reshape(1, -1)
    weight = float(model.get("classifier_weight", 1.0))
    fused = fused_ranking_scores(probabilities, distances, weight)[0]
    classes = model["classes"]
    order = np.argsort(-fused, kind="stable")
    return {
        "mode": "shadow",
        "method": model["method"],
        "classifier_weight": weight,
        "probability_claim": False,
        "candidates": [{
            "rank": rank + 1,
            "cause": classes[int(index)],
            "classifier_probability": float(probabilities[0, index]),
            "kpi_prototype_distance": float(distances[index]),
            "ranking_score": float(fused[index]),
        } for rank, index in enumerate(order)],
    }
