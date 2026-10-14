"""Metrics for the complete detector-then-classifier workflow."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)


def anomaly_origin(record: dict) -> str:
    """Read provenance metadata without inferring or inventing a sample origin."""
    labels = record.get("labels") or {}
    anomaly = record.get("anomalies") or {}
    raw = (record.get("anomaly_origin") or labels.get("anomaly_origin")
           or labels.get("origin") or anomaly.get("origin"))
    if raw is None or not str(raw).strip():
        return "unknown"
    value = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "synthetic": "synthetic", "synth": "synthetic", "generated": "synthetic",
        "合成": "synthetic", "模拟": "synthetic",
        "controlled_real": "controlled_real", "real_controlled": "controlled_real",
        "controlled": "controlled_real", "measured": "controlled_real",
        "受控真实": "controlled_real", "真实实测": "controlled_real",
        "real": "real", "production": "real", "field": "real", "现网": "real",
    }
    return aliases.get(value, value[:80] or "unknown")


def evaluate_by_origin(true_causes, detector_scores, threshold: float,
                       origins) -> dict:
    """Report held-out detection metrics by declared data origin."""
    causes = np.asarray(true_causes, dtype=str)
    scores = np.asarray(detector_scores, dtype=float)
    provenance = np.asarray(origins, dtype=str)
    if (causes.ndim != 1 or scores.ndim != 1 or provenance.ndim != 1
            or len(causes) == 0 or len(causes) != len(scores)
            or len(causes) != len(provenance)):
        raise ValueError("来源审计输入须是一维、非空且长度一致。")
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("来源审计评分和阈值须为 0–1 的有限数值。")
    if not set(causes).issubset({"Normal", "Congestion", "Equipment", "Interference", "Mobility"}):
        raise ValueError("来源审计发现未知窗口标签。")

    rows = []
    predicted = scores >= threshold
    actual = causes != "Normal"
    for origin in sorted(set(provenance)):
        selected = provenance == origin
        positive = actual[selected]
        flagged = predicted[selected]
        tp = int(np.sum(positive & flagged))
        fp = int(np.sum(~positive & flagged))
        fn = int(np.sum(positive & ~flagged))
        tn = int(np.sum(~positive & ~flagged))
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        fpr = fp / (fp + tn) if fp + tn else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else 0.0)
        rows.append({
            "origin": origin,
            "windows": int(selected.sum()),
            "normal_windows": int((~positive).sum()),
            "anomaly_windows": int(positive.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "confusion": [[tn, fp], [fn, tp]],
        })
    known = [row for row in rows if row["origin"] != "unknown"]
    return {
        "status": "ready" if known else "missing_origin",
        "known_origins": sorted(row["origin"] for row in known),
        "unknown_windows": int(np.sum(provenance == "unknown")),
        "groups": rows,
    }


def cluster_bootstrap_binary_metrics(true_labels, detector_scores, threshold: float,
                                     group_ids, *, replicates: int = 500,
                                     seed: int = 42) -> dict:
    """Estimate metric uncertainty by resampling independent overlap groups, not windows."""
    labels = np.asarray(true_labels)
    scores = np.asarray(detector_scores, dtype=float)
    groups = np.asarray(group_ids)
    if (labels.ndim != 1 or scores.ndim != 1 or groups.ndim != 1
            or len(labels) == 0 or len(labels) != len(scores) or len(labels) != len(groups)):
        raise ValueError("分组自助法输入须是一维、非空且长度一致。")
    if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) != 2:
        raise ValueError("分组自助法需要同时包含正常和异常窗口。")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("分组自助法评分须为 0–1 的有限数值。")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("分组自助法阈值须为 0–1 的有限数值。")
    if not isinstance(replicates, int) or not 100 <= replicates <= 10000:
        raise ValueError("分组自助法重复次数须为 100–10000 的整数。")

    group_values = np.unique(groups)
    if len(group_values) < 2:
        raise ValueError("分组自助法至少需要 2 个不重叠时间组。")
    members = {group: np.flatnonzero(groups == group) for group in group_values}
    positive = labels.astype(int)
    predicted = scores >= threshold

    def measure(indices):
        actual = positive[indices]
        flags = predicted[indices]
        tp = int(np.sum(actual & flags))
        fp = int(np.sum((1 - actual) & flags))
        fn = int(np.sum(actual & ~flags))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = float(roc_auc_score(actual, scores[indices])) if len(np.unique(actual)) == 2 else None
        ap = float(average_precision_score(actual, scores[indices])) if np.any(actual) else None
        return {"precision": precision, "recall": recall, "f1": f1,
                "roc_auc": auc, "average_precision": ap}

    point = measure(np.arange(len(labels)))
    samples = {name: [] for name in point}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        draw = rng.choice(group_values, size=len(group_values), replace=True)
        indices = np.concatenate([members[group] for group in draw])
        row = measure(indices)
        for name, value in row.items():
            if value is not None and np.isfinite(value):
                samples[name].append(value)
    return {
        "method": "overlap_group_cluster_bootstrap_percentile",
        "confidence_level": 0.95,
        "independent_groups": int(len(group_values)),
        "replicates": replicates,
        "seed": seed,
        "metrics": {
            name: {
                "estimate": float(value),
                "ci95": ([float(np.quantile(samples[name], 0.025)),
                          float(np.quantile(samples[name], 0.975))]
                         if samples[name] else None),
                "valid_replicates": len(samples[name]),
            }
            for name, value in point.items()
        },
    }


def calibration_diagnostics(true_labels, probabilities, *, bins: int = 5) -> dict:
    """Report Brier score and top-label ECE for binary or multiclass probabilities."""
    labels = np.asarray(true_labels)
    values = np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("概率校准评估需要非空的一维真实标签。")
    if not isinstance(bins, int) or not 2 <= bins <= 20:
        raise ValueError("概率校准分箱数须为 2–20。")
    if values.ndim == 1:
        if len(values) != len(labels) or not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("二分类概率校准输入长度须一致，标签只能是 0 或 1。")
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("二分类概率须为 0–1 的有限数值。")
        predicted = (values >= 0.5).astype(int)
        confidence = np.where(predicted == 1, values, 1 - values)
        correct = predicted == labels.astype(int)
        brier = float(np.mean((values - labels.astype(float)) ** 2))
        class_count = 2
    elif values.ndim == 2:
        if len(values) != len(labels) or values.shape[1] < 2:
            raise ValueError("多分类概率须与标签等长且至少含两个类别。")
        if (not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)):
            raise ValueError("多分类概率须为有限数值且每行总和为 1。")
        if (not np.issubdtype(labels.dtype, np.integer)
                or np.any((labels < 0) | (labels >= values.shape[1]))):
            raise ValueError("多分类真实标签须为有效的整数类别编号。")
        class_count = int(values.shape[1])
        predicted = np.argmax(values, axis=1)
        confidence = np.max(values, axis=1)
        correct = predicted == labels
        one_hot = np.eye(class_count)[labels.astype(int)]
        brier = float(np.mean(np.sum((values - one_hot) ** 2, axis=1)))
    else:
        raise ValueError("概率校准输入须为一维二分类分数或二维多分类概率。")

    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        selected = bin_ids == index
        count = int(np.sum(selected))
        if not count:
            continue
        accuracy = float(np.mean(correct[selected]))
        mean_confidence = float(np.mean(confidence[selected]))
        ece += count / len(labels) * abs(accuracy - mean_confidence)
        reliability.append({
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "count": count,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
        })
    return {
        "samples": int(len(labels)),
        "classes": class_count,
        "brier_score": brier,
        "top_label_ece": float(ece),
        "accuracy": float(np.mean(correct)),
        "mean_confidence": float(np.mean(confidence)),
        "bins": reliability,
    }


def evaluate_two_stage(true_causes, detector_scores, cause_predictions, threshold: float,
                       cause_classes: list[str]) -> dict:
    """Score the deployed workflow; the cause classifier is used only on detections."""
    labels = np.asarray(true_causes, dtype=str)
    scores = np.asarray(detector_scores, dtype=float)
    causes = np.asarray(cause_predictions, dtype=str)
    classes = list(cause_classes)
    if labels.ndim != 1 or scores.ndim != 1 or causes.ndim != 1:
        raise ValueError("端到端评估输入须是一维序列。")
    if len(labels) == 0 or len(labels) != len(scores) or len(labels) != len(causes):
        raise ValueError("端到端评估输入长度须相同且至少包含一个窗口。")
    if not classes or len(set(classes)) != len(classes) or "Normal" in classes:
        raise ValueError("故障类别须唯一且不能包含 Normal。")
    if not np.isfinite(scores).all() or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("端到端评估的异常评分和阈值须为 0–1 的有限数值。")
    names = ["Normal", *classes]
    if not set(labels).issubset(names) or not set(causes).issubset(classes):
        raise ValueError("端到端评估发现未知的窗口标签或故障类别。")

    detected = scores >= threshold
    predicted = np.where(detected, causes, "Normal")
    matrix = confusion_matrix(labels, predicted, labels=names)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=names, zero_division=0
    )
    fault_f1 = f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)
    detected_faults = int(np.sum(detected & (labels != "Normal")))
    correctly_classified_faults = int(np.trace(matrix[1:, 1:]))
    return {
        "classes": names,
        "confusion": matrix.tolist(),
        "accuracy": float(np.mean(labels == predicted)),
        "macro_f1": float(f1_score(labels, predicted, labels=names, average="macro", zero_division=0)),
        "fault_macro_f1": float(fault_f1),
        "detected_faults": detected_faults,
        "correctly_classified_faults": correctly_classified_faults,
        "misclassified_detected_faults": detected_faults - correctly_classified_faults,
        "missed_faults": int(np.sum(~detected & (labels != "Normal"))),
        "false_alarms": int(np.sum(detected & (labels == "Normal"))),
        "per_class": {
            name: {"precision": float(precision[index]), "recall": float(recall[index]),
                   "f1": float(f1[index]), "support": int(support[index])}
            for index, name in enumerate(names)
        },
    }
