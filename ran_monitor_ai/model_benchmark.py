"""Leakage-aware comparison of the supervised detector and an unsupervised baseline."""
from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from .evaluation import cluster_bootstrap_binary_metrics
from .features import extract_features, true_cause


def _median_inference_ms(predict, features, *, warmups: int = 3, repeats: int = 15) -> float:
    for _ in range(warmups):
        predict(features)
    durations = []
    for _ in range(repeats):
        started = perf_counter()
        predict(features)
        durations.append(perf_counter() - started)
    return float(median(durations) * 1000 / len(features))


def threshold_at_recall(labels, scores, target_recall: float = 0.90) -> dict:
    """Choose the strictest threshold that reaches the requested validation recall."""
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    if (y.ndim != 1 or values.ndim != 1 or len(y) == 0 or len(y) != len(values)
            or not set(np.unique(y)).issubset({0, 1}) or len(np.unique(y)) != 2):
        raise ValueError("基线阈值校准需要等长的一维正常/异常验证标签。")
    if not np.isfinite(values).all():
        raise ValueError("基线验证评分必须为有限数值。")
    if not np.isfinite(target_recall) or not 0 < target_recall <= 1:
        raise ValueError("目标召回率必须大于 0 且不超过 1。")
    precision, recall, thresholds = precision_recall_curve(y, values)
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    if len(eligible):
        index = int(eligible[-1])
        threshold = float(thresholds[index])
        met = True
    else:
        threshold = float(np.min(values))
        met = False
    predicted = values >= threshold
    return {
        "threshold": threshold,
        "target_recall": float(target_recall),
        "target_met": met,
        "validation_precision": float(precision_score(y, predicted, zero_division=0)),
        "validation_recall": float(recall_score(y, predicted, zero_division=0)),
        "validation_confusion": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def _score_report(labels, scores, threshold, groups, *, seed: int) -> dict:
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    predicted = values >= threshold
    return {
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, values)),
        "average_precision": float(average_precision_score(y, values)),
        "confusion": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "group_bootstrap_95ci": cluster_bootstrap_binary_metrics(
            y, values, threshold, groups, replicates=500, seed=seed,
        ),
    }


def compare_with_isolation_forest(records: list[dict], bundle: dict, *,
                                  target_recall: float = 0.90,
                                  n_estimators: int = 300, seed: int = 42) -> dict:
    """Compare XGBoost with normal-only Isolation Forest on the same grouped split.

    Feature transforms are fitted from training-normal scores. Both operating
    thresholds are selected on validation labels; the test set remains untouched.
    """
    if not records or not isinstance(bundle, dict):
        raise ValueError("模型比较需要非空记录和已训练模型包。")
    if not isinstance(n_estimators, int) or not 50 <= n_estimators <= 2000:
        raise ValueError("Isolation Forest 树数须为 50–2000。")
    if not isinstance(seed, int):
        raise ValueError("随机种子须为整数。")

    from .train import group_overlapping_windows

    rows = pd.DataFrame([extract_features(record) for record in records])
    feature_names = list(bundle["kpi_features"])
    medians = pd.Series(bundle["medians"], dtype=float)
    clean = rows.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    train_idx = np.asarray(bundle["train_indices"], dtype=int)
    validation_idx = np.asarray(bundle["validation_indices"], dtype=int)
    test_idx = np.asarray(bundle["test_indices"], dtype=int)
    labels = np.asarray([true_cause(record) != "Normal" for record in records], dtype=int)
    groups = group_overlapping_windows(records)[test_idx]

    validation_labels = labels[validation_idx]
    test_labels = labels[test_idx]
    normal_train_idx = train_idx[labels[train_idx] == 0]
    train_x = clean.iloc[normal_train_idx][feature_names]
    validation_x = clean.iloc[validation_idx][feature_names]
    test_x = clean.iloc[test_idx][feature_names]

    forest = IsolationForest(
        n_estimators=n_estimators, max_samples="auto", contamination="auto",
        n_jobs=4, random_state=seed,
    )
    started = perf_counter()
    forest.fit(train_x)
    fit_seconds = perf_counter() - started

    raw_train = -forest.decision_function(train_x)
    location = float(np.median(raw_train))
    scale = max(float(np.quantile(raw_train, 0.75) - np.quantile(raw_train, 0.25)), 1e-6)

    def forest_scores(frame):
        raw = -forest.decision_function(frame)
        return expit((raw - location) / scale)

    validation_forest_scores = forest_scores(validation_x)
    test_forest_scores = forest_scores(test_x)
    forest_inference_ms = _median_inference_ms(forest_scores, test_x)
    forest_calibration = threshold_at_recall(validation_labels, validation_forest_scores,
                                             target_recall)

    detector = bundle["detector"]
    validation_xgb_scores = detector.predict_proba(validation_x)[:, 1]
    test_xgb_scores = detector.predict_proba(test_x)[:, 1]
    xgb_inference_ms = _median_inference_ms(
        lambda frame: detector.predict_proba(frame)[:, 1], test_x,
    )
    xgb_calibration = threshold_at_recall(validation_labels, validation_xgb_scores,
                                          target_recall)

    model_id = bundle.get("metadata", {}).get("model_id", "unknown")
    return {
        "protocol": "same overlap-group train/validation/test split; both thresholds calibrated on validation recall",
        "target_validation_recall": target_recall,
        "model_id": model_id,
        "split_sizes": {"train": int(len(train_idx)), "validation": int(len(validation_idx)),
                        "test": int(len(test_idx)), "test_overlap_groups": int(len(np.unique(groups)))},
        "models": {
            "xgboost_supervised": {
                "fit_scope": "supervised detector trained on training split",
                "test_inference_ms_per_window": xgb_inference_ms,
                "inference_benchmark_repeats": 15,
                "calibration": xgb_calibration,
                "test": _score_report(test_labels, test_xgb_scores, xgb_calibration["threshold"],
                                      groups, seed=seed),
            },
            "isolation_forest_normal_only": {
                "fit_scope": "normal training windows only; no fault labels used during fit",
                "fit_normal_windows": int(len(normal_train_idx)),
                "n_estimators": n_estimators,
                "fit_seconds": float(fit_seconds),
                "test_inference_ms_per_window": forest_inference_ms,
                "inference_benchmark_repeats": 15,
                "calibration": forest_calibration,
                "test": _score_report(test_labels, test_forest_scores,
                                      forest_calibration["threshold"], groups, seed=seed),
            },
        },
        "scope_note": "单一分层组留出和小型公开数据子集；结果用于模型选择实验，不代表现网成绩。",
    }
