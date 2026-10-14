"""Development-only leave-one-cause-family-out and equal-FPR ablation."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .features import extract_features, true_cause
from .structure_detector import score_structure
from .train import DATA_PATH, frozen_development_split_path, load_records


def _threshold_at_fpr(normal_scores, target_fpr):
    values = np.asarray(normal_scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("等误报率阈值校准需要有效的正常验证评分。")
    if not np.isfinite(target_fpr) or not 0 <= target_fpr < 1:
        raise ValueError("目标误报率须在 [0, 1) 范围内。")
    candidates = np.unique(np.concatenate((values, [np.nextafter(values.max(), np.inf)])))
    eligible = [(float(threshold), float(np.mean(values >= threshold)))
                for threshold in candidates if np.mean(values >= threshold) <= target_fpr]
    return min(eligible, key=lambda row: row[0])


def _detection_metrics(labels, scores, threshold):
    y = np.asarray(labels, dtype=int)
    values = np.asarray(scores, dtype=float)
    predicted = values >= threshold
    negatives = y == 0
    return {
        "threshold": float(threshold),
        "windows": int(len(y)),
        "fault_windows": int(y.sum()),
        "false_positive_rate": float(np.sum(predicted & negatives) / max(int(negatives.sum()), 1)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, values)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, values)) if y.any() else None,
        "confusion": [[int(np.sum(~predicted & negatives)), int(np.sum(predicted & negatives))],
                      [int(np.sum(~predicted & (y == 1))), int(np.sum(predicted & (y == 1))) ]],
    }


def _upper_tail_evidence(reference, values):
    baseline = np.asarray(reference, dtype=float)
    current = np.asarray(values, dtype=float)
    return 1.0 - np.asarray([(1 + np.sum(baseline >= score)) / (len(baseline) + 1)
                             for score in current], dtype=float)


def leave_one_cause_out_experiment(records=None, bundle=None, *, target_fpr=0.05,
                                   cause_weights=(0.0, 0.25, 0.5, 0.75, 1.0)):
    """Compare supervised, structure-only, and fused channels at a fixed validation FPR.

    Fault-family samples in the test partition are held out from each fold's detector
    training and fusion-weight selection. The project demo has no event/site IDs, so
    the report is explicitly development-only.
    """
    if records is None:
        records = load_records(DATA_PATH)
    if bundle is None:
        raise ValueError("请传入同一数据切分训练得到的模型包。")
    classes = list(bundle["classes"])
    if set(classes) != {"Congestion", "Equipment", "Interference", "Mobility"}:
        raise ValueError("留一故障类别实验要求四类原因标签。")
    labels = np.asarray([true_cause(record) for record in records], dtype=str)
    matrix = pd.DataFrame([extract_features(record) for record in records])
    feature_names = list(bundle["kpi_features"])
    medians = pd.Series(bundle["medians"], dtype=float)
    clean = matrix.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    train_idx = np.asarray(bundle["train_indices"], dtype=int)
    val_idx = np.asarray(bundle["validation_indices"], dtype=int)
    test_idx = np.asarray(bundle["test_indices"], dtype=int)
    normal_train = train_idx[labels[train_idx] == "Normal"]
    normal_val_mask = labels[val_idx] == "Normal"
    normal_val_idx = val_idx[normal_val_mask]
    normal_test_idx = test_idx[labels[test_idx] == "Normal"]
    structure_model = bundle.get("structure_detector")
    if not isinstance(structure_model, dict):
        raise ValueError("缺少已训练的正常结构影子模型。")

    structure_scores = np.asarray([
        score_structure(records[int(index)], structure_model)["score"]
        for index in np.concatenate((val_idx, test_idx))
    ], dtype=float)
    val_structure = structure_scores[:len(val_idx)]
    test_structure = structure_scores[len(val_idx):]
    fold_results = {}
    load_group_reports = {}
    for held_out_cause in classes:
        known_train = train_idx[(labels[train_idx] != "Normal") & (labels[train_idx] != held_out_cause)]
        positive_count = len(known_train)
        if not positive_count:
            fold_results[held_out_cause] = {"status": "insufficient_training_faults"}
            continue
        fold_train = np.concatenate((normal_train, known_train))
        y_fold = (labels[fold_train] != "Normal").astype(int)
        detector = xgb.XGBClassifier(
            n_estimators=260, max_depth=4, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, n_jobs=4,
            scale_pos_weight=max(1, int(np.sum(y_fold == 0))) / positive_count,
            eval_metric="logloss", random_state=42,
        )
        detector.fit(clean.iloc[fold_train][feature_names], y_fold)
        val_xgb = detector.predict_proba(clean.iloc[val_idx][feature_names])[:, 1]
        test_xgb = detector.predict_proba(clean.iloc[test_idx][feature_names])[:, 1]

        val_normal_xgb = val_xgb[normal_val_mask]
        xgb_threshold, xgb_val_fpr = _threshold_at_fpr(val_normal_xgb, target_fpr)
        structure_val_normal = val_structure[normal_val_mask]
        structure_threshold, structure_val_fpr = _threshold_at_fpr(structure_val_normal, target_fpr)

        val_xgb_tail = _upper_tail_evidence(val_normal_xgb, val_xgb)
        test_xgb_tail = _upper_tail_evidence(val_normal_xgb, test_xgb)
        val_structure_tail = _upper_tail_evidence(structure_val_normal, val_structure)
        test_structure_tail = _upper_tail_evidence(structure_val_normal, test_structure)
        known_val_mask = (labels[val_idx] != "Normal") & (labels[val_idx] != held_out_cause)
        candidates = []
        for xgb_weight in cause_weights:
            fused_val = xgb_weight * val_xgb_tail + (1.0 - xgb_weight) * val_structure_tail
            fused_threshold, fused_val_fpr = _threshold_at_fpr(
                fused_val[normal_val_mask], target_fpr,
            )
            known_recall = (float(np.mean(fused_val[known_val_mask] >= fused_threshold))
                            if np.any(known_val_mask) else 0.0)
            candidates.append({"xgb_weight": float(xgb_weight), "threshold": fused_threshold,
                               "validation_false_positive_rate": fused_val_fpr,
                               "validation_known_fault_recall": known_recall})
        selected = max(candidates, key=lambda row: (
            row["validation_known_fault_recall"], row["xgb_weight"],
        ))

        held_out_test_mask = labels[test_idx] == held_out_cause
        selected_test_idx = np.flatnonzero((labels[test_idx] == "Normal") | held_out_test_mask)
        y_test = held_out_test_mask[selected_test_idx].astype(int)
        test_xgb_selected = test_xgb[selected_test_idx]
        test_structure_selected = test_structure[selected_test_idx]
        fused_test = (selected["xgb_weight"] * test_xgb_tail
                      + (1.0 - selected["xgb_weight"]) * test_structure_tail)
        fold_results[held_out_cause] = {
            "status": "evaluated",
            "training_fault_windows_excluding_family": int(positive_count),
            "test_fault_windows": int(held_out_test_mask.sum()),
            "test_normal_windows": int((labels[test_idx] == "Normal").sum()),
            "target_validation_false_positive_rate": float(target_fpr),
            "validation_false_positive_rate": {
                "xgboost": xgb_val_fpr,
                "structure_only": structure_val_fpr,
                "fused": selected["validation_false_positive_rate"],
            },
            "fusion_selection": {
                "selected_xgboost_weight": selected["xgb_weight"],
                "selection_basis": "validation recall on known fault families only; held-out family excluded",
                "candidate_results": candidates,
            },
            "test": {
                "xgboost_only": _detection_metrics(y_test, test_xgb_selected, xgb_threshold),
                "structure_only": _detection_metrics(y_test, test_structure_selected, structure_threshold),
                "fused": _detection_metrics(y_test, fused_test[selected_test_idx], selected["threshold"]),
            },
        }

        # Inspect whether normal traffic level changes false alarms at the same fixed thresholds.
        train_load = clean.iloc[normal_train]["RX_Bytes_mean"].to_numpy(dtype=float)
        load_edges = np.quantile(train_load, [.25, .5, .75])
        normal_test_local = np.flatnonzero(labels[test_idx] == "Normal")
        normal_test_indices = test_idx[normal_test_local]
        test_load = clean.iloc[normal_test_indices]["RX_Bytes_mean"].to_numpy(dtype=float)
        group_ids = np.digitize(test_load, load_edges, right=True)
        # Preserve validation-selected thresholds for all traffic slices.
        fused_test_all = selected["xgb_weight"] * test_xgb_tail + (1.0 - selected["xgb_weight"]) * test_structure_tail
        load_group_reports[held_out_cause] = {
            "cut_points_from_training_normal_rx_bytes_mean": [float(value) for value in load_edges],
            "groups": {},
        }
        for group_id, name in enumerate(("较低负载", "偏低负载", "偏高负载", "较高负载")):
            selected_normals = group_ids == group_id
            if not np.any(selected_normals):
                continue
            test_positions = normal_test_local[selected_normals]
            load_group_reports[held_out_cause]["groups"][name] = {
                "normal_windows": int(np.sum(selected_normals)),
                "xgboost_false_positive_rate": float(np.mean(test_xgb[test_positions] >= xgb_threshold)),
                "structure_false_positive_rate": float(np.mean(test_structure[test_positions] >= structure_threshold)),
                "fused_false_positive_rate": float(np.mean(fused_test_all[test_positions] >= selected["threshold"])),
            }

    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    split_path = frozen_development_split_path(DATA_PATH)
    split_digest = hashlib.sha256(split_path.read_bytes()).hexdigest() if split_path.exists() else None
    return {
        "schema": "leave-one-cause-out-development-report-v1",
        "dataset_sha256": digest,
        "split_manifest_sha256": split_digest,
        "model_id": bundle.get("metadata", {}).get("model_id"),
        "protocol": "reuse_frozen_development_overlap_group_split",
        "evaluation_scope": "development_only_not_future_site_or_independent_event_validation",
        "target_validation_false_positive_rate": float(target_fpr),
        "split_sizes": {"train": int(len(train_idx)), "validation": int(len(val_idx)),
                        "test": int(len(test_idx))},
        "holdout_family_results": fold_results,
        "normal_load_false_positive_analysis": load_group_reports,
        "limitations": [
            "数据缺少明确基站、数据流和故障事件身份；该切分不能证明独立事件泛化。",
            "类别样本较少；验证误报预算是经验值，测试误报率可能不同。",
            "正常负载分组使用训练正常窗口的 RX_Bytes 均值四分位点，仅作敏感性审计。",
        ],
    }
