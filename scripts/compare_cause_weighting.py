"""Compare cause-classifier sample weighting with group CV on training data only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from ran_monitor_ai.features import extract_features, true_cause
from ran_monitor_ai.train import (DATA_PATH, _dataset_sha256,
                                 frozen_development_split_path,
                                 group_overlapping_windows,
                                 load_frozen_development_split, load_records,
                                 measured_feature_names,
                                 overlap_group_class_weights)


def _classifier(class_count: int, seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=220, max_depth=4, learning_rate=0.07,
        subsample=0.9, colsample_bytree=0.9, n_jobs=1,
        objective="multi:softprob", num_class=class_count,
        eval_metric="mlogloss", random_state=seed,
    )


def _summarize(labels: np.ndarray, predictions: np.ndarray,
               classes: list[str]) -> dict:
    ids = list(range(len(classes)))
    return {
        "samples": int(len(labels)),
        "macro_f1": float(f1_score(labels, predictions, labels=ids,
                                    average="macro", zero_division=0)),
        "per_class": {
            cause: {
                "support": int(np.sum(labels == class_id)),
                "precision": float(precision_score(
                    labels, predictions, labels=[class_id], average="micro", zero_division=0)),
                "recall": float(recall_score(
                    labels, predictions, labels=[class_id], average="micro", zero_division=0)),
                "f1": float(f1_score(
                    labels, predictions, labels=[class_id], average="micro", zero_division=0)),
            }
            for class_id, cause in enumerate(classes)
        },
    }


def compare_cause_weighting(data_path: Path = DATA_PATH, *, folds: int = 3,
                            seed: int = 42) -> dict:
    """Run out-of-fold comparison without reading validation or test labels for selection."""
    records = load_records(data_path)
    train_idx, _validation_idx, _test_idx, split_note = load_frozen_development_split(
        records, data_path,
    )
    causes = [true_cause(record) for record in records]
    classes = sorted({cause for cause in causes if cause != "Normal"})
    if len(classes) < 2:
        raise ValueError("原因加权对照至少需要两个故障类别。")
    class_to_id = {cause: index for index, cause in enumerate(classes)}
    groups = group_overlapping_windows(records)
    anomaly_indices = np.asarray([
        index for index in train_idx if causes[int(index)] in class_to_id
    ], dtype=int)
    labels = np.asarray([class_to_id[causes[index]] for index in anomaly_indices], dtype=int)
    anomaly_groups = groups[anomaly_indices]
    group_counts = {
        cause: len({int(anomaly_groups[row]) for row, label in enumerate(labels)
                    if label == class_id})
        for class_id, cause in enumerate(classes)
    }
    if min(group_counts.values(), default=0) < folds:
        raise ValueError(
            f"训练集最少类别只有 {min(group_counts.values())} 个独立组，"
            f"不足以执行 {folds} 折分组交叉验证。"
        )

    raw = pd.DataFrame([extract_features(record) for record in records])
    feature_names = measured_feature_names(list(raw.columns))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True,
                                    random_state=seed)
    outputs = {
        "window_weighted_baseline": np.full(len(labels), -1, dtype=int),
        "equal_cause_and_group_weight": np.full(len(labels), -1, dtype=int),
    }
    fold_reports = []
    split_iterator = splitter.split(np.zeros(len(labels)), labels, anomaly_groups)
    for fold_id, (fit_rows, held_rows) in enumerate(split_iterator):
        if (set(np.unique(labels[fit_rows])) != set(range(len(classes)))
                or set(np.unique(labels[held_rows])) != set(range(len(classes)))):
            raise ValueError(
                f"第 {fold_id + 1} 折没有覆盖全部原因类别；"
                "请补充独立故障组，或降低折数并重新审查协议。"
            )
        fit_indices = anomaly_indices[fit_rows]
        held_indices = anomaly_indices[held_rows]
        train_frame = raw.iloc[fit_indices][feature_names].replace(
            [np.inf, -np.inf], np.nan,
        )
        medians = train_frame.median().fillna(0.0)
        fit_x = train_frame.fillna(medians).fillna(0.0)
        held_x = raw.iloc[held_indices][feature_names].replace(
            [np.inf, -np.inf], np.nan,
        ).fillna(medians).fillna(0.0)
        weights = overlap_group_class_weights(
            anomaly_groups[fit_rows], labels[fit_rows],
        )
        baseline = _classifier(len(classes), seed=43 + fold_id)
        weighted = _classifier(len(classes), seed=43 + fold_id)
        baseline.fit(fit_x, labels[fit_rows])
        weighted.fit(fit_x, labels[fit_rows], sample_weight=weights)
        outputs["window_weighted_baseline"][held_rows] = baseline.predict(held_x)
        outputs["equal_cause_and_group_weight"][held_rows] = weighted.predict(held_x)
        fold_reports.append({
            "fold": fold_id + 1,
            "train_windows": int(len(fit_rows)),
            "heldout_windows": int(len(held_rows)),
            "heldout_groups": int(len(np.unique(anomaly_groups[held_rows]))),
        })

    if any(np.any(predictions < 0) for predictions in outputs.values()):
        raise RuntimeError("分组交叉验证没有为每个训练故障窗口生成一次样本外预测。")
    report = {
        "schema": "cause-training-weight-cv-v1",
        "dataset_sha256": _dataset_sha256(data_path),
        "split_manifest_sha256": hashlib.sha256(
            frozen_development_split_path(data_path).read_bytes()
        ).hexdigest(),
        "evaluation_scope": "training_partition_group_cv_only",
        "selection_scope": "validation_and_test_labels_not_used",
        "model": {
            "family": "XGBoost multiclass",
            "n_estimators": 220,
            "max_depth": 4,
            "learning_rate": 0.07,
            "seed_base": 43,
        },
        "split": {
            "method": "StratifiedGroupKFold",
            "folds": folds,
            "seed": seed,
            "training_anomaly_windows": int(len(anomaly_indices)),
            "training_cause_group_support": group_counts,
            "folds_detail": fold_reports,
            "development_split_note": split_note,
        },
        "comparison": {
            name: _summarize(labels, predictions, classes)
            for name, predictions in outputs.items()
        },
        "weighting": {
            "method": "equal total weight per cause and equal total weight per overlap group within cause",
            "normalization": "sample weights normalized to mean 1 in each training fold",
        },
        "limitations": [
            "The smallest cause has only a few independent groups; fold estimates are unstable.",
            "This compares training-window generalization only and does not establish future-site performance.",
        ],
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "cause-training-weight-cv.json")
    args = parser.parse_args()
    report = compare_cause_weighting(folds=args.folds, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
