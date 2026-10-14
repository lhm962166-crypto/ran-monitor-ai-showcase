"""Compare root-cause models using only the existing training partition.

Validation and held-out test windows are excluded from model selection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ran_monitor_ai.cli import atomic_output
from ran_monitor_ai.features import extract_features, true_cause
from experimental_features import extract_temporal_features
from ran_monitor_ai.train import group_overlapping_windows, load_records, measured_feature_names, train


def main():
    bundle, _ = train()
    all_records = load_records()
    records = [all_records[index] for index in bundle["train_indices"]]
    rows = pd.DataFrame([extract_features(record) | extract_temporal_features(record) for record in records])
    coarse_labels = np.array([true_cause(record) for record in records])
    classes = bundle["classes"]
    labels = np.array([classes.index(cause) if cause in classes else -1 for cause in coarse_labels])
    groups = group_overlapping_windows(records)
    baseline = bundle["cause_features"]
    temporal = measured_feature_names(list(rows.columns))
    experiments = []
    started = perf_counter()
    folds = []
    for seed in (42, 43, 44):
        for train_idx, val_idx in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(rows, coarse_labels, groups):
            train_idx = train_idx[labels[train_idx] >= 0]
            val_idx = val_idx[labels[val_idx] >= 0]
            if set(labels[train_idx]) != set(range(len(classes))) or not len(val_idx):
                continue
            folds.append((train_idx, val_idx))
    for feature_set, names in (("statistical", baseline), ("temporal", temporal)):
        for depth in (2, 3, 4):
            for balanced in (False, True):
                scores, mobility_scores, pooled_labels, pooled_predictions = [], [], [], []
                for train_idx, val_idx in folds:
                    model = xgb.XGBClassifier(n_estimators=220, max_depth=depth, learning_rate=.07,
                                              subsample=.9, colsample_bytree=.9, objective="multi:softprob",
                                              num_class=len(classes), eval_metric="mlogloss", random_state=43, n_jobs=2)
                    weights = None
                    if balanced:
                        counts = np.bincount(labels[train_idx], minlength=len(classes))
                        weights = len(train_idx) / (len(classes) * counts[labels[train_idx]])
                    model.fit(rows.iloc[train_idx][names], labels[train_idx], sample_weight=weights)
                    prediction = model.predict(rows.iloc[val_idx][names])
                    pooled_labels.extend(labels[val_idx].tolist())
                    pooled_predictions.extend(prediction.tolist())
                    scores.append(float(f1_score(labels[val_idx], prediction, labels=np.arange(len(classes)), average="macro", zero_division=0)))
                    mobility_scores.append(float(f1_score(labels[val_idx] == classes.index("Mobility"),
                                                         prediction == classes.index("Mobility"), zero_division=0)))
                report = {"features": feature_set, "depth": depth, "balanced": balanced,
                          "folds": len(scores), "macro_f1_mean": float(np.mean(scores)),
                          "macro_f1_std": float(np.std(scores)), "mobility_f1_mean": float(np.mean(mobility_scores)),
                          "pooled_macro_f1": float(f1_score(pooled_labels, pooled_predictions, labels=np.arange(len(classes)),
                                                            average="macro", zero_division=0)),
                          "fold_scores": scores}
                experiments.append(report)
                print(json.dumps({key: value for key, value in report.items() if key != "fold_scores"}), flush=True)
                atomic_output(ROOT / "artifacts" / "temporal-comparison.json", json.dumps({
                    "selection_basis": "3 repeated five-fold group splits within the training partition only",
                    "training_windows": len(records), "elapsed_seconds": perf_counter() - started,
                    "experiments": sorted(experiments, key=lambda result: result["pooled_macro_f1"], reverse=True),
                }, ensure_ascii=False, indent=2).encode("utf-8"))


if __name__ == "__main__":
    main()
