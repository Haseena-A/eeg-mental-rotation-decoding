"""Score classification and regression under participant-independent splits.

Metrics are reported per fold. A single pooled number is also returned, together
with the standard deviation across folds, and is not treated as proof of
generalisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .model_training import (
    feature_frame,
    iter_group_splits,
    make_estimator,
    modelling_table,
    _preprocessor,
)


def _classification_fold(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
    scores = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        scores["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    return scores


def _regression_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _summarise(folds: list[dict[str, float]]) -> dict[str, Any]:
    table = pd.DataFrame(folds)
    summary = {}
    for column in table.columns:
        summary[column] = {
            "mean": float(table[column].mean()),
            "std": float(table[column].std(ddof=1)) if len(table) > 1 else 0.0,
        }
    return {"folds": folds, "summary": summary}


def evaluate_classification(frame: pd.DataFrame, kind: str, estimator_name: str, config: dict[str, Any]) -> dict[str, Any]:
    table = modelling_table(frame).dropna(subset=["correct", "participant_id"])
    table = table.loc[table["correct"].isin([0, 1])]
    features, numeric, categorical = feature_frame(table, kind)
    if features.shape[1] == 0 or len(table) < 4:
        return {"status": "not_run", "reason": "no usable rows or features", "kind": kind, "estimator": estimator_name}
    y = table["correct"].to_numpy(dtype=int)
    groups = table["participant_id"]
    splits, n_splits = iter_group_splits(groups, config["models"]["n_splits"], config["project"]["seed"])
    folds = []
    pooled_true: list[int] = []
    pooled_pred: list[int] = []
    pooled_prob: list[float] = []
    for fold, (train, test) in enumerate(splits, start=1):
        pipeline = Pipeline(
            [
                ("prepare", _preprocessor(numeric, categorical)),
                ("model", make_estimator(estimator_name, "classification", config)),
            ]
        )
        pipeline.fit(features.iloc[train], y[train])
        predicted = pipeline.predict(features.iloc[test])
        probability = None
        if hasattr(pipeline, "predict_proba"):
            probability = pipeline.predict_proba(features.iloc[test])[:, 1]
        scores = _classification_fold(y[test], predicted, probability)
        scores["fold"] = fold
        scores["n_test"] = int(len(test))
        scores["n_test_participants"] = int(groups.iloc[test].nunique())
        folds.append(scores)
        pooled_true.extend(y[test].tolist())
        pooled_pred.extend(predicted.tolist())
        if probability is not None:
            pooled_prob.extend(probability.tolist())
    result = _summarise(folds)
    result.update(
        {
            "status": "ok",
            "task": "classification",
            "kind": kind,
            "estimator": estimator_name,
            "n_splits": n_splits,
            "n_rows": int(len(table)),
            "n_participants": int(groups.nunique()),
            "class_counts": {str(key): int(value) for key, value in zip(*np.unique(y, return_counts=True))},
            "confusion_matrix": confusion_matrix(pooled_true, pooled_pred, labels=[0, 1]).tolist(),
            "pooled_true": pooled_true,
            "pooled_pred": pooled_pred,
            "pooled_prob": pooled_prob,
        }
    )
    return result


def evaluate_regression(frame: pd.DataFrame, kind: str, estimator_name: str, config: dict[str, Any]) -> dict[str, Any]:
    table = modelling_table(frame).dropna(subset=["response_time_s", "participant_id"])
    features, numeric, categorical = feature_frame(table, kind)
    if features.shape[1] == 0 or len(table) < 4:
        return {"status": "not_run", "reason": "no usable rows or features", "kind": kind, "estimator": estimator_name}
    y = table["response_time_s"].to_numpy(dtype=float)
    groups = table["participant_id"]
    splits, n_splits = iter_group_splits(groups, config["models"]["n_splits"], config["project"]["seed"])
    folds = []
    pooled_true: list[float] = []
    pooled_pred: list[float] = []
    for fold, (train, test) in enumerate(splits, start=1):
        pipeline = Pipeline(
            [
                ("prepare", _preprocessor(numeric, categorical)),
                ("model", make_estimator(estimator_name, "regression", config)),
            ]
        )
        pipeline.fit(features.iloc[train], y[train])
        predicted = pipeline.predict(features.iloc[test])
        scores = _regression_fold(y[test], predicted)
        scores["fold"] = fold
        scores["n_test"] = int(len(test))
        folds.append(scores)
        pooled_true.extend(y[test].tolist())
        pooled_pred.extend(predicted.tolist())
    result = _summarise(folds)
    result.update(
        {
            "status": "ok",
            "task": "regression",
            "kind": kind,
            "estimator": estimator_name,
            "n_splits": n_splits,
            "n_rows": int(len(table)),
            "n_participants": int(groups.nunique()),
            "pooled_true": pooled_true,
            "pooled_pred": pooled_pred,
        }
    )
    return result


def run_model_suite(frame: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for kind in ("eeg", "metadata", "combined"):
        for name in config["models"]["classification_estimators"]:
            results.append(evaluate_classification(frame, kind, name, config))
        for name in config["models"]["regression_estimators"]:
            results.append(evaluate_regression(frame, kind, name, config))
    return results
