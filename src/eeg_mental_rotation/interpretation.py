"""Held-out permutation importance.

Importance is the drop in balanced accuracy, or the rise in MAE, when a feature
is shuffled inside a test partition whose participants were not used for fitting.
It is not evidence of a neural mechanism.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .model_evaluation import _classification_fold, _regression_fold
from .model_training import feature_frame, iter_group_splits, make_estimator, modelling_table, _preprocessor
from .schema import CHANNELS, BANDS


def _importance_one_fold(
    pipeline: Pipeline,
    features: pd.DataFrame,
    y: np.ndarray,
    task: str,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    baseline_pred = pipeline.predict(features)
    if task == "classification":
        probability = pipeline.predict_proba(features)[:, 1] if hasattr(pipeline, "predict_proba") else None
        baseline = _classification_fold(y, baseline_pred, probability)["balanced_accuracy"]
    else:
        baseline = _regression_fold(y, baseline_pred)["mae"]
    generator = np.random.default_rng(seed)
    rows = []
    for column in features.columns:
        drops = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[column] = generator.permutation(shuffled[column].to_numpy())
            predicted = pipeline.predict(shuffled)
            if task == "classification":
                probability = pipeline.predict_proba(shuffled)[:, 1] if hasattr(pipeline, "predict_proba") else None
                score = _classification_fold(y, predicted, probability)["balanced_accuracy"]
                drops.append(baseline - score)
            else:
                score = _regression_fold(y, predicted)["mae"]
                drops.append(score - baseline)
        rows.append({"feature": column, "importance": float(np.mean(drops))})
    return pd.DataFrame(rows)


def permutation_importance(frame: pd.DataFrame, task: str, kind: str, estimator_name: str, config: dict[str, Any]) -> pd.DataFrame:
    outcome = "correct" if task == "classification" else "response_time_s"
    table = modelling_table(frame).dropna(subset=[outcome, "participant_id"])
    features, numeric, categorical = feature_frame(table, kind)
    if features.empty:
        return pd.DataFrame(columns=["feature", "importance"])
    y = table[outcome].to_numpy()
    groups = table["participant_id"]
    splits, _ = iter_group_splits(groups, config["models"]["n_splits"], config["project"]["seed"])
    collected = []
    for fold, (train, test) in enumerate(splits):
        pipeline = Pipeline(
            [
                ("prepare", _preprocessor(numeric, categorical)),
                ("model", make_estimator(estimator_name, task, config)),
            ]
        )
        pipeline.fit(features.iloc[train], y[train])
        # Permutation is applied to the original columns. The pipeline transforms them.
        collected.append(
            _importance_one_fold(
                pipeline,
                features.iloc[test].reset_index(drop=True),
                y[test],
                task,
                config["models"]["permutation_repeats"],
                config["project"]["seed"] + fold,
            )
        )
    combined = pd.concat(collected, ignore_index=True)
    return combined.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)


def group_importance(importance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Sum importance by channel and by frequency band. Metadata features are kept apart."""
    if importance.empty:
        return {"channel": importance, "band": importance}
    frame = importance.copy()

    def channel_of(name: str) -> str:
        for channel in CHANNELS:
            if name.endswith(f"_{channel}") or f"_{channel}_" in name:
                return channel
        return "not_channel"

    def band_of(name: str) -> str:
        for band in BANDS:
            if band in name:
                return band
        return "not_band"

    frame["channel"] = frame["feature"].map(channel_of)
    frame["band"] = frame["feature"].map(band_of)
    return {
        "channel": frame.groupby("channel", as_index=False)["importance"].sum().sort_values("importance", ascending=False),
        "band": frame.groupby("band", as_index=False)["importance"].sum().sort_values("importance", ascending=False),
    }
