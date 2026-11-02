"""Participant-independent models of accuracy and response time.

Every row from one participant stays inside a single partition of each split.
Scaling is fit on the training partition only. Correctness is never a predictor
of correctness, and response time is never a predictor of response time.
Task metadata that defines the target is excluded. A metadata model may use
condition and difficulty, which are properties of the item, not of the answer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR

LEAKAGE_COLUMNS = {
    "participant_id",
    "trial_key",
    "question_number",
    "verdict",
    "correct",
    "response_time",
    "response_time_s",
    "response_time_unit_resolved",
    "routine_start",
    "routine_end",
    "routine_stamp",
    "routine_start_s",
    "routine_end_s",
    "window_start_s",
    "window_end_s",
    "window_duration_s",
    "window_includes_post_response",
    "feedback_time",
    "match_or_nomatch",
    "is_sample",
    "n_eeg_samples",
    "condition",
}


def eeg_feature_columns(frame: pd.DataFrame) -> list[str]:
    prefixes = ("vendor_", "recomputed_", "mean_", "std_", "variance_", "rms_", "range_", "skewness_", "kurtosis_", "hjorth_", "spectral_")
    return [
        column
        for column in frame.columns
        if column not in LEAKAGE_COLUMNS
        and column.startswith(prefixes)
        and "_relative" not in column
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("category", "difficulty") if column in frame.columns]


def make_estimator(name: str, task: str, config: dict[str, Any]):
    models = config["models"]
    if task == "classification":
        if name == "dummy":
            return DummyClassifier(strategy="most_frequent")
        if name == "logistic":
            return LogisticRegression(C=models["logistic_c"], max_iter=500, class_weight="balanced")
        if name == "svm":
            return SVC(C=models["svm_c"], kernel="rbf", probability=True, class_weight="balanced")
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=models["forest_trees"],
                min_samples_leaf=models["forest_min_leaf"],
                class_weight="balanced_subsample",
                random_state=config["project"]["seed"],
                n_jobs=1,
            )
    if name == "dummy":
        return DummyRegressor(strategy="median")
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=models["forest_trees"],
            min_samples_leaf=models["forest_min_leaf"],
            random_state=config["project"]["seed"],
            n_jobs=1,
        )
    if name == "svr":
        return SVR(C=models["svm_c"])
    raise ValueError(f"unknown estimator {name} for {task}")


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def feature_frame(frame: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric = eeg_feature_columns(frame) if kind in {"eeg", "combined"} else []
    categorical = metadata_columns(frame) if kind in {"metadata", "combined"} else []
    columns = numeric + categorical
    return frame[columns].copy(), numeric, categorical


def iter_group_splits(groups: pd.Series, n_splits: int, seed: int):
    """GroupKFold, reduced when there are fewer participants than requested folds."""
    n_groups = int(groups.nunique())
    splits = min(n_splits, n_groups)
    if splits < 2:
        raise ValueError("participant-independent validation needs at least two participants")
    splitter = GroupKFold(n_splits=splits)
    # GroupKFold ignores y. A constant vector keeps the call valid for regression.
    dummy = np.zeros(len(groups))
    return splitter.split(dummy, dummy, groups), splits


def modelling_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop sample items and trials whose EEG window could not exclude the response."""
    table = frame.copy()
    if "is_sample" in table.columns:
        table = table.loc[~table["is_sample"].fillna(False)]
    if "window_includes_post_response" in table.columns:
        table = table.loc[~table["window_includes_post_response"].fillna(True)]
    return table.reset_index(drop=True)
