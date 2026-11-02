"""A participant must not appear on both sides of a validation split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_mental_rotation.config import load_config
from eeg_mental_rotation.model_evaluation import evaluate_classification
from eeg_mental_rotation.model_training import eeg_feature_columns, iter_group_splits


def _table(n_participants: int = 6, rows_each: int = 8) -> pd.DataFrame:
    generator = np.random.default_rng(1)
    rows = []
    for person in range(n_participants):
        for _ in range(rows_each):
            signal = generator.normal(loc=person % 2)
            rows.append(
                {
                    "participant_id": f"P{person}",
                    "correct": float(person % 2),
                    "response_time_s": 1.5 + 0.2 * (person % 2) + abs(generator.normal(scale=0.05)),
                    "category": 1 + (person % 3),
                    "difficulty": "Easy",
                    "vendor_alpha_AF7": signal,
                    "window_includes_post_response": False,
                    "is_sample": False,
                }
            )
    return pd.DataFrame(rows)


def test_group_splits_do_not_share_participants():
    frame = _table()
    splits, n_splits = iter_group_splits(frame["participant_id"], n_splits=5, seed=1)
    seen = []
    for train, test in splits:
        train_ids = set(frame.iloc[train]["participant_id"])
        test_ids = set(frame.iloc[test]["participant_id"])
        assert train_ids.isdisjoint(test_ids)
        seen.append(test_ids)
    assert n_splits >= 2
    assert set().union(*seen) == set(frame["participant_id"])


def test_correctness_is_not_an_eeg_feature():
    columns = eeg_feature_columns(_table())
    assert columns == ["vendor_alpha_AF7"]
    assert "correct" not in columns
    assert "response_time_s" not in columns


def test_classification_pipeline_returns_fold_metrics():
    result = evaluate_classification(_table(), "eeg", "logistic", load_config())
    assert result["status"] == "ok"
    assert len(result["folds"]) >= 2
    assert "balanced_accuracy" in result["summary"]
    for fold in result["folds"]:
        assert fold["n_test_participants"] >= 1
