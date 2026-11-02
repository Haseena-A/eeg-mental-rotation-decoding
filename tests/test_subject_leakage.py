"""The leakage test is separate because it is a scientific requirement, not a convenience check."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_mental_rotation.model_training import iter_group_splits, modelling_table


def test_every_partition_is_participant_pure():
    frame = pd.DataFrame(
        {
            "participant_id": np.repeat([f"P{i}" for i in range(10)], 5),
            "correct": 1,
        }
    )
    splits, _ = iter_group_splits(frame["participant_id"], n_splits=5, seed=0)
    for train, test in splits:
        overlap = set(frame.iloc[train]["participant_id"]) & set(frame.iloc[test]["participant_id"])
        assert overlap == set()


def test_post_response_windows_are_removed_before_modelling():
    frame = pd.DataFrame(
        {
            "participant_id": ["P1", "P1"],
            "is_sample": [False, False],
            "window_includes_post_response": [False, True],
            "correct": [1, 0],
        }
    )
    kept = modelling_table(frame)
    assert len(kept) == 1
    assert bool(kept.iloc[0]["window_includes_post_response"]) is False
