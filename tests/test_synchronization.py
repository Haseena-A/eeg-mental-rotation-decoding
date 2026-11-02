"""Synchronization must fail closed when the clocks do not overlap."""

from __future__ import annotations

import pandas as pd

from eeg_mental_rotation.synchronization import assign_level, prepare_trials, synchronization_report, trial_windows


def _behaviour():
    return pd.DataFrame(
        {
            "participant_id": ["P1", "P1"],
            "category": [1, 2],
            "verdict": ["Correct", "Incorrect"],
            "response_time": [2.0, 3.0],
            "routine_start": [1_000.0, 2_000.0],
            "routine_end": [1_002.0, 2_003.0],
            "trial_key": ["sample-1", "1-2"],
            "difficulty": ["Easy", "Difficult"],
        }
    )


def test_sample_items_are_marked_when_the_key_uses_the_published_spl_prefix():
    frame = _behaviour()
    frame.loc[0, "trial_key"] = "1spl1"
    frame.loc[0, "question_number"] = "spl1"
    trials = prepare_trials(frame)
    assert bool(trials.loc[0, "is_sample"])
    assert not bool(trials.loc[1, "is_sample"])


def test_response_window_stops_at_the_response():
    trials = trial_windows(prepare_trials(_behaviour()))
    assert trials.loc[0, "correct"] == 1
    assert trials.loc[0, "is_sample"]
    assert trials.loc[0, "window_end_s"] == 1_002.0
    assert not trials.loc[0, "window_includes_post_response"]


def test_millisecond_response_time_is_converted_when_the_window_agrees():
    frame = _behaviour()
    frame.loc[0, "response_time"] = 2000
    trials = trial_windows(prepare_trials(frame))
    assert abs(trials.loc[0, "response_time_s"] - 2.0) < 1e-6


def test_disjoint_clocks_are_not_called_aligned():
    trials = trial_windows(prepare_trials(_behaviour()))
    eeg_time = pd.Series([10.0, 11.0, 12.0])
    report = synchronization_report(eeg_time, trials, min_samples=1)
    assert assign_level(report, min_coverage=0.8) == "unaligned"
    assert report["coverage"] == 0.0


def test_overlapping_clocks_are_accepted():
    trials = trial_windows(prepare_trials(_behaviour()))
    eeg_time = pd.Series([1_000.0 + index * 0.25 for index in range(20)] + [2_000.0 + index * 0.25 for index in range(20)])
    report = synchronization_report(eeg_time, trials, min_samples=3)
    assert assign_level(report, min_coverage=0.8) == "trial"
