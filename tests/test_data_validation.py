"""Tests use synthetic recordings. They do not stand in for the published dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_mental_rotation.data_loader import load_behaviour, load_eeg
from eeg_mental_rotation.data_validation import assess_eeg, continuity_report, parse_timestamps
from eeg_mental_rotation.schema import canonical_columns, EEG_ALIASES


def test_published_eeg_headers_are_recognised():
    headers = [
        "TimeStamp",
        "Delta_TP9", "Delta_AF7", "Delta_AF8", "Delta_TP10",
        "RAW_AF7", "HSI_TP10", "Accelerometer_X", "Gyro_Z", "HeadBandOn", "Battery",
    ]
    found = canonical_columns(headers, EEG_ALIASES)
    assert found["timestamp"] == "TimeStamp"
    assert found["delta_TP9"] == "Delta_TP9"
    assert found["raw_AF7"] == "RAW_AF7"
    assert found["hsi_TP10"] == "HSI_TP10"


def test_loader_renames_real_spellings(tmp_path: Path):
    path = tmp_path / "EEG.csv"
    pd.DataFrame(
        {
            "TimeStamp": [1_700_000_000.0, 1_700_000_000.004],
            "RAW_AF7": [0.1, 0.2],
            "Delta_AF7": [1.0, 1.1],
        }
    ).to_csv(path, index=False)
    frame, mapping = load_eeg(path)
    assert "timestamp" in frame.columns
    assert mapping["raw_AF7"] == "RAW_AF7"


def test_behaviour_aliases(tmp_path: Path):
    path = tmp_path / "PSY.csv"
    pd.DataFrame(
        {
            "Category": [1],
            "QuestionNumber": [1],
            "Verdict": ["Correct"],
            "ResponseTime": [2.5],
            "routineStart": [1_700_000_000],
            "routineEnd": [1_700_000_003],
            "Difficulty": ["Easy"],
            "match Ornomatch": ["match"],
        }
    ).to_csv(path, index=False)
    frame, mapping = load_behaviour(path)
    assert mapping["verdict"] == "Verdict"
    assert "match_or_nomatch" in frame.columns


def test_unix_milliseconds_are_scaled():
    parsed = parse_timestamps(pd.Series([1_700_000_000_000, 1_700_000_001_000]))
    assert parsed.attrs["unit"] == "milliseconds"
    assert parsed.iloc[0] == pytest.approx(1_700_000_000.0)


def test_duplicate_timestamps_are_counted():
    times = pd.Series(np.array([0.0, 0.004, 0.004, 0.008, 0.012]))
    report = continuity_report(times, gap_intervals=5)
    assert report["duplicate_timestamps"] == 1
    assert report["sampling_rate_hz"] == pytest.approx(250.0, rel=0.01)


def test_missing_raw_channel_is_reported():
    frame = pd.DataFrame({"timestamp": [1.0, 2.0], "raw_AF7": [0.0, 1.0]})
    report = assess_eeg(frame, gap_intervals=5, artefact_z=6)
    assert "raw_AF8" in report["missing_required"]
