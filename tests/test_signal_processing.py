"""Signal-processing tests on synthetic oscillations of a known frequency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_mental_rotation.signal_processing import band_power_from_psd, supported_bands, welch_psd


def test_gamma_is_dropped_when_nyquist_cannot_represent_it():
    bands = supported_bands(40.0, {"delta": [0.5, 4], "alpha": [8, 13], "gamma": [30, 45]})
    assert "alpha" in bands
    assert "gamma" not in bands


def test_ten_hertz_sine_lands_in_alpha():
    rate = 256.0
    time = np.arange(0, 8, 1 / rate)
    values = np.sin(2 * np.pi * 10 * time)
    frequencies, power = welch_psd(values, rate, segment_s=2.0, overlap=0.5)
    alpha = band_power_from_psd(frequencies, power, 8, 13)
    beta = band_power_from_psd(frequencies, power, 13, 30)
    assert alpha > beta * 5


def test_irregular_grid_is_not_treated_as_a_uniform_series():
    from eeg_mental_rotation.signal_processing import preprocess_raw

    intervals = np.full(200, 0.001)
    intervals[::10] = 0.03
    frame = pd.DataFrame(
        {
            "unix_time": 1_700_000_000 + np.cumsum(intervals),
            "raw_AF7": np.random.default_rng(0).normal(size=200),
        }
    )
    result = preprocess_raw(frame, {"alpha": [8, 13]}, 50, [0.5, 45], 64, 6)
    assert result["status"] == "irregular_sampling"
    assert "raw_AF7_clean" not in result["frame"].columns


def test_slow_export_is_not_filtered_as_if_it_were_raw_eeg():
    from eeg_mental_rotation.signal_processing import preprocess_raw

    frame = pd.DataFrame(
        {
            "timestamp": np.arange(1_700_000_000, 1_700_000_020, 1.0),
            "raw_AF7": np.random.default_rng(0).normal(size=20),
            "delta_AF7": np.ones(20),
        }
    )
    result = preprocess_raw(frame, {"delta": [0.5, 4], "gamma": [30, 45]}, 50, [0.5, 45], 64, 6)
    assert result["status"] == "vendor_bands_only"
    assert "raw_AF7_clean" not in result["frame"].columns
