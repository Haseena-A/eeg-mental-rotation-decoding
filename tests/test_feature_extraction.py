"""Feature names stay traceable to a channel and a source."""

from __future__ import annotations

import numpy as np
import pandas as pd

from eeg_mental_rotation.feature_extraction import features_for_window


def test_vendor_bands_are_averaged_inside_the_window():
    window = pd.DataFrame({"delta_AF7": [1.0, 3.0], "theta_AF7": [2.0, 2.0]})
    features = features_for_window(window, sampling_rate_hz=None, bands={}, use_raw_spectrum=False, segment_s=2, overlap=0.5)
    assert features["vendor_delta_AF7"] == 2.0
    assert "recomputed_delta_absolute_AF7" not in features


def test_raw_spectrum_uses_a_distinct_prefix():
    rate = 256.0
    time = np.arange(0, 4, 1 / rate)
    window = pd.DataFrame({"raw_AF7": np.sin(2 * np.pi * 10 * time)})
    features = features_for_window(
        window,
        sampling_rate_hz=rate,
        bands={"alpha": (8.0, 13.0), "beta": (13.0, 30.0)},
        use_raw_spectrum=True,
        segment_s=2.0,
        overlap=0.5,
    )
    assert features["recomputed_alpha_absolute_AF7"] > features["recomputed_beta_absolute_AF7"]
    assert "hjorth_activity_AF7" in features
