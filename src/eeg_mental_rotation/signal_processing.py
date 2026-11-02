"""EEG preprocessing that follows the sampling rate actually present in a file.

Mind Monitor can export band power at about 1 Hz or raw samples at the headset
rate. The paper does not say which export was used. This module measures the
rate and refuses spectral analysis when the Nyquist limit does not support it.
Vendor band-power columns are never filtered again.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import signal

from .data_validation import estimate_sampling_rate, parse_timestamps, robust_z, time_column
from .schema import BANDS, CHANNELS


def supported_bands(sampling_rate_hz: float | None, bands: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    """Keep bands whose upper edge is below the Nyquist frequency."""
    if sampling_rate_hz is None or sampling_rate_hz <= 0:
        return {}
    nyquist = sampling_rate_hz / 2.0
    kept = {}
    for name, edges in bands.items():
        low, high = float(edges[0]), float(edges[1])
        if high < nyquist and low < high:
            kept[name] = (low, min(high, nyquist * 0.95))
    return kept


def bandpass(values: np.ndarray, sampling_rate_hz: float, low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase Butterworth band-pass. Samples that are NaN are interpolated only inside the call."""
    finite = np.isfinite(values)
    if finite.sum() < 16 or high_hz >= sampling_rate_hz / 2:
        return values.astype(float)
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    sos = signal.butter(4, [low_hz, high_hz], btype="bandpass", fs=sampling_rate_hz, output="sos")
    filtered = signal.sosfiltfilt(sos, filled)
    filtered = filtered.astype(float)
    filtered[~finite] = np.nan
    return filtered


def notch(values: np.ndarray, sampling_rate_hz: float, line_hz: float) -> np.ndarray:
    """Notch the mains frequency when the sampling rate can represent it."""
    if line_hz <= 0 or line_hz >= sampling_rate_hz / 2:
        return values.astype(float)
    finite = np.isfinite(values)
    if finite.sum() < 16:
        return values.astype(float)
    filled = pd.Series(values).interpolate(limit_direction="both").to_numpy(dtype=float)
    quality = 30.0
    b, a = signal.iirnotch(line_hz, quality, fs=sampling_rate_hz)
    filtered = signal.filtfilt(b, a, filled).astype(float)
    filtered[~finite] = np.nan
    return filtered


def welch_psd(values: np.ndarray, sampling_rate_hz: float, segment_s: float, overlap: float) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density. Returns frequency in Hz and power in signal-units squared per Hz."""
    clean = values[np.isfinite(values)]
    nperseg = int(segment_s * sampling_rate_hz)
    nperseg = max(8, min(nperseg, len(clean)))
    noverlap = int(nperseg * overlap)
    frequencies, power = signal.welch(
        clean,
        fs=sampling_rate_hz,
        nperseg=nperseg,
        noverlap=min(noverlap, nperseg - 1),
        detrend="constant",
        scaling="density",
    )
    return frequencies, power


def band_power_from_psd(frequencies: np.ndarray, power: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (frequencies >= low_hz) & (frequencies < high_hz)
    if mask.sum() == 0:
        return float("nan")
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(power[mask], frequencies[mask]))


def preprocess_raw(
    frame: pd.DataFrame,
    bands: dict[str, list[float]],
    line_hz: float,
    bandpass_hz: list[float],
    min_raw_fs_hz: float,
    artefact_z: float,
    max_interval_ratio: float = 5.0,
) -> dict[str, Any]:
    """Filter raw channels when the sampling rate supports it, and mark artefacts.

    Returns the sampling decision and a copy of the frame with ``raw_<channel>_clean``
    columns. Vendor band columns are copied through untouched.
    """
    clock = time_column(frame)
    if clock is None:
        return {"status": "no_timestamp", "frame": frame.copy()}
    timestamps = parse_timestamps(frame[clock])
    sampling = estimate_sampling_rate(timestamps)
    rate = sampling.get("sampling_rate_hz")
    output = frame.copy()
    output["time_s"] = timestamps.to_numpy()
    decision: dict[str, Any] = {
        "time_column": clock,
        "timestamp_unit": timestamps.attrs.get("unit"),
        "sampling": sampling,
        "raw_spectral_analysis": False,
        "reason": "",
    }
    median_dt = sampling.get("median_interval_s")
    interval_p95 = sampling.get("interval_p95_s")
    irregular = (
        median_dt is not None
        and interval_p95 is not None
        and median_dt > 0
        and interval_p95 > max_interval_ratio * median_dt
    )
    if rate is None or rate < min_raw_fs_hz or irregular:
        if irregular:
            decision["reason"] = (
                f"median interval is {median_dt:.6f} s (about {rate:.1f} Hz) but the 95th percentile "
                f"interval is {interval_p95:.6f} s, {interval_p95 / median_dt:.1f} times the median. "
                "The export is not a uniform series, so a band-pass filter and Welch's method would "
                "treat gaps and held values as samples. Raw spectral analysis is not run. "
                "Vendor band-power columns are retained and are not filtered."
            )
            decision["status"] = "irregular_sampling"
            status = "irregular_sampling"
        else:
            decision["reason"] = (
                f"median sampling rate is {rate} Hz, below {min_raw_fs_hz} Hz. "
                "Raw spectral analysis is not identifiable at this rate. "
                "Vendor band-power columns are retained and are not filtered."
            )
            decision["status"] = "vendor_bands_only"
            status = "vendor_bands_only"
        return {"status": status, "decision": decision, "frame": output}

    usable = supported_bands(rate, bands)
    low, high = float(bandpass_hz[0]), min(float(bandpass_hz[1]), rate / 2 * 0.95)
    decision["raw_spectral_analysis"] = True
    decision["bands_supported"] = {name: list(edges) for name, edges in usable.items()}
    decision["bands_omitted"] = [name for name in bands if name not in usable]
    decision["reason"] = (
        f"median sampling rate is {rate:.2f} Hz. "
        f"A {low:.1f}-{high:.1f} Hz band-pass and a {line_hz:.0f} Hz notch are applied to raw channels only."
    )
    decision["status"] = "raw_filtered"
    for channel in CHANNELS:
        column = f"raw_{channel}"
        if column not in output.columns:
            continue
        values = pd.to_numeric(output[column], errors="coerce").to_numpy(dtype=float)
        filtered = bandpass(values, rate, low, high)
        filtered = notch(filtered, rate, line_hz)
        output[f"{column}_clean"] = filtered
        scores = robust_z(filtered)
        output[f"{column}_artefact"] = np.abs(scores) > artefact_z
    return {"status": "raw_filtered", "decision": decision, "frame": output}


def epoch_mask(time_s: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Boolean mask for samples inside a half-open ``[start, end)`` window."""
    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.zeros(len(time_s), dtype=bool)
    return (time_s >= start_s) & (time_s < end_s)
