"""Trial-level features from a response-locked EEG window.

Features are computed on the samples that fall inside the trial window. When the
raw sampling rate is too low for a spectrum, only the vendor band-power columns
are summarised. Recomputed band power is stored under a different prefix so the
two sources are never mixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .schema import BANDS, CHANNELS
from .signal_processing import band_power_from_psd, epoch_mask, welch_psd


def _finite(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def time_domain(values: np.ndarray) -> dict[str, float]:
    clean = _finite(values)
    if len(clean) < 3:
        return {}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)),
        "variance": float(np.var(clean, ddof=1)),
        "rms": float(np.sqrt(np.mean(clean**2))),
        "range": float(np.ptp(clean)),
        "skewness": float(stats.skew(clean, bias=False)),
        "kurtosis": float(stats.kurtosis(clean, bias=False, fisher=True)),
    }


def hjorth(values: np.ndarray) -> dict[str, float]:
    """Hjorth activity, mobility and complexity. Requires a regularly sampled window."""
    clean = _finite(values)
    if len(clean) < 8:
        return {}
    activity = float(np.var(clean, ddof=1))
    first = np.diff(clean)
    second = np.diff(first)
    var_first = float(np.var(first, ddof=1))
    var_second = float(np.var(second, ddof=1)) if len(second) > 2 else float("nan")
    mobility = float(np.sqrt(var_first / activity)) if activity > 0 else float("nan")
    mobility_first = float(np.sqrt(var_second / var_first)) if var_first > 0 else float("nan")
    complexity = mobility_first / mobility if mobility and np.isfinite(mobility) and mobility > 0 else float("nan")
    return {"hjorth_activity": activity, "hjorth_mobility": mobility, "hjorth_complexity": complexity}


def spectral_entropy(power: np.ndarray) -> float:
    positive = power[np.isfinite(power) & (power > 0)]
    if len(positive) < 2:
        return float("nan")
    probability = positive / positive.sum()
    entropy = -np.sum(probability * np.log(probability))
    return float(entropy / np.log(len(probability)))


def spectral_features(
    values: np.ndarray,
    sampling_rate_hz: float,
    bands: dict[str, tuple[float, float]],
    segment_s: float,
    overlap: float,
) -> dict[str, float]:
    if len(_finite(values)) < 16 or sampling_rate_hz <= 0:
        return {}
    frequencies, power = welch_psd(values, sampling_rate_hz, segment_s, overlap)
    total_low = min(edge[0] for edge in bands.values())
    total_high = max(edge[1] for edge in bands.values())
    total = band_power_from_psd(frequencies, power, total_low, total_high)
    features: dict[str, float] = {"spectral_entropy": spectral_entropy(power)}
    for name, (low, high) in bands.items():
        absolute = band_power_from_psd(frequencies, power, low, high)
        features[f"recomputed_{name}_absolute"] = absolute
        features[f"recomputed_{name}_relative"] = absolute / total if total and total > 0 else float("nan")
    if "theta" in bands and "alpha" in bands:
        theta = features.get("recomputed_theta_absolute")
        alpha = features.get("recomputed_alpha_absolute")
        features["recomputed_theta_alpha_ratio"] = (
            theta / alpha if alpha and np.isfinite(alpha) and alpha > 0 else float("nan")
        )
    if "beta" in bands and "alpha" in bands:
        beta = features.get("recomputed_beta_absolute")
        alpha = features.get("recomputed_alpha_absolute")
        features["recomputed_beta_alpha_ratio"] = (
            beta / alpha if alpha and np.isfinite(alpha) and alpha > 0 else float("nan")
        )
    return features


def vendor_band_summary(window: pd.DataFrame) -> dict[str, float]:
    """Mean of the Mind Monitor band columns inside the trial. Units are as recorded."""
    features: dict[str, float] = {}
    for channel in CHANNELS:
        powers = []
        for band in BANDS:
            column = f"{band}_{channel}"
            if column not in window.columns:
                continue
            values = pd.to_numeric(window[column], errors="coerce").to_numpy(dtype=float)
            clean = _finite(values)
            if len(clean) == 0:
                continue
            features[f"vendor_{band}_{channel}"] = float(np.mean(clean))
            powers.append(float(np.mean(clean)))
        total = float(np.sum(powers)) if powers else float("nan")
        for band in BANDS:
            key = f"vendor_{band}_{channel}"
            if key in features and np.isfinite(total) and total != 0:
                features[f"vendor_{band}_{channel}_relative"] = features[key] / total
    return features


def features_for_window(
    window: pd.DataFrame,
    sampling_rate_hz: float | None,
    bands: dict[str, tuple[float, float]],
    use_raw_spectrum: bool,
    segment_s: float,
    overlap: float,
) -> dict[str, float]:
    """Features for one trial window. An empty window returns an empty mapping."""
    if window.empty:
        return {}
    features = vendor_band_summary(window)
    features["n_eeg_samples"] = float(len(window))
    if not use_raw_spectrum or sampling_rate_hz is None:
        return features
    for channel in CHANNELS:
        column = f"raw_{channel}_clean" if f"raw_{channel}_clean" in window.columns else f"raw_{channel}"
        if column not in window.columns:
            continue
        values = pd.to_numeric(window[column], errors="coerce").to_numpy(dtype=float)
        if f"raw_{channel}_artefact" in window.columns:
            artefact = window[f"raw_{channel}_artefact"].to_numpy(dtype=bool)
            values = values.copy()
            values[artefact] = np.nan
        for name, value in time_domain(values).items():
            features[f"{name}_{channel}"] = value
        for name, value in hjorth(values).items():
            features[f"{name}_{channel}"] = value
        for name, value in spectral_features(values, sampling_rate_hz, bands, segment_s, overlap).items():
            features[f"{name}_{channel}"] = value
    return features


def extract_trial_features(
    eeg: pd.DataFrame,
    trials: pd.DataFrame,
    sampling_rate_hz: float | None,
    bands: dict[str, tuple[float, float]],
    use_raw_spectrum: bool,
    segment_s: float,
    overlap: float,
) -> pd.DataFrame:
    """One feature row per trial. Trials without a valid window are kept, with empty features."""
    if "time_s" not in eeg.columns:
        raise ValueError("EEG frame has no time_s column; run preprocessing first")
    time_s = eeg["time_s"].to_numpy(dtype=float)
    vendor_columns = [
        f"{band}_{channel}"
        for channel in CHANNELS
        for band in BANDS
        if f"{band}_{channel}" in eeg.columns
    ]
    vendor_matrix = (
        eeg[vendor_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if vendor_columns
        else None
    )
    rows: list[dict[str, Any]] = []
    for record in trials.to_dict(orient="records"):
        start = record.get("window_start_s")
        end = record.get("window_end_s")
        mask = epoch_mask(
            time_s,
            float(start) if start is not None else np.nan,
            float(end) if end is not None else np.nan,
        )
        if not use_raw_spectrum or sampling_rate_hz is None:
            features = _vendor_from_matrix(vendor_matrix, vendor_columns, mask)
        else:
            features = features_for_window(
                eeg.loc[mask], sampling_rate_hz, bands, True, segment_s, overlap
            )
        record.update(features)
        rows.append(record)
    return pd.DataFrame(rows)


def _vendor_from_matrix(
    matrix: np.ndarray | None,
    columns: list[str],
    mask: np.ndarray,
) -> dict[str, float]:
    """Time-weighted mean of vendor band columns. Held samples stay held, which is the recorded series."""
    features: dict[str, float] = {"n_eeg_samples": float(mask.sum())}
    if matrix is None or not mask.any():
        return features if mask.any() else {}
    with np.errstate(invalid="ignore"):
        means = np.nanmean(matrix[mask], axis=0)
    by_channel: dict[str, list[tuple[str, float]]] = {channel: [] for channel in CHANNELS}
    for column, value in zip(columns, means):
        band, channel = column.split("_", 1)
        if not np.isfinite(value):
            continue
        features[f"vendor_{band}_{channel}"] = float(value)
        by_channel[channel].append((band, float(value)))
    for channel, pairs in by_channel.items():
        total = float(sum(value for _, value in pairs))
        if total == 0:
            continue
        for band, value in pairs:
            features[f"vendor_{band}_{channel}_relative"] = value / total
    return features
