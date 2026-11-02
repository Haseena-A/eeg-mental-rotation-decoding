"""Quality checks for EEG recordings and PsychoPy logs.

Sampling rate and timestamp units are estimated from the values in each file.
The paper says Mind Monitor was aligned with Unix timestamps, but it does not
state the sampling rate or whether those timestamps are seconds or milliseconds.
Both are therefore measured and reported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schema import BANDS, CHANNELS

REQUIRED_EEG = ["timestamp", *[f"raw_{channel}" for channel in CHANNELS]]
REQUIRED_BEHAVIOUR = ["category", "verdict", "response_time", "routine_start", "routine_end"]


def time_column(frame: pd.DataFrame) -> str | None:
    """Prefer the numeric Unix column when the file has both a clock string and Unix time."""
    if "unix_time" in frame.columns:
        return "unix_time"
    if "timestamp" in frame.columns:
        return "timestamp"
    return None


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse a timestamp column into seconds since the Unix epoch, when possible.

    Numeric values are scaled by their magnitude: around 1e9 is seconds, 1e12
    milliseconds, 1e15 microseconds. Datetime strings are parsed as UTC. The
    returned series is float seconds. Values that cannot be parsed are NaN.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() > 0.8:
        finite = numeric[numeric.notna() & (numeric > 0)]
        if finite.empty:
            return numeric
        median = float(finite.median())
        if median > 1e14:
            scale = 1e6
            unit = "microseconds"
        elif median > 1e11:
            scale = 1e3
            unit = "milliseconds"
        else:
            scale = 1.0
            unit = "seconds"
        parsed = numeric / scale
        parsed.attrs["unit"] = unit
        return parsed

    parsed_dt = pd.to_datetime(values, errors="coerce", utc=True)
    seconds = parsed_dt.astype("int64") / 1e9
    seconds = seconds.where(parsed_dt.notna())
    seconds.attrs["unit"] = "datetime"
    return seconds


def estimate_sampling_rate(timestamps_s: pd.Series) -> dict[str, Any]:
    """Estimate the sampling rate from positive timestamp intervals."""
    values = timestamps_s.dropna().sort_values().to_numpy(dtype=float)
    if len(values) < 3:
        return {"n": int(len(values)), "sampling_rate_hz": None, "reason": "fewer than 3 timestamps"}
    intervals = np.diff(values)
    positive = intervals[intervals > 0]
    if len(positive) == 0:
        return {"n": int(len(values)), "sampling_rate_hz": None, "reason": "no positive intervals"}
    median_dt = float(np.median(positive))
    rate = 1.0 / median_dt if median_dt > 0 else None
    return {
        "n": int(len(values)),
        "median_interval_s": median_dt,
        "sampling_rate_hz": None if rate is None else float(rate),
        "interval_p05_s": float(np.quantile(positive, 0.05)),
        "interval_p95_s": float(np.quantile(positive, 0.95)),
        "n_nonpositive_intervals": int(np.sum(intervals <= 0)),
        "duplicate_timestamps": int(np.sum(intervals == 0)),
    }


def continuity_report(timestamps_s: pd.Series, gap_intervals: float) -> dict[str, Any]:
    """Count gaps that are large relative to the median interval."""
    rate = estimate_sampling_rate(timestamps_s)
    values = timestamps_s.dropna().sort_values().to_numpy(dtype=float)
    if rate.get("median_interval_s") is None or len(values) < 3:
        return {**rate, "n_gaps": None}
    intervals = np.diff(values)
    threshold = gap_intervals * float(rate["median_interval_s"])
    gaps = intervals[intervals > threshold]
    span = float(values[-1] - values[0]) if len(values) else 0.0
    covered = float(np.sum(intervals[intervals <= threshold]))
    return {
        **rate,
        "duration_s": span,
        "gap_threshold_s": threshold,
        "n_gaps": int(len(gaps)),
        "largest_gap_s": float(gaps.max()) if len(gaps) else 0.0,
        "fraction_continuous": covered / span if span > 0 else None,
    }


def missingness(frame: pd.DataFrame, columns: list[str]) -> dict[str, float]:
    present = [column for column in columns if column in frame.columns]
    if not present or frame.empty:
        return {}
    return {column: float(frame[column].isna().mean()) for column in present}


def robust_z(values: np.ndarray) -> np.ndarray:
    """Modified z-score using the median and the median absolute deviation."""
    finite = np.isfinite(values)
    scores = np.full(values.shape, np.nan)
    if finite.sum() < 5:
        return scores
    centre = np.median(values[finite])
    deviation = np.median(np.abs(values[finite] - centre))
    scale = 1.4826 * deviation
    if scale == 0:
        scores[finite] = 0.0
        return scores
    scores[finite] = (values[finite] - centre) / scale
    return scores


def assess_eeg(frame: pd.DataFrame, gap_intervals: float, artefact_z: float) -> dict[str, Any]:
    """Summarise one EEG recording. Does not modify the recording."""
    clock = time_column(frame)
    required = (["unix_time"] if clock == "unix_time" else ["timestamp"]) + [
        f"raw_{channel}" for channel in CHANNELS
    ]
    present = [column for column in required if column in frame.columns]
    missing_required = [column for column in required if column not in frame.columns]
    report: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "missing_required": missing_required,
        "missing_fraction": missingness(
            frame,
            ["timestamp", "unix_time", "question_key"]
            + [f"raw_{channel}" for channel in CHANNELS]
            + [f"{band}_{channel}" for band in BANDS for channel in CHANNELS]
            + [f"hsi_{channel}" for channel in CHANNELS],
        ),
    }
    if clock is None:
        report["sampling"] = {"sampling_rate_hz": None, "reason": "no timestamp column"}
        return report
    timestamps = parse_timestamps(frame[clock])
    report["timestamp_unit"] = timestamps.attrs.get("unit")
    report["sampling"] = continuity_report(timestamps, gap_intervals)
    artefact_fraction: dict[str, float] = {}
    for channel in CHANNELS:
        column = f"raw_{channel}"
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        scores = robust_z(values)
        finite = np.isfinite(scores)
        artefact_fraction[channel] = (
            float(np.mean(np.abs(scores[finite]) > artefact_z)) if finite.any() else float("nan")
        )
    report["raw_artefact_fraction"] = artefact_fraction
    if any(f"hsi_{channel}" in frame.columns for channel in CHANNELS):
        report["hsi_value_counts"] = {
            channel: {
                str(key): int(value)
                for key, value in frame[f"hsi_{channel}"].value_counts(dropna=False).head(8).items()
            }
            for channel in CHANNELS
            if f"hsi_{channel}" in frame.columns
        }
        report["hsi_note"] = (
            "The paper names HSI columns but does not define the numeric encoding. "
            "Counts are reported without labelling a value as good or bad."
        )
    report["vendor_bands_present"] = all(
        f"{band}_{channel}" in frame.columns for band in BANDS for channel in CHANNELS
    )
    report["raw_present"] = all(f"raw_{channel}" in frame.columns for channel in CHANNELS)
    report["usable_columns"] = present
    return report


def assess_behaviour(frame: pd.DataFrame) -> dict[str, Any]:
    missing_required = [column for column in REQUIRED_BEHAVIOUR if column not in frame.columns]
    report: dict[str, Any] = {
        "n_rows": int(len(frame)),
        "missing_required": missing_required,
        "missing_fraction": missingness(frame, list(frame.columns)),
    }
    if "verdict" in frame.columns:
        report["verdict_values"] = {
            str(key): int(value)
            for key, value in frame["verdict"].astype(str).str.strip().str.lower().value_counts().items()
        }
    if "difficulty" in frame.columns:
        report["difficulty_values"] = {
            str(key): int(value)
            for key, value in frame["difficulty"].astype(str).str.strip().value_counts().items()
        }
    if "category" in frame.columns:
        report["category_values"] = {
            str(key): int(value) for key, value in frame["category"].value_counts(dropna=False).items()
        }
    if "response_time" in frame.columns:
        numeric = pd.to_numeric(frame["response_time"], errors="coerce")
        report["response_time"] = {
            "n": int(numeric.notna().sum()),
            "median": None if numeric.notna().sum() == 0 else float(numeric.median()),
            "min": None if numeric.notna().sum() == 0 else float(numeric.min()),
            "max": None if numeric.notna().sum() == 0 else float(numeric.max()),
            "unit_note": (
                "The paper names ResponseTime but does not state the unit. "
                "The median is left in the recorded unit."
            ),
        }
    for column in ("routine_start", "routine_end"):
        if column in frame.columns:
            parsed = parse_timestamps(frame[column])
            report[f"{column}_unit"] = parsed.attrs.get("unit")
    if frame.duplicated().any():
        report["n_duplicate_rows"] = int(frame.duplicated().sum())
    return report


def infer_response_time_seconds(response_time: pd.Series, window_s: pd.Series) -> pd.Series:
    """Convert response time into seconds using the trial window as the reference.

    If the median response time is within an order of magnitude of the median
    routine duration, the recorded unit is already seconds. If it is about 1000
    times larger, the recorded unit is milliseconds. Anything else is left as
    NaN and the caller must not invent a conversion.
    """
    recorded = pd.to_numeric(response_time, errors="coerce")
    duration = pd.to_numeric(window_s, errors="coerce")
    both = pd.DataFrame({"recorded": recorded, "duration": duration}).dropna()
    both = both[both["duration"] > 0]
    if both.empty:
        return pd.Series(np.nan, index=response_time.index)
    ratio = float(np.median(both["recorded"] / both["duration"]))
    if 0.2 <= ratio <= 5:
        return recorded
    if 200 <= ratio <= 5000:
        return recorded / 1000.0
    return pd.Series(np.nan, index=response_time.index)
