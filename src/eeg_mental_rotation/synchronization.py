"""Align PsychoPy trials with EEG samples. Alignment is accepted only when the clocks overlap.

The paper states that Mind Monitor and iMotions were aligned with Unix time, and
it warns that the alignment may not be error-free. This module therefore reports
the overlap and refuses a trial-level table when coverage is below the configured
threshold. In that case the caller uses participant-by-condition summaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_validation import infer_response_time_seconds, parse_timestamps
from .schema import CONDITION_LABELS


def _normalise_verdict(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"correct", "true", "1", "yes"}:
        return 1.0
    if text in {"incorrect", "false", "0", "no", "wrong"}:
        return 0.0
    return float("nan")


def prepare_trials(behaviour: pd.DataFrame) -> pd.DataFrame:
    """Standardise one participant's PsychoPy log. Sample rows are flagged, not dropped."""
    frame = behaviour.copy()
    if "routine_start" in frame.columns:
        start = parse_timestamps(frame["routine_start"])
        frame["routine_start_s"] = start.to_numpy()
        frame.attrs["routine_start_unit"] = start.attrs.get("unit")
    if "routine_end" in frame.columns:
        end = parse_timestamps(frame["routine_end"])
        frame["routine_end_s"] = end.to_numpy()
        frame.attrs["routine_end_unit"] = end.attrs.get("unit")
    if {"routine_start_s", "routine_end_s", "response_time"}.issubset(frame.columns):
        duration = frame["routine_end_s"] - frame["routine_start_s"]
        frame["response_time_s"] = infer_response_time_seconds(frame["response_time"], duration).to_numpy()
        frame["response_time_unit_resolved"] = frame["response_time_s"].notna().any()
    if "verdict" in frame.columns:
        frame["correct"] = frame["verdict"].map(_normalise_verdict)
    if "category" in frame.columns:
        category = pd.to_numeric(frame["category"], errors="coerce")
        frame["category"] = category
        frame["condition"] = category.map(lambda value: CONDITION_LABELS.get(int(value)) if pd.notna(value) else np.nan)
    if "trial_key" in frame.columns or "question_number" in frame.columns:
        markers = pd.Series(False, index=frame.index)
        if "trial_key" in frame.columns:
            key = frame["trial_key"].astype(str).str.lower()
            markers = markers | key.str.contains("sample") | key.str.contains("spl")
        if "question_number" in frame.columns:
            question = frame["question_number"].astype(str).str.lower()
            markers = markers | question.str.contains("sample") | question.str.startswith("spl")
        frame["is_sample"] = markers
    else:
        frame["is_sample"] = False
    return frame


def trial_windows(trials: pd.DataFrame) -> pd.DataFrame:
    """EEG window is the response interval, excluding feedback when that interval is known.

    The window starts at ``routine_start``. It ends at start plus the resolved
    response time when that conversion succeeded, because feedback in category 2
    occurs after the response and must not enter a model of the response.
    Otherwise the window is the routine itself, and those trials are marked
    ``window_includes_post_response`` so they can be excluded from modelling.
    """
    frame = trials.copy()
    start = frame.get("routine_start_s", pd.Series(np.nan, index=frame.index))
    end = frame.get("routine_end_s", pd.Series(np.nan, index=frame.index))
    response = frame.get("response_time_s", pd.Series(np.nan, index=frame.index))
    resolved = response.notna() & start.notna() & (response > 0)
    frame["window_start_s"] = start
    frame["window_end_s"] = np.where(resolved, start + response, end)
    frame["window_includes_post_response"] = ~resolved
    frame["window_duration_s"] = frame["window_end_s"] - frame["window_start_s"]
    return frame


def synchronization_report(eeg_time_s: pd.Series, trials: pd.DataFrame, min_samples: int) -> dict[str, Any]:
    """Describe clock overlap. This is the evidence for or against trial-level alignment."""
    eeg_time = pd.to_numeric(eeg_time_s, errors="coerce").dropna()
    report: dict[str, Any] = {"n_trials": int(len(trials)), "n_eeg_samples": int(len(eeg_time))}
    if eeg_time.empty or "window_start_s" not in trials.columns:
        report["status"] = "insufficient_timestamps"
        report["coverage"] = 0.0
        return report
    eeg_start, eeg_end = float(eeg_time.min()), float(eeg_time.max())
    starts = pd.to_numeric(trials["window_start_s"], errors="coerce")
    ends = pd.to_numeric(trials["window_end_s"], errors="coerce")
    valid = starts.notna() & ends.notna() & (ends > starts)
    report["n_trials_with_window"] = int(valid.sum())
    report["eeg_span_s"] = [eeg_start, eeg_end]
    report["trial_span_s"] = [
        None if starts.dropna().empty else float(starts.min()),
        None if ends.dropna().empty else float(ends.max()),
    ]
    if valid.sum() == 0:
        report["status"] = "no_valid_windows"
        report["coverage"] = 0.0
        return report
    overlap = (starts[valid] < eeg_end) & (ends[valid] > eeg_start)
    counts = []
    eeg_values = eeg_time.to_numpy()
    for start, end in zip(starts[valid], ends[valid]):
        counts.append(int(np.sum((eeg_values >= start) & (eeg_values < end))))
    covered = int(np.sum(np.array(counts) >= min_samples))
    report["n_trials_overlapping_recording"] = int(overlap.sum())
    report["n_trials_with_enough_samples"] = covered
    report["coverage"] = covered / int(valid.sum())
    report["median_samples_per_trial"] = float(np.median(counts)) if counts else 0.0
    report["clock_offset_s"] = float(starts[valid].median() - eeg_time.median())
    return report


def question_key_coverage(eeg: pd.DataFrame, trials: pd.DataFrame) -> dict[str, Any]:
    """Fraction of scored trial keys that also appear in the EEG ``QuestionKey`` column.

    The published files contain this column. It is corroboration of the clock
    window, not a substitute for it: a key can be written on only some of the
    rows that fall inside the response interval.
    """
    if "question_key" not in eeg.columns or "trial_key" not in trials.columns:
        return {"status": "absent"}
    present = {str(value).strip() for value in eeg["question_key"].dropna().unique()}
    present.discard("")
    scored = trials.loc[~trials["is_sample"].fillna(False)] if "is_sample" in trials.columns else trials
    keys = scored["trial_key"].astype(str)
    matched = keys.isin(present)
    return {
        "status": "present",
        "n_scored_trials": int(len(keys)),
        "n_keys_found_in_eeg": int(matched.sum()),
        "fraction": float(matched.mean()) if len(keys) else 0.0,
    }


def assign_level(report: dict[str, Any], min_coverage: float) -> str:
    coverage = report.get("coverage") or 0.0
    if coverage >= min_coverage:
        return "trial"
    if report.get("n_trials_overlapping_recording", 0) > 0:
        return "partial"
    return "unaligned"


def condition_level_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Participant-by-condition behaviour, used when trial alignment is not supported."""
    frame = trials.copy()
    if "correct" not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby(["participant_id", "category"], dropna=False)
    summary = grouped.agg(
        n_trials=("correct", "size"),
        accuracy=("correct", "mean"),
        median_response_time_s=("response_time_s", "median") if "response_time_s" in frame.columns else ("correct", "size"),
    ).reset_index()
    return summary
