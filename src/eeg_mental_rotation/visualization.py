"""Publication-style figures. Each function writes one file and returns its path.

A figure is skipped, rather than drawn from invented numbers, when its table is empty.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .schema import CHANNELS

plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 160,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
sns.set_theme(style="whitegrid", context="paper")


def _save(figure: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    target = str(path)
    try:
        figure.savefig(target, bbox_inches="tight")
    except OSError:
        figure.savefig(target)
    plt.close(figure)
    return path


def plot_raw_eeg(frame: pd.DataFrame, path: Path, seconds: float = 10.0) -> Path | None:
    columns = [f"raw_{channel}" for channel in CHANNELS if f"raw_{channel}" in frame.columns]
    if not columns or "time_s" not in frame.columns:
        return None
    time = frame["time_s"].to_numpy(dtype=float)
    start = np.nanmin(time)
    window = frame.loc[time <= start + seconds]
    figure, axes = plt.subplots(len(columns), 1, figsize=(9, 1.6 * len(columns)), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, column in zip(axes, columns):
        axis.plot(window["time_s"] - start, pd.to_numeric(window[column], errors="coerce"), lw=0.7, color="#1f4e79")
        axis.set_ylabel(column.replace("raw_", ""))
    axes[-1].set_xlabel("Time from excerpt start (s)")
    figure.suptitle("Raw EEG excerpt")
    return _save(figure, path)


def plot_psd(frequencies: np.ndarray, power: np.ndarray, path: Path, channel: str) -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.semilogy(frequencies, power, color="#1f4e79")
    axis.set_xlim(0, min(45, float(np.nanmax(frequencies)) if len(frequencies) else 45))
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Power spectral density (signal units²/Hz)")
    axis.set_title(f"Welch PSD, channel {channel}")
    return _save(figure, path)


def plot_band_by_condition(summary: pd.DataFrame, path: Path, feature: str) -> Path | None:
    if summary.empty or feature not in summary.columns:
        return None
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=summary, x="category", y=feature, ax=axis, color="#d6e3f0")
    axis.set_xlabel("Condition (1 no feedback, 2 feedback, 3 time limit)")
    axis.set_ylabel(feature)
    axis.set_title(f"{feature} by experimental condition")
    return _save(figure, path)


def plot_correct_vs_incorrect(frame: pd.DataFrame, path: Path, feature: str) -> Path | None:
    if frame.empty or feature not in frame.columns or "correct" not in frame.columns:
        return None
    figure, axis = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=frame, x="correct", y=feature, ax=axis, inner="box", color="#d6e3f0")
    axis.set_xlabel("Response correct (0 no, 1 yes)")
    axis.set_ylabel(feature)
    axis.set_title(f"{feature} for correct and incorrect responses")
    return _save(figure, path)


def plot_feature_vs_rt(frame: pd.DataFrame, path: Path, feature: str) -> Path | None:
    if frame.empty or feature not in frame.columns or "response_time_s" not in frame.columns:
        return None
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.scatter(frame[feature], frame["response_time_s"], s=12, alpha=0.6, color="#1f4e79")
    axis.set_xlabel(feature)
    axis.set_ylabel("Response time (s)")
    axis.set_title(f"{feature} against response time")
    return _save(figure, path)


def plot_model_comparison(results: pd.DataFrame, path: Path, metric: str) -> Path | None:
    if results.empty or metric not in results.columns:
        return None
    figure, axis = plt.subplots(figsize=(8, 4))
    sns.barplot(data=results, x="estimator", y=metric, hue="kind", ax=axis)
    axis.set_xlabel("Model")
    axis.set_ylabel(metric)
    axis.set_title(f"Participant-independent {metric}")
    axis.legend(title="Features")
    return _save(figure, path)


def plot_confusion(matrix: list[list[int]], path: Path, title: str) -> Path:
    figure, axis = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(np.array(matrix), annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticklabels(["incorrect", "correct"])
    axis.set_yticklabels(["incorrect", "correct"])
    axis.set_title(title)
    return _save(figure, path)


def plot_roc(y_true: list[int], y_prob: list[float], path: Path) -> Path | None:
    if not y_prob or len(set(y_true)) < 2:
        return None
    from sklearn.metrics import roc_curve

    false_positive, true_positive, _ = roc_curve(y_true, y_prob)
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.plot(false_positive, true_positive, color="#1f4e79")
    axis.plot([0, 1], [0, 1], ls="--", color="grey", lw=0.8)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curve, pooled held-out predictions")
    return _save(figure, path)


def plot_predicted_vs_actual(y_true: list[float], y_pred: list[float], path: Path) -> Path | None:
    if not y_true:
        return None
    figure, axis = plt.subplots(figsize=(4.5, 4))
    axis.scatter(y_true, y_pred, s=14, alpha=0.65, color="#1f4e79")
    lo = min(min(y_true), min(y_pred))
    hi = max(max(y_true), max(y_pred))
    axis.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=0.8)
    axis.set_xlabel("Actual response time (s)")
    axis.set_ylabel("Predicted response time (s)")
    axis.set_title("Predicted versus actual response time")
    return _save(figure, path)


def plot_importance(importance: pd.DataFrame, path: Path, title: str, top: int = 15) -> Path | None:
    if importance.empty:
        return None
    shown = importance.head(top)
    figure, axis = plt.subplots(figsize=(7, 0.35 * len(shown) + 1.5))
    axis.barh(shown["feature"][::-1], shown["importance"][::-1], color="#1f4e79")
    axis.set_xlabel("Held-out permutation importance")
    axis.set_title(title)
    return _save(figure, path)
