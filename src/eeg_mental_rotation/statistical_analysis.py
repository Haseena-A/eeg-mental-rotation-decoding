"""Statistics on participant-level summaries.

Trials from one person are not treated as independent biological replicates.
Group comparisons and correlations are computed on one summary per participant
(and, where the design requires it, per condition). Association is not causation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def cohens_d(left: np.ndarray, right: np.ndarray) -> float:
    left = left[np.isfinite(left)]
    right = right[np.isfinite(right)]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled = np.sqrt(((len(left) - 1) * np.var(left, ddof=1) + (len(right) - 1) * np.var(right, ddof=1)) / (len(left) + len(right) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(left) - np.mean(right)) / pooled)


def participant_feature_means(frame: pd.DataFrame, feature: str, by: list[str]) -> pd.DataFrame:
    columns = ["participant_id", *by, feature]
    present = frame.dropna(subset=[feature])
    return present.groupby(["participant_id", *by], dropna=False)[feature].mean().reset_index()


def compare_conditions(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    """Friedman test across the three conditions, using one mean per participant."""
    means = participant_feature_means(frame, feature, ["category"])
    wide = means.pivot(index="participant_id", columns="category", values=feature).dropna()
    result: dict[str, Any] = {
        "feature": feature,
        "n_participants_complete": int(len(wide)),
        "condition_means": {str(column): float(wide[column].mean()) for column in wide.columns},
    }
    if wide.shape[1] < 2 or len(wide) < 3:
        result["test"] = None
        result["reason"] = "fewer than three participants with every condition"
        return result
    statistic, pvalue = stats.friedmanchisquare(*[wide[column].to_numpy() for column in wide.columns])
    result["test"] = "friedman"
    result["statistic"] = float(statistic)
    result["p_value"] = float(pvalue)
    return result


def associate_with_outcome(frame: pd.DataFrame, feature: str, outcome: str) -> dict[str, Any]:
    """Spearman correlation of participant-mean feature with a participant-mean outcome."""
    means = (
        frame.dropna(subset=[feature, outcome])
        .groupby("participant_id")[[feature, outcome]]
        .mean()
        .dropna()
    )
    if len(means) < 5:
        return {"feature": feature, "outcome": outcome, "n": int(len(means)), "reason": "fewer than 5 participants"}
    coefficient, pvalue = stats.spearmanr(means[feature], means[outcome])
    return {
        "feature": feature,
        "outcome": outcome,
        "n": int(len(means)),
        "spearman_r": float(coefficient),
        "p_value": float(pvalue),
    }


def correct_pvalues(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Benjamini-Hochberg adjustment. Rows without a p-value are left unchanged."""
    indexed = [(index, row) for index, row in enumerate(rows) if row.get("p_value") is not None]
    if not indexed:
        return rows
    _, adjusted, _, _ = multipletests([row["p_value"] for _, row in indexed], method="fdr_bh")
    updated = [dict(row) for row in rows]
    for (index, _), pvalue in zip(indexed, adjusted):
        updated[index]["p_value_fdr"] = float(pvalue)
    return updated


def descriptive_by_condition(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for category, group in frame.groupby("category"):
        for column in columns:
            if column not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "category": category,
                    "feature": column,
                    "n_trials": int(len(values)),
                    "n_participants": int(group["participant_id"].nunique()),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "median": float(values.median()),
                }
            )
    return pd.DataFrame(rows)
