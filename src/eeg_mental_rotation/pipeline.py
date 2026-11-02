"""Run the analysis on whatever recordings are present locally.

The function returns a status dictionary. It does not invent a sampling rate, a
synchronization, or a model score. When the archive has not been downloaded, the
status says so and no scientific result is written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config, resolve
from .data_loader import find_dataset_root, iter_zip_participants, list_participants, load_participant, locate_zip
from .data_validation import assess_behaviour, assess_eeg
from .feature_extraction import extract_trial_features
from .interpretation import group_importance, permutation_importance
from .model_evaluation import run_model_suite
from .signal_processing import preprocess_raw, supported_bands
from .statistical_analysis import associate_with_outcome, compare_conditions, correct_pvalues, descriptive_by_condition
from .synchronization import (
    assign_level,
    condition_level_summary,
    prepare_trials,
    question_key_coverage,
    synchronization_report,
    trial_windows,
)
from .visualization import (
    plot_band_by_condition,
    plot_confusion,
    plot_correct_vs_incorrect,
    plot_feature_vs_rt,
    plot_importance,
    plot_model_comparison,
    plot_predicted_vs_actual,
    plot_raw_eeg,
    plot_roc,
)
from . import reporting


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, pd.DataFrame):
        payload.to_csv(path, index=False)
    else:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def prepare_data(config: dict[str, Any]) -> tuple[str, Path] | None:
    """Locate recordings. A zip is read in place; it is not extracted.

    Uncompressed EEG tables are several gigabytes. Extraction is available as
    ``extract_recordings`` for machines that have the disk, and is not the default.
    """
    raw_dir = resolve(config, "paths.raw_dir")
    root = find_dataset_root(raw_dir)
    if root is not None:
        return ("directory", root)
    zip_path = locate_zip(raw_dir, config["dataset"]["zip_name"])
    if zip_path is None:
        return None
    return ("zip", zip_path)


def run(config_path: Path | None = None, max_participants: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    tables = resolve(config, "paths.tables_dir")
    figures = resolve(config, "paths.figures_dir")
    reports = resolve(config, "paths.reports_dir")
    source = prepare_data(config)
    if source is None:
        status = {
            "status": "no_data",
            "reason": "STData was not found. Run scripts/download_data.py and place the archive in data/raw.",
        }
        _write(tables / "pipeline_status.json", status)
        reporting.write_report(tables, reports / "research_report.md")
        return status

    print(f"reading {source[0]}: {source[1]}", flush=True)
    if source[0] == "directory":
        folders = list_participants(source[1])
        if max_participants is not None:
            folders = folders[:max_participants]
        recordings = (load_participant(folder) for folder in folders)
    else:
        recordings = iter_zip_participants(source[1], limit=max_participants)

    quality_rows = []
    trial_frames = []
    sync_rows = []
    inventory_rows = []
    first_eeg = None
    for loaded in recordings:
        print(f"participant {loaded['participant_id']}", flush=True)
        inventory_rows.append(
            {
                "participant_id": loaded["participant_id"],
                "eeg_rows": int(len(loaded["eeg"])) if "eeg" in loaded else None,
                "behaviour_rows": int(len(loaded["behaviour"])) if "behaviour" in loaded else None,
                "eeg_canonical": sorted(loaded.get("eeg_columns", {})),
                "behaviour_canonical": sorted(loaded.get("behaviour_columns", {})),
                "problems": loaded["problems"],
            }
        )
        if "eeg" not in loaded or "behaviour" not in loaded:
            quality_rows.append({"participant_id": loaded["participant_id"], "problems": loaded["problems"]})
            continue
        eeg_quality = assess_eeg(loaded["eeg"], config["signal"]["gap_intervals"], config["signal"]["artefact_robust_z"])
        behaviour_quality = assess_behaviour(loaded["behaviour"])
        quality_rows.append(
            {
                "participant_id": loaded["participant_id"],
                "eeg": eeg_quality,
                "behaviour": behaviour_quality,
            }
        )
        processed = preprocess_raw(
            loaded["eeg"],
            config["bands"],
            config["signal"]["line_noise_hz"],
            config["signal"]["bandpass_hz"],
            config["signal"]["min_raw_fs_hz"],
            config["signal"]["artefact_robust_z"],
            config["signal"]["max_interval_ratio"],
        )
        eeg = processed["frame"]
        if first_eeg is None:
            first_eeg = eeg
            plot_raw_eeg(eeg, figures / "raw_eeg_excerpt.png")
        trials = trial_windows(prepare_trials(loaded["behaviour"]))
        report = synchronization_report(eeg["time_s"], trials, config["sync"]["min_samples_per_trial"])
        level = assign_level(report, config["sync"]["min_trial_coverage"])
        report["participant_id"] = loaded["participant_id"]
        report["level"] = level
        report["question_key"] = question_key_coverage(eeg, trials)
        report["preprocessing"] = processed.get("decision", {"status": processed.get("status")})
        sync_rows.append(report)
        rate = (processed.get("decision") or {}).get("sampling", {}).get("sampling_rate_hz")
        use_raw = bool((processed.get("decision") or {}).get("raw_spectral_analysis"))
        bands = supported_bands(rate, config["bands"]) if use_raw else {}
        features = extract_trial_features(
            eeg,
            trials,
            rate,
            bands,
            use_raw,
            config["signal"]["welch_segment_s"],
            config["signal"]["welch_overlap"],
        )
        trial_frames.append(features)

    _write(
        tables / "dataset_inventory.json",
        {
            "source": source[0],
            "path": str(source[1]),
            "n_participant_folders": len(inventory_rows),
            "participants": inventory_rows,
        },
    )
    _write(tables / "quality_report.json", quality_rows)
    aligned_rows = [row for row in sync_rows if row.get("level") == "trial"]
    coverage_values = [row.get("coverage") or 0.0 for row in sync_rows]
    mean_coverage = float(sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0
    aligned_coverage = (
        float(sum(row.get("coverage") or 0.0 for row in aligned_rows) / len(aligned_rows)) if aligned_rows else 0.0
    )
    # A participant whose recording does not cover the task is excluded.
    # Their zero coverage is not evidence that an overlapping recording is misaligned.
    level = "trial" if len(aligned_rows) >= 2 else "participant_condition"
    sync_summary = {
        "level": level,
        "coverage": mean_coverage,
        "coverage_among_aligned": aligned_coverage,
        "n_participants_processed": len(sync_rows),
        "n_participants_aligned": len(aligned_rows),
        "aligned_participant_ids": [row["participant_id"] for row in aligned_rows],
        "unaligned_participant_ids": [row["participant_id"] for row in sync_rows if row.get("level") != "trial"],
        "reason": (
            f"{len(aligned_rows)} of {len(sync_rows)} participants have at least "
            f"{config['sync']['min_trial_coverage']:.0%} of trials inside the EEG recording. "
            "Models use only those participants. Recordings that do not overlap the task are listed and excluded."
            if level == "trial"
            else "Fewer than two participants met the trial-coverage rule. "
            "Condition-level behaviour is saved. EEG models are not fit."
        ),
        "participants": sync_rows,
    }
    _write(tables / "synchronization_report.json", sync_summary)

    if not trial_frames:
        status = {"status": "no_usable_participants", "level": level}
        _write(tables / "pipeline_status.json", status)
        reporting.write_report(tables, reports / "research_report.md")
        return status

    trials = pd.concat(trial_frames, ignore_index=True)
    if level == "trial":
        aligned_ids = {str(participant_id) for participant_id in sync_summary["aligned_participant_ids"]}
        trials = trials.loc[trials["participant_id"].astype(str).isin(aligned_ids)].reset_index(drop=True)
    trials = drop_raw_spectral_features(trials)
    trials.to_csv(tables / "trial_features.csv", index=False)
    return analyse_trials(trials, config, level, mean_coverage, len(aligned_rows))


_RAW_FEATURE_PREFIXES = (
    "recomputed_",
    "mean_",
    "std_",
    "variance_",
    "rms_",
    "range_",
    "skewness_",
    "kurtosis_",
    "hjorth_",
    "spectral_",
)


def drop_raw_spectral_features(trials: pd.DataFrame) -> pd.DataFrame:
    """Remove spectra and time-domain features computed on the raw column.

    The published export is not a uniform series. A filter or Welch spectrum on
    those rows is not an EEG spectrum, including for the occasional file whose
    interval ratio falls just under the cutoff. Vendor band-power means remain.
    """
    drop = [column for column in trials.columns if column.startswith(_RAW_FEATURE_PREFIXES)]
    return trials.drop(columns=drop)


def analyse_trials(
    trials: pd.DataFrame,
    config: dict[str, Any],
    level: str,
    mean_coverage: float,
    n_aligned: int,
) -> dict[str, Any]:
    """Statistics, models and figures for a trial table that has already been aligned."""
    tables = resolve(config, "paths.tables_dir")
    figures = resolve(config, "paths.figures_dir")
    reports = resolve(config, "paths.reports_dir")
    vendor_columns = [column for column in trials.columns if column.startswith("vendor_") and column.count("_") == 2]
    descriptive_by_condition(trials, vendor_columns[:8]).to_csv(tables / "descriptive_features.csv", index=False)

    if level != "trial":
        if "correct" in trials.columns:
            condition_level_summary(trials).to_csv(tables / "condition_summary.csv", index=False)
        status = {
            "status": "aligned_insufficient",
            "level": level,
            "coverage": mean_coverage,
            "n_participants_aligned": n_aligned,
        }
        _write(tables / "pipeline_status.json", status)
        reporting.write_report(tables, reports / "research_report.md")
        return status

    stats_rows = []
    for column in vendor_columns[:12]:
        stats_rows.append(compare_conditions(trials, column))
        if "correct" in trials.columns:
            stats_rows.append(associate_with_outcome(trials, column, "correct"))
        if "response_time_s" in trials.columns:
            stats_rows.append(associate_with_outcome(trials, column, "response_time_s"))
    _write(tables / "statistics.json", correct_pvalues(stats_rows))

    if vendor_columns:
        means = trials.groupby(["participant_id", "category"], dropna=False)[vendor_columns[0]].mean().reset_index()
        for call in (
            lambda: plot_band_by_condition(means, figures / "band_by_condition.png", vendor_columns[0]),
            lambda: plot_correct_vs_incorrect(trials, figures / "correct_vs_incorrect.png", vendor_columns[0]),
            lambda: plot_feature_vs_rt(trials, figures / "feature_vs_response_time.png", vendor_columns[0]),
        ):
            try:
                call()
            except OSError as exc:
                print(f"figure skipped: {exc}", flush=True)

    model_results = run_model_suite(trials, config)
    _write(tables / "model_results.json", model_results)
    comparison_rows = []
    for result in model_results:
        if result.get("status") != "ok":
            continue
        metric = "balanced_accuracy" if result["task"] == "classification" else "mae"
        comparison_rows.append(
            {
                "task": result["task"],
                "kind": result["kind"],
                "estimator": result["estimator"],
                metric: result["summary"][metric]["mean"],
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        comparison.to_csv(tables / "model_comparison.csv", index=False)
        if "balanced_accuracy" in comparison.columns:
            plot_model_comparison(
                comparison.loc[comparison["task"] == "classification"],
                figures / "classification_comparison.png",
                "balanced_accuracy",
            )
    for result in model_results:
        if result.get("status") != "ok":
            continue
        if result["task"] == "classification" and result["kind"] == "eeg" and result["estimator"] == "random_forest":
            plot_confusion(result["confusion_matrix"], figures / "confusion_matrix.png", "EEG random forest, pooled held-out")
            plot_roc(result["pooled_true"], result["pooled_prob"], figures / "roc_curve.png")
        if result["task"] == "regression" and result["kind"] == "eeg" and result["estimator"] == "random_forest":
            plot_predicted_vs_actual(result["pooled_true"], result["pooled_pred"], figures / "predicted_vs_actual.png")

    importance = permutation_importance(trials, "classification", "eeg", "random_forest", config)
    importance.to_csv(tables / "permutation_importance.csv", index=False)
    grouped = group_importance(importance)
    grouped["channel"].to_csv(tables / "importance_by_channel.csv", index=False)
    grouped["band"].to_csv(tables / "importance_by_band.csv", index=False)
    plot_importance(importance, figures / "feature_importance.png", "EEG features, held-out permutation importance")

    status = {
        "status": "ok",
        "level": level,
        "coverage": mean_coverage,
        "n_participants_aligned": n_aligned,
        "n_trials": int(len(trials)),
    }
    _write(tables / "pipeline_status.json", status)
    reporting.write_report(tables, reports / "research_report.md")
    return status
