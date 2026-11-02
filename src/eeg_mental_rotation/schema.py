"""Column names published by Ashwin et al. (2025) and the aliases used to find them.

The paper describes attributes in prose. It does not print a header row. Loaders
therefore match headers by a normalised form (lowercase, alphanumeric only) and
record the original spelling. A missing required column is reported; it is never
invented.
"""

from __future__ import annotations

import re
from typing import Iterable

CHANNELS = ("AF7", "AF8", "TP9", "TP10")
BANDS = ("delta", "theta", "alpha", "beta", "gamma")

# Normalised header -> canonical name. Only names the paper actually describes.
EEG_ALIASES: dict[str, str] = {"timestamp": "timestamp"}
for _band in BANDS:
    for _channel in CHANNELS:
        EEG_ALIASES[f"{_band}{_channel}".lower()] = f"{_band}_{_channel}"
for _channel in CHANNELS:
    EEG_ALIASES[f"raw{_channel}".lower()] = f"raw_{_channel}"
    EEG_ALIASES[f"hsi{_channel}".lower()] = f"hsi_{_channel}"
EEG_ALIASES.update(
    {
        "unixtime": "unix_time",
        "questionkey": "question_key",
        "auxright": "aux_right",
        "accelerometerx": "accelerometer_x",
        "accelerometery": "accelerometer_y",
        "accelerometerz": "accelerometer_z",
        "gyrox": "gyro_x",
        "gyroy": "gyro_y",
        "gyroz": "gyro_z",
        "headbandon": "headband_on",
        "battery": "battery",
        "elements": "elements",
    }
)

# PsychoPy attributes named in the "Logdata from Psychopy" section.
BEHAVIOUR_ALIASES: dict[str, str] = {
    "key": "trial_key",
    "category": "category",
    "questionnumber": "question_number",
    "matchornomatch": "match_or_nomatch",
    "difficulty": "difficulty",
    "verdict": "verdict",
    "responsetime": "response_time",
    "routinestart": "routine_start",
    "routinestamp": "routine_stamp",
    "routineend": "routine_end",
    "cat2feedbacktime": "feedback_time",
}

CONDITION_LABELS = {
    1: "no time restriction, no feedback",
    2: "no time restriction, with feedback",
    3: "time restriction, no feedback",
}


def normalise_header(name: str) -> str:
    """Collapse a header to lowercase alphanumeric characters."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def canonical_columns(columns: Iterable[str], aliases: dict[str, str]) -> dict[str, str]:
    """Map original headers onto canonical names. Unknown headers are omitted."""
    found: dict[str, str] = {}
    for original in columns:
        canonical = aliases.get(normalise_header(original))
        if canonical and canonical not in found:
            found[canonical] = original
    return found
