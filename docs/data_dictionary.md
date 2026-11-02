# Data dictionary

Names in the **Published** column are the attributes described by Ashwin et al. (2025). Names in the **Canonical** column are the headers this code uses after matching. The match ignores case, spaces and punctuation, so `match Ornomatch` and `matchOrnomatch` both become `match_or_nomatch`. The original spelling is stored in the inventory.

The paper says the EEG file has 40 attributes and describes 38 of them in prose. The downloaded `STData.zip` (Figshare file 51439640) was opened and the first participant table was read. That file is `{participant}/{participant}_EEG.csv` and has 41 columns. The three columns that the prose list does not name are `UnixTime`, `QuestionKey` and `Elements`. `UnixTime` is a numeric Unix timestamp in seconds. `QuestionKey` repeats the PsychoPy `Key`. `Elements` is present and is not defined in the paper, so it is stored and not interpreted. The same inspection is repeated for every participant by `scripts/validate_dataset.py`; this dictionary does not assume the other files are identical until that inventory is written.

`TimeStamp` in the inspected file is a datetime string. The loader prefers `UnixTime` when both exist. The PsychoPy log is `{participant}_PSY.csv`. In the inspected log, `ResponseTime` is in the same unit as `routineEnd - routineStart` (seconds). Sample items use keys such as `1spl1` and question numbers such as `spl1`. Verdict values are `CORRECT` and `INCORRECT`.

## EEG file (stem `EEG`)

| Published description | Canonical | Meaning in the paper |
|---|---|---|
| TimeStamp | timestamp | Datetime string in the inspected file. |
| UnixTime | unix_time | Numeric Unix time. In the inspected file the magnitude is seconds. |
| QuestionKey | question_key | Same identifier as the PsychoPy `Key`. |
| Elements | elements | Present in the file. The paper does not define it. |
| Delta/Theta/Alpha/Beta/Gamma × TP9, AF7, AF8, TP10 | `{band}_{channel}` | Vendor band power. Unit is not stated. |
| RAW × four channels | `raw_{channel}` | Raw EEG. Unit is not stated. |
| AUX_RIGHT | aux_right | Auxiliary sensor on the right side of the headband. |
| Accelerometer X, Y, Z | accelerometer_x/y/z | Head movement. |
| Gyro X, Y, Z | gyro_x/y/z | Rotation. |
| HeadBandOn | headband_on | Whether the headband was worn. |
| HSI × four channels | hsi_{channel} | Signal quality. Encoding is not defined in the paper. |
| Battery | battery | Headband battery. |

Channels: AF7, AF8 (forehead), TP9, TP10 (ears).

## PsychoPy file (stem `PSY`)

| Published name | Canonical | Meaning in the paper |
|---|---|---|
| Key | trial_key | Category and question, including sample items. |
| Category | category | 1, 2 or 3. |
| QuestionNumber | question_number | Includes sample questions. |
| match Ornomatch | match_or_nomatch | Participant choice: match or not. |
| Difficulty | difficulty | Easy, Medium or Difficult. |
| Verdict | verdict | Correct or incorrect. |
| ResponseTime | response_time | Time to respond. Unit is not stated; converted only when it agrees with the routine duration. |
| routineStart | routine_start | Start timestamp. |
| routineStamp | routine_stamp | Timestamp of the routine. |
| routineEnd | routine_end | End timestamp. |
| Cat2FeedbackTime | feedback_time | Feedback time in category 2. |

## Columns this analysis adds

| Column | Meaning |
|---|---|
| participant_id | Folder name in the archive. |
| time_s | EEG timestamp in Unix seconds. |
| correct | 1 if the verdict is correct, 0 if incorrect. |
| response_time_s | Response time in seconds, or missing when the unit cannot be resolved. |
| window_start_s, window_end_s | EEG interval used for features. |
| window_includes_post_response | True when the window could not be closed at the response. |
| is_sample | True when the trial key or question number contains "sample" or "spl". |
| vendor_* | Mean vendor band power inside the window. |
| recomputed_* | Welch band power from raw EEG, only when the sampling rate allows it. |

## Files this pipeline does not analyse

The same folders contain eye tracking, GSR, facial affect, DLOT, NASA-TLX and an exposure summary. They are described in the paper and are left in the archive. They are outside the question this repository asks.
